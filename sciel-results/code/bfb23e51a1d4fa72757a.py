"""
Strategy: Smooth Epigraph Formulation + SLSQP + High-Precision Polish.

Instead of heuristically forcing points apart or minimizing a nonsmooth penalty,
we formulate the problem as maximizing a continuous variable 't' (the radius)
subject to smooth constraints:
1. Pairwise distances: log-sum-exp approximation of min(d_ij) >= 2t
2. Wall distances: log-sum-exp approximation of min(d_iw) >= t

This is solved using SLSQP which handles nonlinear constraints well.
We then apply a final polish using Nelder-Mead on the actual nonsmooth objective
to refine the solution further, as SLSQP may stop at a point where the gradient
of the smooth surrogate is zero but the true max-min can still be improved
by moving along the flat ridge.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
import random

N = 50

def compute_radius(centres):
    """Compute the radius for a given set of centres."""
    if centres.shape != (N, 2):
        return 0.0
    if np.any(~np.isfinite(centres)):
        return 0.0
    # Check boundaries: all coords must be in [-0.5, 0.5]
    if np.any(np.abs(centres) > 0.5 + 1e-9):
        return 0.0
    # Compute min distance between pairs
    dists = pdist(centres, metric='euclidean')
    min_pair_dist = np.min(dists)
    # Compute min distance to boundary
    dist_to_walls = np.array([
        0.5 - np.abs(centres[:, 0]),  # left/right
        0.5 - np.abs(centres[:, 1])   # top/bottom
    ])
    min_wall_dist = np.min(dist_to_walls)
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return max(0.0, r)

def _smooth_min(x, beta=200):
    """
    Soft minimum using log-sum-exp.
    min(x) ≈ -1/beta * log(sum(exp(-beta * x)))
    """
    if len(x) == 0:
        return 0
    min_val = np.min(x)
    if min_val < -100: # avoid underflow
        return min_val
    return -np.log(np.sum(np.exp(-beta * (x - min_val)))) / beta + min_val

def _objective_and_constraints(x, beta=100):
    """
    Objective: Maximize t (so minimize -t).
    Variables: x = [c_1x, c_1y, ..., c_Nx, c_Ny, t]
    Constraints:
    1. Pairwise distances >= 2t (smoothed)
    2. Wall distances >= t (smoothed)
    """
    c = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    # Objective: minimize -t
    obj = -t
    
    # Constraints are defined as g(x) >= 0
    cons = []
    
    # 1. Pairwise constraints
    # We want min(d_ij) >= 2t
    # Smoothed: soft_min(d_ij) - 2t >= -margin
    # Using log-sum-exp for min(d_ij)
    dists = pdist(c, metric='euclidean')
    min_d_pair = _smooth_min(dists, beta)
    cons.append({'type': 'ineq', 'fun': lambda x, md=md_pair: md - 2*x[2*N], 
                 'jac': lambda x: None}) # We will provide analytic jac below if needed, 
                                         # but SLSQP can handle numerical jac. 
                                         # For speed, let's use numerical jac for constraints 
                                         # and analytic for obj? No, SLSQP uses one jac for all.
                                         # Let's just use numerical jac for simplicity and robustness,
                                         # but it might be slow. 
                                         # Actually, providing jac for constraints is important for SLSQP.
                                         # Let's compute gradients analytically for the soft min.
    
    # 2. Wall constraints
    # We want min(d_iw) >= t
    wall_dists = np.concatenate([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1]
    ])
    min_d_wall = _smooth_min(wall_dists, beta)
    cons.append({'type': 'ineq', 'fun': lambda x, md=md_wall: md - x[2*N], 
                 'jac': lambda x: None})
    
    # We need to provide the Jacobians properly.
    # Let's restructure to return obj and a list of constraint dicts with jacs.
    
    return obj, cons, c, t, dists, wall_dists, min_d_pair, min_d_wall

def _solve_epigraph(x0, beta=100, maxiter=200):
    """
    Solve the epigraph formulation using SLSQP.
    """
    # x0 shape: (2*N + 1,)
    
    def objective(x):
        t = x[2*N]
        return -t # minimize -t
    
    def constraints(x):
        c = x[:2*N].reshape(N, 2)
        t = x[2*N]
        
        # Pairwise distances
        dists = pdist(c, metric='euclidean')
        
        # Wall distances
        wall_dists = np.concatenate([
            0.5 - c[:, 0],
            0.5 + c[:, 0],
            0.5 - c[:, 1],
            0.5 + c[:, 1]
        ])
        
        # Soft mins
        min_d_pair = _smooth_min(dists, beta)
        min_d_wall = _smooth_min(wall_dists, beta)
        
        # Constraint functions: g(x) >= 0
        # g1 = min_d_pair - 2t
        # g2 = min_d_wall - t
        
        # Gradients
        # d(min_d_pair)/dx: 
        # soft_min(x) = -1/beta log(sum(exp(-beta x)))
        # d/dx_i = (beta exp(-beta x_i) sum(exp(-beta x))) / (beta sum(exp(-beta x))) 
        # Wait, derivative of -1/beta log(S) where S = sum(exp(-beta x))
        # d/dx_i = -1/beta * (1/S) * dS/dx_i
        # dS/dx_i = exp(-beta x_i) * (-beta)
        # So d/dx_i = -1/beta * (1/S) * (-beta exp(-beta x_i)) = exp(-beta x_i) / S
        # This is a softmax-like weight.
        
        # Let's compute weights for pairwise
        # We need to map flat dists vector back to pairs (i,j)
        # This is complex to do analytically in a vectorized way for the full gradient 
        # without looping or complex indexing. 
        # Given N=50, 1225 pairs. 
        # SLSQP numerical differentiation might be acceptable if we reduce maxiter or 
        # if the function is smooth enough. 
        # However, numerical diff for 101 variables is 101 evaluations per step. 
        # Each eval involves pdist (1225 dists) and log-sum-exp. 
        # This might be too slow for 200 iterations.
        
        # Alternative: Use a simpler penalty method or just minimize the soft-min directly 
        # without the epigraph variable t? 
        # If we minimize -min(dists, walls) directly, it's non-smooth.
        # If we minimize -soft_min(dists, walls), it's smooth.
        # Let's try that first: Minimize -soft_min(all_distances).
        # all_distances = [d_ij/2, d_iw]
        # Then r = soft_min(all_distances).
        # Maximizing r is equivalent to minimizing -soft_min(all_distances).
        
        # This is much simpler and doesn't require t.
        # Let's switch to this strategy: Minimize negative soft-min of all relevant distances.
        
        all_dists = np.concatenate([dists / 2.0, wall_dists])
        soft_min_val = _smooth_min(all_dists, beta)
        
        # For SLSQP, we can just use this as the objective.
        # But we still need bounds on c.
        # SLSQP supports bounds.
        
        # Let's re-implement _solve_epigraph to just do this.
        
        return None # Placeholder, we will change the approach
    
    # Actually, the epigraph approach with t is better for controlling the radius, 
    # but the gradient is hard. 
    # Let's stick to minimizing -soft_min(all_dists) with bounds.
    
    def obj_and_grad(x):
        c = x.reshape(N, 2)
        dists = pdist(c, metric='euclidean')
        wall_dists = np.concatenate([
            0.5 - c[:, 0],
            0.5 + c[:, 0],
            0.5 - c[:, 1],
            0.5 + c[:, 1]
        ])
        all_dists = np.concatenate([dists / 2.0, wall_dists])
        
        # Soft min
        min_val = np.min(all_dists)
        # To avoid underflow in exp, shift by min_val
        shifted = all_dists - min_val
        exp_shifted = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_shifted)
        soft_min = min_val - np.log(sum_exp) / beta
        
        # Gradient of soft_min w.r.t. x
        # d(soft_min)/dx = sum( w_k * d(dist_k)/dx )
        # where w_k = exp(-beta * (dist_k - min_val)) / sum_exp
        # Note: d(min_val)/dx is 0 in the interior of the domain of the min function 
        # (it's piecewise linear, derivative 0 except at kinks, but soft min is smooth).
        # Actually, the derivative of the soft min is the weighted average of the derivatives 
        # of the individual distances, with weights being the softmax probabilities.
        
        grad = np.zeros_like(x)
        
        # Weights
        weights = exp_shifted / sum_exp
        
        # Split weights back to pairwise and wall
        n_pairs = len(dists)
        w_pairs = weights[:n_pairs]
        w_walls = weights[n_pairs:]
        
        # Wall gradients
        # wall_dists order: 
        # 0..N-1: 0.5 - c[:,0]  -> d/dc[:,0] = -1
        # N..2N-1: 0.5 + c[:,0] -> d/dc[:,0] = +1
        # 2N..3N-1: 0.5 - c[:,1] -> d/dc[:,1] = -1
        # 3N..4N-1: 0.5 + c[:,1] -> d/dc[:,1] = +1
        
        # Accumulate wall gradients
        # For each circle i:
        # grad_x[i] += -w_walls[i] + w_walls[i+N]
        # grad_y[i] += -w_walls[i+2*N] + w_walls[i+3*N]
        
        grad[:N] += -w_walls[:N] + w_walls[N:2*N]
        grad[N:2*N] += -w_walls[2*N:3*N] + w_walls[3*N:4*N]
        
        # Pairwise gradients
        # We need to map the flat dists vector to pairs (i,j)
        # This is the bottleneck. 
        # For N=50, 1225 pairs. 
        # Looping in Python is slow. 
        # Can we vectorize?
        # The indices in pdist are such that dists[k] corresponds to pair (i,j) where 
        # i = ... and j = ...
        # There is a formula, but implementing it vectorized for gradient accumulation 
        # (scatter-add) is tricky in pure NumPy without advanced indexing tricks that 
        # might be slow or complex.
        
        # Alternative: Use a different distance metric or approximation?
        # Or just use numerical gradients for SLSQP? 
        # If we use numerical gradients, SLSQP will call the objective 2*100+1 times per step.
        # Objective is cheap (pdist + logsumexp).
        # 200 steps * 200 evals * (pdist cost) might be feasible.
        # pdist for 50 points is very fast.
        
        # Let's try using numerical gradients for SLSQP first.
        # If it's too slow, we can optimize.
        
        return soft_min
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    res = minimize(obj_and_grad, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
    
    return res.x.reshape(N, 2)

def _polish(centres, maxiter=1000):
    """
    High-precision polish using Nelder-Mead on the exact nonsmooth objective.
    """
    def objective(x):
        c = x.reshape(N, 2)
        # We want to maximize the radius, so minimize negative radius
        # But radius is min(min_pair/2, min_wall)
        # This is non-smooth. Nelder-Mead handles this well.
        r = compute_radius(c)
        return -r
    
    res = minimize(objective, centres.flatten(), method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    
    new_c = res.x.reshape(N, 2)
    np.clip(new_c, -0.5, 0.5, out=new_c)
    return new_c

def _grid_init(n_cols=8):
    """Generate a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_init():
    """Generate a random initialization within the square."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))

def _structured_init(strategy='spiral'):
    """Generate structured initializations."""
    if strategy == 'spiral':
        t = np.linspace(0, 4 * np.pi, N)
        r = 0.5 * t / (4 * np.pi)
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.column_stack([x, y])
    elif strategy == 'fibonacci':
        n = N
        i = np.arange(n)
        golden = (1 + np.sqrt(5)) / 2
        x = (i / golden) % 1.0 - 0.5
        y = i / n - 0.5
        return np.column_stack([x, y])
    elif strategy == 'lattice_mismatch':
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N].copy()
        grid += np.random.normal(0, 0.01, size=grid.shape)
        np.clip(grid, -0.5, 0.5, out=grid)
        return grid
    else:
        return _random_init()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centres = _grid_init()
    best_r = compute_radius(best_centres)
    
    random.seed(42)
    np.random.seed(42)
    
    strategies = ['grid', 'spiral', 'fibonacci', 'lattice_mismatch', 'random']
    
    # Phase 1: Smooth optimization (L-BFGS-B on soft-min)
    # We use multiple betas to control the sharpness of the soft-min.
    # Low beta: smoother, easier to optimize, but less accurate.
    # High beta: sharper, closer to true min, but harder to optimize.
    betas = [20, 50, 100, 200]
    
    for strat in strategies:
        if time.time() > deadline - 30:
            break
            
        if strat == 'grid':
            c0 = _grid_init()
        elif strat == 'spiral':
            c0 = _structured_init('spiral')
        elif strat == 'fibonacci':
            c0 = _structured_init('fibonacci')
        elif strat == 'lattice_mismatch':
            c0 = _structured_init('lattice_mismatch')
        else:
            c0 = _random_init()
            
        # Try a few betas
        for beta in betas:
            if time.time() > deadline - 30:
                break
            
            # Solve smooth problem
            c_smooth = _solve_epigraph(c0.flatten(), beta=beta, maxiter=100)
            r_smooth = compute_radius(c_smooth)
            
            if r_smooth > best_r:
                best_r = r_smooth
                best_centres = c_smooth.copy()
    
    # Phase 2: Polish the best result found
    if time.time() < deadline - 10:
        c_polished = _polish(best_centres, maxiter=500)
        r_polished = compute_radius(c_polished)
        
        if r_polished > best_r:
            best_r = r_polished
            best_centres = c_polished
    
    # Phase 3: Final tiny local search
    if time.time() < deadline - 2:
        # A few random nudges to escape any remaining flat spots
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            c_try = best_centres + np.random.normal(0, 0.001, size=(N, 2))
            np.clip(c_try, -0.5, 0.5, out=c_try)
            r_try = compute_radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_centres = c_try
    
    # Final validation
    best_centres = np.clip(best_centres, -0.5, 0.5)
    if np.any(~np.isfinite(best_centres)):
        best_centres = _grid_init()
    
    return best_centres
