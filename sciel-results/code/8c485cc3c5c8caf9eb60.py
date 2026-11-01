"""
Strategy: Smooth Surrogate Optimization (L-BFGS-B) with High-Precision Polish.
This approach reformulates the packing problem as a smooth optimization task.
Instead of simulating physical forces, we define a differentiable objective function
using a 'soft-min' (log-sum-exp) approximation of the minimum distance.
We maximize this objective using L-BFGS-B, which efficiently handles the gradient
information. To escape local minima, we use multiple random restarts.
Finally, a 'polish' phase uses very high soft-min parameters to sharply focus
on the critical constraints, allowing the optimizer to fine-tune the positions
to machine precision.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # Safety margin before 150s kill

def _compute_score(pts):
    """Compute the true packing radius (nonsmooth)."""
    n = len(pts)
    if n < 2:
        return np.min(1.0 - np.linalg.norm(pts, axis=1))
    
    # Vectorized pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Container margin
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    min_margin = np.min(1.0 - radii)
    
    return min(min_pair, min_margin)

def _softmin(vals, beta):
    """
    Soft-min approximation of min(vals).
    As beta -> infinity, softmin -> min.
    """
    neg = -vals
    m = np.max(neg)
    # Avoid overflow in exp
    if m > 1e12:
        return m
    exp_vals = np.exp(neg - m)
    log_sum = np.log(np.sum(exp_vals)) + m
    return -log_sum / beta

def _objective_and_grad(x, beta):
    """
    Objective: minimize -softmin(constraints).
    Constraints:
    1. Pairwise distances (dist_ij)
    2. Wall distances (1 - |c_i|)
    
    We want to maximize the minimum constraint.
    So we minimize the negative soft-min of the constraints.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # 1. Pairwise distances
    # dists[i,j] = |p_i - p_j|
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Only take upper triangle to avoid double counting in the soft-min
    # Actually, soft-min is symmetric, so we can just use all unique pairs.
    # Let's extract unique pairs for efficiency and correctness of gradient.
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # 2. Wall distances
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    wall_dists = 1.0 - radii
    
    # Combine all constraints
    # We want min of (pair_dists, wall_dists)
    # Note: pair_dists are distances between centers.
    # Wall_dists are distances from center to boundary.
    # The radius r is min(min_pair_dist/2, min_wall_dist).
    # So the constraints for r are: pair_dists/2 and wall_dists.
    # Let's define c_i such that r <= c_i.
    # c_pair = pair_dists / 2
    # c_wall = wall_dists
    
    c_pairs = pair_dists / 2.0
    c_walls = wall_dists
    
    # Combine
    all_c = np.concatenate([c_pairs, c_walls])
    
    # Objective: -softmin(all_c, beta)
    val = -_softmin(all_c, beta)
    
    # Gradient
    # d/dx (-softmin(c, beta)) = - (1/beta) * d/dx (-log(sum(exp(-beta*c_i))))
    # Let S = sum(exp(-beta*c_i))
    # softmin = -1/beta * log(S)
    # -softmin = 1/beta * log(S)
    # d(-softmin)/dx = 1/beta * (1/S) * dS/dx
    # dS/dx = sum(-beta * exp(-beta*c_i) * dc_i/dx)
    # So d(-softmin)/dx = - sum(exp(-beta*c_i) * dc_i/dx) / S
    
    neg_beta_c = -beta * all_c
    # Normalize for numerical stability
    m = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - m)
    S = np.sum(exp_vals)
    
    # Weights w_i = exp(-beta*c_i) / S
    w_pairs = exp_vals[:len(c_pairs)] / S
    w_walls = exp_vals[len(c_pairs):] / S
    
    grad = np.zeros_like(pts)
    
    # Gradient of pair distances
    # dc_pair/dx_i = (p_i - p_j) / dist_ij
    # dc_pair/dx_j = (p_j - p_i) / dist_ij = -dc_pair/dx_i
    
    # We need to accumulate gradients for each point.
    # For each pair (i, j) with weight w:
    # grad_i += w * (p_i - p_j) / dist_ij
    # grad_j += w * (p_j - p_i) / dist_ij
    
    # Vectorized accumulation is tricky with variable indices, so we loop over pairs.
    # N=50, pairs=1225. This is fast enough.
    for k in range(len(c_pairs)):
        i = i_idx[k]
        j = j_idx[k]
        w = w_pairs[k]
        if w < 1e-12:
            continue
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            continue
        unit_vec = diff / dist
        grad[i] += w * unit_vec
        grad[j] -= w * unit_vec
        
    # Gradient of wall distances
    # c_wall_i = 1 - |p_i|
    # dc_wall_i/dx_i = -p_i / |p_i|
    for i in range(n):
        w = w_walls[i]
        if w < 1e-12:
            continue
        r = radii[i]
        if r < 1e-12:
            continue
        grad[i] -= w * (pts[i] / r)
        
    # Reshape gradient to match x
    return val, grad.reshape(-1)

def _hex_lattice_init():
    """Generate a hexagonal lattice initialization."""
    pts = []
    spacing = 0.2
    for i in range(-10, 10):
        for j in range(-10, 10):
            x = i * spacing
            y = j * spacing * np.sqrt(3)
            if i % 2 != 0:
                y += spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 0.8:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) > N:
        indices = np.random.choice(len(pts), N, replace=False)
        pts = pts[indices]
    elif len(pts) < N:
        # Pad with random points if lattice is too sparse
        extra = np.random.uniform(-0.5, 0.5, size=(N - len(pts), 2))
        pts = np.vstack([pts, extra])
    return pts

def _random_init():
    """Generate a random initialization."""
    pts = np.random.uniform(-0.8, 0.8, size=(N, 2))
    radii = np.linalg.norm(pts, axis=1)
    mask = radii > 0.9
    pts[mask] = pts[mask] / radii[mask][:, None] * 0.9
    return pts

def _optimize_lbfgs(x0, beta, maxiter=500):
    """Run L-BFGS-B optimization."""
    res = minimize(
        _objective_and_grad,
        x0,
        args=(beta,),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return res.x, res.fun

def _polish(x, maxiter=500):
    """
    High-precision polish phase.
    Uses very high beta values to sharpen the soft-min, effectively
    optimizing the active constraints.
    """
    current_x = x.copy()
    best_x = current_x.copy()
    best_val = _objective_and_grad(current_x, beta=1000)[0]
    
    # Ramp up beta
    betas = [1000, 5000, 20000, 50000]
    
    for b in betas:
        res = minimize(
            _objective_and_grad,
            current_x,
            args=(b,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        current_x = res.x
        val = _objective_and_grad(current_x, b)[0]
        if val < best_val:
            best_val = val
            best_x = current_x.copy()
            
    return best_x

def construct():
    t_start = time.time()
    
    best_pts = None
    best_score = -np.inf
    
    # Strategy:
    # 1. Try several initializations (Hex, Random).
    # 2. For each, run L-BFGS with increasing beta.
    # 3. Polish the best result found.
    
    # Number of restarts
    # We have ~135s. L-BFGS is fast for N=50.
    # Let's allocate time for 10-15 restarts.
    
    num_restarts = 15
    
    for i in range(num_restarts):
        if time.time() - t_start > DEADLINE - 20:
            break
            
        # Choose initialization
        if i % 2 == 0:
            x0 = _hex_lattice_init().flatten()
        else:
            x0 = _random_init().flatten()
            
        # Multi-scale optimization
        # Start with low beta (smooth landscape) and increase
        betas = [10, 50, 200, 1000]
        
        current_x = x0
        for b in betas:
            if time.time() - t_start > DEADLINE - 10:
                break
            current_x, _ = _optimize_lbfgs(current_x, b, maxiter=200)
            
        # Check score
        pts = current_x.reshape(-1, 2)
        score = _compute_score(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    # Final Polish on the best result
    if best_pts is not None:
        if time.time() - t_start < DEADLINE - 5:
            polished_x = _polish(best_pts.flatten(), maxiter=200)
            polished_pts = polished_x.reshape(-1, 2)
            polished_score = _compute_score(polished_pts)
            
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts
    
    # Fallback if something went wrong
    if best_pts is None:
        best_pts = _random_init()
        
    return best_pts
