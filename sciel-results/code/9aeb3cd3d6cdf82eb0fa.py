"""Heilbronn triangles: maximize min triangle area for 11 points in unit equilateral triangle."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
TIME_LIMIT = 140  # seconds, safety margin before 150s kill

# Triangle vertices in Cartesian for final output
# A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
# Barycentric coords (u, v, w) with u+v+w=1, u,v,w >= 0
# Point = u*A + v*B + w*C
# Since A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2):
# x = v*1 + w*0.5
# y = w*sqrt(3)/2

# Precompute indices for all triples (i, j, k) with i < j < k
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*A + v*B + w*C
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords relative to unit triangle area? No, relative to standard basis).
    Actually, the area of a triangle formed by 3 points in barycentric coords (u_i, v_i, w_i) is proportional to the determinant of the matrix of barycentric coords.
    Specifically, Area(P_i P_j P_k) = |det(M)| * Area(Triangle)
    where M is the 3x3 matrix with rows (u_i, v_i, w_i), etc.
    Since we work in the simplex u+v+w=1, we can use just u and v.
    Let P_i = (u_i, v_i). The area in the (u,v) plane is 0.5 * |det([[1, u_i, v_i], [1, u_j, v_j], [1, u_k, v_k]])|.
    The actual Cartesian area is this barycentric area multiplied by the area of the reference triangle (which is sqrt(3)/4 for the unit side length triangle defined by (0,0),(1,0),(0.5,sqrt(3)/2)).
    Wait, the problem says "unit-area triangle".
    The triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2) has area sqrt(3)/4.
    If the task implies the triangle *is* the unit area triangle, then we should scale coordinates or normalize the output.
    The prompt says: "Place 11 points on or inside a unit-area triangle... The notebook uses an equilateral triangle and divides by its area."
    And: "Interface: construct() -> array of shape (n, 2): n points inside the equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2)"
    This is slightly contradictory. Usually "unit-area" means area=1. But the vertices given define an area of sqrt(3)/4.
    However, the evaluator measures "raw value". If the evaluator expects the points in the specific triangle (0,0),(1,0),(0.5,sqrt(3)/2), it will calculate areas in that space.
    If the evaluator normalizes by the triangle area, then maximizing the ratio is equivalent to maximizing the area in that specific triangle.
    I will assume the evaluator calculates the area of the triangle formed by the 3 points in Cartesian coordinates and divides by the area of the container triangle (sqrt(3)/4).
    So, maximizing the Cartesian area is the goal.
    
    The determinant of the barycentric matrix D = det([[1, u_i, v_i], [1, u_j, v_j], [1, u_k, v_k]])
    The Cartesian area is 0.5 * |D| * Area(Container).
    Since Area(Container) is constant, maximizing min(Cartesian Area) is equivalent to maximizing min(|D|).
    So we can just work with the determinants in barycentric space.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Extract coordinates for all triples
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    u_i = u[i_idx]
    v_i = v[i_idx]
    u_j = u[j_idx]
    v_j = v[j_idx]
    u_k = u[k_idx]
    v_k = v[k_idx]
    
    # Determinant of [[1, u_i, v_i], [1, u_j, v_j], [1, u_k, v_k]]
    # = 1*(u_j*v_k - v_j*u_k) - u_i*(1*v_k - v_j*1) + v_i*(1*u_k - u_j*1)
    # = u_j*v_k - v_j*u_k - u_i*v_k + u_i*v_j + v_i*u_k - v_i*u_j
    dets = u_j * v_k - v_j * u_k - u_i * v_k + u_i * v_j + v_i * u_k - v_i * u_j
    
    return np.abs(dets)

def _min_det_ratio_from_bary(bary):
    """
    Return the minimum determinant (which is proportional to min area).
    """
    dets = _all_dets_from_bary(bary)
    return np.min(dets)

def _project_to_bary_simplex(bary):
    """
    Project points onto the barycentric simplex (u>=0, v>=0, u+v<=1).
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip u and v to [0, 1]
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    # If u + v > 1, we need to scale back
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
        
    return np.column_stack((u, v))

def _objective_smooth(x, beta=100.0):
    """
    Objective to minimize: -soft_min(det).
    x is flattened barycentric coords (2*N).
    """
    bary = x.reshape(-1, 2)
    bary = _project_to_bary_simplex(bary)
    
    dets = _all_dets_from_bary(bary)
    
    # Soft min: min(dets) approx -log(sum(exp(-beta * dets))) / beta
    # We want to maximize min, so we minimize -min.
    # -min(dets) approx log(sum(exp(-beta * dets))) / beta
    shifted = -beta * dets
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return log_sum / beta

def _objective_grad(x, beta=100.0):
    """
    Gradient of the smooth objective.
    """
    bary = x.reshape(-1, 2)
    bary = _project_to_bary_simplex(bary)
    
    dets = _all_dets_from_bary(bary)
    
    # Gradient of log(sum(exp(-beta * dets))) / beta
    # Let S = sum(exp(-beta * dets))
    # f = log(S) / beta
    # df/dx = (1/beta) * (1/S) * sum( d(exp(-beta * dets))/dx )
    # = (1/beta) * (1/S) * sum( exp(-beta * dets) * (-beta) * (ddets/dx) )
    # = - (1/S) * sum( exp(-beta * dets) * (ddets/dx) )
    
    shifted = -beta * dets
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    S = np.sum(exp_shifted)
    
    # We need gradients of dets w.r.t bary coords.
    # det(i,j,k) = u_j*v_k - v_j*u_k - u_i*v_k + u_i*v_j + v_i*u_k - v_i*u_j
    # This is linear in u and v for fixed indices.
    # Let's compute the gradient of the objective w.r.t each point's u and v.
    
    grad_u = np.zeros(N)
    grad_v = np.zeros(N)
    
    # Weights for each determinant in the soft min
    weights = exp_shifted / S
    
    # Iterate over triples to accumulate gradients
    # This is O(N^3) but N=11, so 165 triples.
    for t in range(N_TRIPLES):
        w = weights[t]
        i, j, k = _IDX_TRIPLES[t]
        
        # d(det)/du_i
        # det = ... - u_i*v_k + u_i*v_j ...
        # d/du_i = -v_k + v_j
        grad_u[i] += w * (v[j] - v[k])
        
        # d(det)/du_j
        # det = u_j*v_k ... - v_i*u_j
        # d/du_j = v_k - v_i
        grad_u[j] += w * (v[k] - v[i])
        
        # d(det)/du_k
        # det = ... - v_j*u_k ... + v_i*u_k
        # d/du_k = -v_j + v_i
        grad_u[k] += w * (v[i] - v[j])
        
        # d(det)/dv_i
        # det = ... - u_i*v_k ... + v_i*u_k - v_i*u_j
        # d/dv_i = -u_k + u_k - u_j = -u_j
        # Wait, let's re-evaluate:
        # det = u_j*v_k - v_j*u_k - u_i*v_k + u_i*v_j + v_i*u_k - v_i*u_j
        # Terms with v_i: + v_i*u_k - v_i*u_j
        # d/dv_i = u_k - u_j
        grad_v[i] += w * (u[k] - u[j])
        
        # Terms with v_j: - v_j*u_k + u_i*v_j
        # d/dv_j = -u_k + u_i
        grad_v[j] += w * (u[i] - u[k])
        
        # Terms with v_k: u_j*v_k - u_i*v_k
        # d/dv_k = u_j - u_i
        grad_v[k] += w * (u[j] - u[i])
        
    # The objective is log_sum / beta.
    # The derivative of log(S) is (1/S) * S'
    # S' = sum(exp(-beta * dets) * (-beta) * d(dets)/dx)
    # So d(obj)/dx = (1/beta) * (1/S) * S' = (1/S) * sum(exp(-beta * dets) * d(dets)/dx)
    # Wait, the sign.
    # obj = log(S)/beta.
    # d(obj)/dx = (1/beta) * (1/S) * dS/dx
    # dS/dx = sum(exp(-beta * dets) * (-beta) * d(dets)/dx)
    # d(obj)/dx = (1/beta) * (1/S) * (-beta) * sum(exp(-beta * dets) * d(dets)/dx)
    # d(obj)/dx = - (1/S) * sum(exp(-beta * dets) * d(dets)/dx)
    # d(obj)/dx = - sum(weights * d(dets)/dx)
    
    # My accumulation above was sum(weights * d(det)/dcoord).
    # So the gradient of the objective is -accumulated_grad.
    
    grad = np.zeros(2 * N)
    grad[0::2] = -grad_u
    grad[1::2] = -grad_v
    
    return grad

def _structured_init_fps(rng, num_points=1000):
    """
    Generate initial points using Farthest Point Sampling on a grid in the simplex.
    """
    # Create a grid of points in the barycentric simplex
    grid_size = 50
    u_grid, v_grid = np.meshgrid(np.linspace(0.05, 0.95, grid_size), np.linspace(0.05, 0.95, grid_size))
    u_grid = u_grid.flatten()
    v_grid = v_grid.flatten()
    
    # Filter valid points
    valid = (u_grid >= 0) & (v_grid >= 0) & (u_grid + v_grid <= 1)
    u_valid = u_grid[valid]
    v_valid = v_grid[valid]
    
    candidates = np.column_stack((u_valid, v_valid))
    
    # FPS
    selected = [np.array([0.5, 0.5])] # Start with center
    dists = np.linalg.norm(candidates - selected[0], axis=1)
    
    for _ in range(N - 1):
        # Pick the point with max distance to any selected point
        min_dists = np.zeros(len(candidates))
        for s in selected:
            d = np.linalg.norm(candidates - s, axis=1)
            min_dists = np.minimum(min_dists, d)
        
        # Add some randomness to avoid deterministic bias
        # Perturb distances slightly
        min_dists += rng.uniform(0, 0.01, size=len(candidates))
        
        next_idx = np.argmax(min_dists)
        selected.append(candidates[next_idx])
        
    return np.array(selected)

def _local_optimize_lbfgs(bary, beta=100.0, max_iter=200):
    """
    Optimize using L-BFGS-B with smooth objective.
    """
    x0 = bary.flatten()
    
    res = minimize(
        _objective_smooth,
        x0,
        jac=_objective_grad,
        args=(beta,),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if res.success:
        return res.x.reshape(-1, 2)
    else:
        return bary

def _polish_slsqp(bary, max_iter=50):
    """
    Polish using SLSQP with epigraph formulation.
    Minimize t subject to det(i,j,k) >= t for all i,j,k.
    """
    def objective(x):
        # x[0] is t, x[1:] are bary coords
        t = x[0]
        bary = x[1:].reshape(-1, 2)
        bary = _project_to_bary_simplex(bary)
        return t
    
    def constraints(x):
        t = x[0]
        bary = x[1:].reshape(-1, 2)
        bary = _project_to_bary_simplex(bary)
        dets = _all_dets_from_bary(bary)
        # Constraints: det - t >= 0
        return [{'type': 'ineq', 'fun': lambda x, d=dets: d - x[0]}] # This lambda won't work correctly with SLSQP if dets changes
    
    # SLSQP requires a single function for constraints or a list of dicts.
    # It's better to define the constraint function properly.
    
    def constraint_func(x):
        t = x[0]
        bary = x[1:].reshape(-1, 2)
        bary = _project_to_bary_simplex(bary)
        dets = _all_dets_from_bary(bary)
        return dets - t
    
    # SLSQP supports a single 'ineq' constraint function that returns an array of values >= 0.
    # But scipy.optimize.minimize SLSQP expects a list of dictionaries or a single dictionary.
    # If we pass a function that returns an array, we need to wrap it.
    # Actually, SLSQP accepts a list of constraints.
    # Let's use a single constraint function that returns the vector of (det - t).
    
    # However, SLSQP's 'ineq' constraint expects the function to return a scalar or an array?
    # Documentation: "A sequence of constraint definitions... each constraint is a dictionary... 'fun' is a function that returns an array of constraint values."
    # So we can define:
    cons = ({'type': 'ineq', 'fun': constraint_func},)
    
    x0 = np.concatenate(([0.0], bary.flatten()))
    
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=cons,
        options={'maxiter': max_iter, 'ftol': 1e-12}
    )
    
    if res.success:
        return res.x[1:].reshape(-1, 2)
    else:
        return bary

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_bary = None
    best_det = 0.0
    
    # Generate multiple initial candidates
    inits = []
    
    # 1. FPS structured init
    fps_init = _structured_init_fps(rng)
    inits.append(fps_init)
    
    # 2. Random inits
    for seed in [42, 123, 456, 789, 101112]:
        r = np.random.RandomState(seed)
        u = r.uniform(0, 1, N)
        v = r.uniform(0, 1, N)
        # Rejection sampling for simplex
        valid = u + v <= 1
        while not np.all(valid):
            u[~valid] = r.uniform(0, 1, np.sum(~valid))
            v[~valid] = r.uniform(0, 1, np.sum(~valid))
            valid = u + v <= 1
        inits.append(np.column_stack((u, v)))
        
    # 3. Grid-like init (corners and center)
    grid_init = np.array([
        [0.1, 0.1], [0.1, 0.8], [0.8, 0.1], # Corners (approx)
        [0.5, 0.5], [0.3, 0.3], [0.7, 0.3], # Center and mid-edges
        [0.5, 0.1], [0.1, 0.5], [0.5, 0.8], # Edge midpoints
        [0.3, 0.6], [0.6, 0.3]
    ])
    # Project to simplex
    grid_init = _project_to_bary_simplex(grid_init)
    inits.append(grid_init)
    
    # Evaluate and sort inits
    scored_inits = []
    for init in inits:
        det = _min_det_ratio_from_bary(init)
        scored_inits.append((det, init))
        
    scored_inits.sort(key=lambda x: -x[0])
    
    # Optimize top candidates
    max_optimize = min(5, len(scored_inits))
    for i in range(max_optimize):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 20:
            break
            
        init = scored_inits[i][1].copy()
        
        # Step 1: L-BFGS-B with smooth objective
        refined = _local_optimize_lbfgs(init, beta=100.0, max_iter=200)
        det = _min_det_ratio_from_bary(refined)
        
        # Step 2: SLSQP polish
        polished = _polish_slsqp(refined, max_iter=100)
        det_polished = _min_det_ratio_from_bary(polished)
        
        if det_polished > best_det:
            best_det = det_polished
            best_bary = polished.copy()
            
        # Check time
        if time.time() - start_time > TIME_LIMIT - 10:
            break

    # If we still have time, do a final polish on the best solution
    if best_bary is not None and time.time() - start_time < TIME_LIMIT - 5:
        final_polished = _polish_slsqp(best_bary, max_iter=200)
        det_final = _min_det_ratio_from_bary(final_polished)
        if det_final > best_det:
            best_bary = final_polished
            best_det = det_final

    # Convert to Cartesian
    if best_bary is not None:
        cart = _bary_to_cart(best_bary)
        return cart
    else:
        # Fallback: return random points
        return _bary_to_cart(np.column_stack((np.ones(N)*0.5, np.ones(N)*0.5)))

