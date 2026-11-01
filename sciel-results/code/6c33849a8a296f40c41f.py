"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_lattice_init():
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    return points

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _objective_and_grad(x_flat, beta):
    """
    Objective: -soft_min(pairwise_dists/2, boundary_dists)
    Gradient: computed analytically.
    """
    x = x_flat.reshape(-1, 2)
    n = len(x)
    
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # We only need i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(x, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min
    y_min = np.min(all_dists)
    exps = np.exp(-beta * (all_dists - y_min))
    sum_exps = np.sum(exps)
    soft_min_val = y_min - (1.0 / beta) * np.log(sum_exps)
    
    # Objective is -soft_min_val
    obj = -soft_min_val
    
    # Gradient
    probs = exps / sum_exps
    
    grad_flat = np.zeros_like(x_flat)
    
    # Gradient of pairwise distances
    safe_dists = np.where(dists < 1e-8, 1e-8, dists)
    
    # We need to map pair index back to (i, j) to build the gradient matrix.
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    A = np.zeros((n, n))
    p_pairs = probs[:len(pair_dists)]
    d_pairs = dists[i_idx, j_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1e-8, d_pairs)
    weights = p_pairs / (2.0 * safe_d_pairs)
    
    A[i_idx, j_idx] = weights
    A[j_idx, i_idx] = weights
    
    row_sums = A.sum(axis=1)
    Ax = A @ x
    
    grad_pairwise = x * row_sums[:, np.newaxis] - Ax
    
    # Gradient of boundary distances
    p_bound = probs[len(pair_dists):]
    safe_norms = np.where(norms < 1e-8, 1e-8, norms)
    grad_bound = -p_bound[:, np.newaxis] * x / safe_norms[:, np.newaxis]
    
    total_grad = grad_pairwise + grad_bound
    
    # Objective is -soft_min, so gradient is -total_grad
    return obj, -total_grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Initialization
    points = _hex_lattice_init()
    
    # 2. L-BFGS Optimization with Soft Min
    # Start with low beta (smooth approximation) and increase beta (sharper approximation)
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    x_current = points.flatten()
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
            
        # Define local objective and gradient for this beta
        def obj_grad(x):
            return _objective_and_grad(x, beta)
        
        try:
            result = minimize(
                lambda x: obj_grad(x)[0],
                x_current,
                jac=lambda x: obj_grad(x)[1],
                method='L-BFGS-B',
                bounds=[(-1.1, 1.1)] * (2 * N), # Loose bounds
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < -1e6: # Check if it improved or ran
                x_current = result.x
        except Exception:
            pass
            
    # 3. Final Polish
    points_final = x_current.reshape(-1, 2)
    
    best_score = _compute_score(points_final)
    best_points = points_final.copy()
    
    # Try a few random perturbations and local optimization if time permits
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            # Perturb
            perturb = np.random.randn(2 * N) * 0.01
            x_pert = best_points.flatten() + perturb
            # Optimize with high beta
            try:
                def obj_grad_pert(x):
                    return _objective_and_grad(x, 1000.0)
                res = minimize(
                    lambda x: obj_grad_pert(x)[0],
                    x_pert,
                    jac=lambda x: obj_grad_pert(x)[1],
                    method='L-BFGS-B',
                    bounds=[(-1.1, 1.1)] * (2 * N),
                    options={'maxiter': 50}
                )
                cand = res.x.reshape(-1, 2)
                score = _compute_score(cand)
                if score > best_score:
                    best_score = score
                    best_points = cand
            except Exception:
                pass
                
    return best_points
# EVOLVE-BLOCK-END
