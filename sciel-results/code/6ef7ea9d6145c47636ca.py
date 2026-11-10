import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _calc_score(x):
    """Calculate the actual packing score (min radius)."""
    d_min = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(x[i] - x[j])
            if d < d_min:
                d_min = d
        d_b = 1.0 - np.linalg.norm(x[i])
        if d_b < d_min:
            d_min = d_b
    return d_min / 2.0

def _hex_lattice_init():
    """Generate initial hexagonal lattice points optimized for N circles."""
    # Try multiple spacings and pick the one that gives best initial score
    best_points = None
    best_score = -1
    
    for trial in range(20):
        spacing = 0.22 + trial * 0.005  # Range from ~0.22 to ~0.315
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points well inside boundary
        margin = spacing / 2.0 + 0.01
        mask = radius <= 1.0 - margin
        points, radius = points[mask], radius[mask]
        
        if len(points) >= N:
            # Select N points: prefer those closer to center for better packing
            idx = np.argsort(radius, kind="stable")[:N]
            candidate = points[idx]
            score = _calc_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
    
    if best_points is None:
        # Fallback: uniform random
        rng = np.random.default_rng(0)
        while True:
            pts = rng.uniform(-0.8, 0.8, size=(N, 2))
            if np.all(np.linalg.norm(pts, axis=1) < 0.9):
                return pts
    return best_points

def _soft_min_energy_and_grad(x, alpha):
    """
    Compute the soft-min energy and gradient.
    x: (N, 2) array of centers.
    alpha: temperature parameter for soft-min.
    Returns: (energy, gradient)
    """
    x = x.reshape(N, 2)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Avoid division by zero in gradient calculation for i=j
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances: 1 - |x_i|
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    b_dists = 1.0 - radii
    
    p_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([p_dists, b_dists])
    
    # Clamp distances to avoid numerical issues
    all_dists = np.maximum(all_dists, 1e-10)
    
    exponents = -alpha * all_dists
    max_exp = np.max(exponents)
    exponents -= max_exp
    
    weights = np.exp(exponents)
    sum_weights = np.sum(weights)
    
    if sum_weights < 1e-300:
        return 0.0, np.zeros_like(x)
        
    energy = (1.0 / alpha) * (np.log(sum_weights) + max_exp)
    
    grad = np.zeros_like(x)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    w_pairs = weights[:len(p_dists)]
    d_pairs = p_dists
    
    diff_vec = x[i_idx] - x[j_idx]
    mask = d_pairs > 1e-8
    grad_unit = np.zeros_like(diff_vec)
    grad_unit[mask] = diff_vec[mask] / d_pairs[mask, np.newaxis]
    
    weighted_grad = w_pairs[:, np.newaxis] * grad_unit
    
    np.add.at(grad, i_idx, -weighted_grad / sum_weights)
    np.add.at(grad, j_idx, weighted_grad / sum_weights)
    
    w_bounds = weights[len(p_dists):]
    radii_safe = np.where(radii < 1e-8, 1e-8, radii)
    grad_bounds = -x / radii_safe[:, np.newaxis]
    
    weighted_bounds = w_bounds[:, np.newaxis] * grad_bounds
    grad += -weighted_bounds / sum_weights
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (150s limit)
    
    # Initial guess
    x0 = _hex_lattice_init()
    
    best_x = x0.copy()
    best_score = _calc_score(best_x)
    
    # Optimization strategy:
    # 1. Multi-stage alpha annealing with fine steps
    # 2. Multiple restarts with varied perturbations
    # 3. Final high-precision polish
    
    # Stage 1: Coarse-to-fine alpha schedule
    alphas_stage1 = [3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0, 70.0, 100.0]
    
    current_x = x0.copy()
    
    for alpha in alphas_stage1:
        if time.time() > deadline - 30:
            break
            
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            x_new = res.x.reshape(N, 2)
            score_new = _calc_score(x_new)
            
            if score_new > best_score:
                best_score = score_new
                best_x = x_new.copy()
            current_x = x_new.copy()
        except Exception:
            pass
    
    # Stage 2: Multiple restarts with varied perturbations
    rng = np.random.default_rng(123)
    restart_configs = [
        (0.01, [30.0, 60.0]),
        (0.02, [20.0, 50.0]),
        (0.03, [15.0, 40.0]),
        (0.015, [25.0, 55.0]),
        (0.025, [20.0, 45.0]),
        (0.035, [10.0, 35.0]),
        (0.005, [40.0, 80.0]),
        (0.04, [8.0, 30.0]),
    ]
    
    for scale, alphas in restart_configs:
        if time.time() > deadline - 20:
            break
        
        x_perturbed = best_x + rng.normal(0, scale, size=best_x.shape)
        
        for alpha in alphas:
            if time.time() > deadline - 15:
                break
                
            try:
                res = minimize(
                    _soft_min_energy_and_grad,
                    x_perturbed,
                    args=(alpha,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                x_new = res.x.reshape(N, 2)
                score_new = _calc_score(x_new)
                
                if score_new > best_score:
                    best_score = score_new
                    best_x = x_new.copy()
                    break  # Move to next restart config if improved
            except Exception:
                pass
    
    # Stage 3: Final high-precision polish
    if time.time() < deadline - 10:
        for alpha in [150.0, 250.0, 400.0]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    _soft_min_energy_and_grad,
                    best_x,
                    args=(alpha,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                
                x_new = res.x.reshape(N, 2)
                score_new = _calc_score(x_new)
                
                if score_new > best_score:
                    best_score = score_new
                    best_x = x_new.copy()
            except Exception:
                pass
    
    return best_x
