import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _calc_score(x):
    """Calculate the actual packing score (min radius). Vectorized."""
    x = np.asarray(x).reshape(N, 2)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    radii = np.sqrt(np.sum(x * x, axis=1))
    min_bound = np.min(1.0 - radii)
    return min(min_pair, min_bound) / 2.0

def _hex_lattice_init():
    """Generate initial hexagonal lattice points."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _soft_min_energy_and_grad(x, alpha):
    """
    Compute the soft-min energy and gradient.
    x: (N, 2) array of centers.
    alpha: temperature parameter for soft-min.
    Returns: (energy, gradient)
    """
    x = np.asarray(x).reshape(N, 2)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    radii = np.sqrt(np.sum(x * x, axis=1))
    b_dists = 1.0 - radii
    
    p_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([p_dists, b_dists])
    
    exponents = -alpha * all_dists
    max_exp = np.max(exponents)
    if max_exp > 0:
        exponents -= max_exp
        shift = max_exp
    else:
        shift = 0.0
        
    weights = np.exp(exponents)
    sum_weights = np.sum(weights)
    
    if sum_weights == 0:
        return 0.0, np.zeros_like(x)
        
    energy = (1.0 / alpha) * (np.log(sum_weights) + shift)
    
    grad = np.zeros_like(x)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    w_pairs = weights[:len(p_dists)]
    d_pairs = p_dists
    
    diff_vec = x[i_idx] - x[j_idx]
    mask = d_pairs > 1e-10
    grad_unit = np.zeros_like(diff_vec)
    grad_unit[mask] = diff_vec[mask] / d_pairs[mask, np.newaxis]
    
    weighted_grad = w_pairs[:, np.newaxis] * grad_unit
    
    np.add.at(grad, i_idx, -weighted_grad / sum_weights)
    np.add.at(grad, j_idx, weighted_grad / sum_weights)
    
    w_bounds = weights[len(p_dists):]
    radii_safe = np.where(radii < 1e-10, 1e-10, radii)
    grad_bounds = -x / radii_safe[:, np.newaxis]
    
    weighted_bounds = w_bounds[:, np.newaxis] * grad_bounds
    grad += -weighted_bounds / sum_weights
    
    return energy, grad

def _anneal_and_polish(x_start, alphas, deadline, maxiter_base=2000, ftol_base=1e-12, gtol_base=1e-10):
    """
    Run annealing sequence of soft-min optimizations.
    Returns best_x, best_score found along the way.
    """
    best_x = x_start.copy()
    best_score = _calc_score(best_x)
    current_x = x_start.copy()
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        try:
            # Scale iterations/tolerance based on alpha for efficiency
            # Higher alpha -> sharper min -> can use tighter tol but maybe fewer iters needed if close
            maxiter = maxiter_base
            ftol = ftol_base
            gtol = gtol_base
            
            if alpha > 50:
                ftol = 1e-14
                gtol = 1e-12
                maxiter = maxiter_base * 1.5
                
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': int(maxiter), 'ftol': ftol, 'gtol': gtol}
            )
            
            x_new = res.x.reshape(N, 2)
            score_new = _calc_score(x_new)
            
            if score_new > best_score:
                best_score = score_new
                best_x = x_new.copy()
                current_x = x_new.copy()
        except Exception:
            pass
            
    return best_x, best_score

def construct():
    start_time = time.time()
    # 150s limit, keep 10s margin for safety
    deadline = start_time + 140.0
    
    # Initial guess
    x0 = _hex_lattice_init()
    
    best_x = x0.copy()
    best_score = _calc_score(best_x)
    
    # Strategy 1: Annealing from lattice
    # Coarse to fine alpha
    alphas_main = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    best_x, best_score = _anneal_and_polish(x0, alphas_main, deadline)
    
    # Strategy 2: Random Restarts / Perturbations
    # Use remaining time to explore local minima near the best solution found
    rng = np.random.default_rng(42)
    
    # We can afford a few restarts if time permits
    # Each restart involves a shorter annealing sequence
    max_restarts = 10
    restarts_done = 0
    
    # If we have > 30s left, try restarts
    while time.time() < deadline - 15.0 and restarts_done < max_restarts:
        # Perturb the current best
        # Scale of perturbation: slightly larger than the radius to allow rearrangement
        # but small enough to stay in the basin
        perturb_scale = 0.01 + 0.01 * (restarts_done / max_restarts)
        
        x_perturbed = best_x + rng.normal(0, perturb_scale, size=best_x.shape)
        
        # Quick annealing sequence for the perturbed point
        # Use fewer iterations since it's a local search
        alphas_restart = [20.0, 50.0, 100.0]
        
        x_candidate, score_candidate = _anneal_and_polish(
            x_perturbed, 
            alphas_restart, 
            deadline, 
            maxiter_base=500, 
            ftol_base=1e-11, 
            gtol_base=1e-9
        )
        
        if score_candidate > best_score:
            best_score = score_candidate
            best_x = x_candidate.copy()
        
        restarts_done += 1
        
    # Final Polish: Very high alpha, tight tolerances
    if time.time() < deadline - 5.0:
        alphas_final = [200.0, 500.0]
        best_x, best_score = _anneal_and_polish(
            best_x, 
            alphas_final, 
            deadline, 
            maxiter_base=3000, 
            ftol_base=1e-15, 
            gtol_base=1e-13
        )
        
    return best_x
