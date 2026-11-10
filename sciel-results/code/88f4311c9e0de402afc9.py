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

def _soft_min_value_and_grad(x, alpha):
    """
    Compute soft-min value and gradient using log-sum-exp for numerical stability.
    We want to maximize the minimum distance, so we minimize -soft_min(distances).
    
    Returns: (soft_min_value, gradient) where gradient is of -soft_min w.r.t. x
    """
    x = np.asarray(x).reshape(N, 2)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    radii = np.sqrt(np.sum(x * x, axis=1))
    b_dists = 1.0 - radii
    
    # Collect all distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    p_dists = dists[i_idx, j_idx]
    all_dists = np.concatenate([p_dists, b_dists])
    
    # Soft-min via log-sum-exp: soft_min(d) = -1/alpha * log(sum(exp(-alpha*d)))
    # For numerical stability, use log-sum-exp trick
    exponents = -alpha * all_dists
    max_exp = np.max(exponents)
    shifted_exponents = exponents - max_exp
    sum_exp = np.sum(np.exp(shifted_exponents))
    
    if sum_exp < 1e-300:
        return 0.0, np.zeros_like(x)
    
    soft_min_val = -(1.0 / alpha) * (np.log(sum_exp) + max_exp)
    
    # Gradient computation
    # d(soft_min)/d(all_dists[k]) = -exp(-alpha*d_k) / sum(exp(-alpha*d_j))
    # So d(-soft_min)/d(all_dists[k]) = exp(-alpha*d_k) / sum(exp(-alpha*d_j))
    weights = np.exp(shifted_exponents) / sum_exp
    
    grad = np.zeros_like(x)
    
    # Pairwise distance gradients
    w_pairs = weights[:len(p_dists)]
    d_pairs = p_dists
    
    diff_vec = x[i_idx] - x[j_idx]
    mask = d_pairs > 1e-12
    grad_unit = np.zeros_like(diff_vec)
    grad_unit[mask] = diff_vec[mask] / d_pairs[mask, np.newaxis]
    
    # d(-soft_min)/dx_i for pair (i,j): w_ij * (x_i - x_j)/d_ij
    # d(-soft_min)/dx_j for pair (i,j): w_ij * (x_j - x_i)/d_ij
    weighted_grad = w_pairs[:, np.newaxis] * grad_unit
    
    np.add.at(grad, i_idx, weighted_grad)
    np.add.at(grad, j_idx, -weighted_grad)
    
    # Boundary distance gradients
    w_bounds = weights[len(p_dists):]
    # d(-soft_min)/dx_i for boundary: w_i * d(-soft_min)/d(b_dist_i) * d(b_dist_i)/dx_i
    # b_dist_i = 1 - |x_i|, so d(b_dist_i)/dx_i = -x_i/|x_i|
    # d(-soft_min)/d(b_dist_i) = w_i
    # So d(-soft_min)/dx_i = w_i * (-x_i/|x_i|)
    radii_safe = np.where(radii < 1e-12, 1e-12, radii)
    grad_bounds = -x / radii_safe[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * grad_bounds
    
    return soft_min_val, grad

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
    Compute the soft-min energy and gradient for minimization.
    We minimize -soft_min(distances) to maximize the minimum distance.
    
    x: (N, 2) array of centers.
    alpha: temperature parameter for soft-min.
    Returns: (energy, gradient)
    """
    soft_min_val, grad = _soft_min_value_and_grad(x, alpha)
    # We want to minimize -soft_min, so energy = -soft_min_val, grad = -grad
    return -soft_min_val, -grad

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
                maxiter = int(maxiter_base * 1.5)
            elif alpha > 20:
                ftol = 1e-13
                gtol = 1e-11
                maxiter = int(maxiter_base * 1.2)
                
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
    # 150s limit, keep 15s margin for safety
    deadline = start_time + 135.0
    
    # Initial guess
    x0 = _hex_lattice_init()
    
    best_x = x0.copy()
    best_score = _calc_score(best_x)
    
    # Strategy 1: Annealing from lattice
    # Coarse to fine alpha with better spacing
    alphas_main = [5.0, 10.0, 15.0, 25.0, 40.0, 60.0, 100.0, 150.0]
    
    best_x, best_score = _anneal_and_polish(x0, alphas_main, deadline)
    
    # Strategy 2: Random Restarts / Perturbations
    # Use remaining time to explore local minima near the best solution found
    rng = np.random.default_rng(42)
    
    # We can afford a few restarts if time permits
    # Each restart involves a shorter annealing sequence
    max_restarts = 15
    restarts_done = 0
    
    # If we have > 30s left, try restarts
    while time.time() < deadline - 20.0 and restarts_done < max_restarts:
        # Perturb the current best
        # Scale of perturbation: slightly larger than the radius to allow rearrangement
        # but small enough to stay in the basin
        perturb_scale = 0.008 + 0.008 * (restarts_done / max_restarts)
        
        x_perturbed = best_x + rng.normal(0, perturb_scale, size=best_x.shape)
        
        # Quick annealing sequence for the perturbed point
        # Use fewer iterations since it's a local search
        alphas_restart = [25.0, 50.0, 100.0]
        
        x_candidate, score_candidate = _anneal_and_polish(
            x_perturbed, 
            alphas_restart, 
            deadline, 
            maxiter_base=400, 
            ftol_base=1e-12, 
            gtol_base=1e-10
        )
        
        if score_candidate > best_score:
            best_score = score_candidate
            best_x = x_candidate.copy()
        
        restarts_done += 1
        
    # Final Polish: Very high alpha, tight tolerances
    if time.time() < deadline - 10.0:
        alphas_final = [150.0, 250.0, 400.0]
        best_x, best_score = _anneal_and_polish(
            best_x, 
            alphas_final, 
            deadline, 
            maxiter_base=2500, 
            ftol_base=1e-14, 
            gtol_base=1e-12
        )
        
    return best_x
