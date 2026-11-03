"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_norm_log_and_grad(roots, n_quad=300):
    """
    Compute the log of the L2 norm squared of the polynomial P(x) = prod (x - r_i)
    with weight e^{-x}, and its gradient w.r.t. roots.
    We want to MAXIMIZE this norm to MINIMIZE the C4 bound (assuming Bound ~ 1/Norm).
    """
    m = len(roots)
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log P(x)^2 = 2 * sum log|x - r_i|
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        diff = np.abs(x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 2.0 * np.log(diff)
    
    # Integrand: e^{-x} * P(x)^2
    log_integrand = -x_quad + log_p_sq
    
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf, np.zeros(m)
    
    exp_shifted = np.exp(log_integrand - max_log)
    weighted = w_quad * exp_shifted
    sum_val = np.sum(weighted)
    
    if sum_val <= 0:
        return -np.inf, np.zeros(m)
        
    log_norm_sq = max_log + np.log(sum_val)
    
    # Gradient: d/d(r_j) log_norm_sq = 2 * sum_i w_i * exp(log_integrand_i - max_log) * (1/(x_i - r_j)) / sum_val
    # More precisely: d log_norm_sq / d r_j = 2 * sum_i w_i P(x_i)^2 e^{-x_i} * (1/(x_i - r_j)) / norm_sq
    # = 2 * sum_i weighted_i * (1/(x_i - r_j))
    grad = np.zeros(m)
    for j, r in enumerate(roots):
        diff = x_quad - r
        # Avoid division by zero
        diff[np.abs(diff) < 1e-300] = 1e-300
        grad[j] = 2.0 * np.sum(weighted / diff)
    
    return log_norm_sq, grad

def _dyson_gas_init(m=12):
    """
    Initialize root positions using Dyson gas equilibrium with repulsion exponent 2
    and linear confining potential. This gives a good starting point for optimization.
    
    The equilibrium condition for a Dyson gas with beta=2 (Gaussian unitary ensemble-like)
    and linear potential V(x) = x is:
    sum_{j != i} 2/(x_i - x_j) = x_i
    
    We solve this iteratively.
    """
    # Start from uniform spacing
    roots = np.linspace(2.0, 45.0, m)
    
    for iteration in range(500):
        # Compute forces
        forces = np.zeros(m)
        for i in range(m):
            # Confining force (from linear potential)
            forces[i] = roots[i]
            # Repulsive forces from other particles
            for j in range(m):
                if i != j:
                    forces[i] -= 2.0 / (roots[i] - roots[j])
        
        # Update positions (gradient descent with adaptive step)
        # The equilibrium is where forces = 0
        # Use simple relaxation
        step_size = 0.01
        for i in range(m):
            roots[i] += step_size * forces[i]
            roots[i] = max(roots[i], 0.5)  # Keep positive
        
        # Check convergence
        if np.max(np.abs(forces)) < 1e-10:
            break
    
    return np.sort(roots)

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Use Dyson gas initialization for better starting point
    try:
        dyson_roots = _dyson_gas_init(12)
    except Exception:
        dyson_roots = np.array([4.0 * i for i in range(1, 13)])
    
    # Also keep the seed roots as a fallback
    seed_roots = np.array([4.0 * i for i in range(1, 13)])
    
    # Evaluate both starting points
    best_roots = dyson_roots.copy()
    best_log_norm, _ = _compute_norm_log_and_grad(best_roots)
    
    seed_log_norm, _ = _compute_norm_log_and_grad(seed_roots)
    if seed_log_norm > best_log_norm:
        best_roots = seed_roots.copy()
        best_log_norm = seed_log_norm
    
    # We want to MAXIMIZE the norm, so MINIMIZE the negative log norm
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            return 1e10  # Penalty for timeout
        roots = np.sort(params)
        if any(r <= 1e-6 for r in roots):
            return 1e10
        log_norm, _ = _compute_norm_log_and_grad(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm
    
    def objective_grad(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            return 1e10, np.zeros(len(params))
        roots = np.sort(params)
        if any(r <= 1e-6 for r in roots):
            return 1e10, np.zeros(len(params))
        log_norm, grad = _compute_norm_log_and_grad(roots)
        if not np.isfinite(log_norm):
            return 1e10, np.zeros(len(params))
        return -log_norm, -grad  # Minimize negative log norm
    
    # Optimization with multiple strategies
    try:
        # Strategy 1: L-BFGS-B from best initial point
        result = minimize(objective, best_roots, jac=objective_grad,
                         method='L-BFGS-B',
                         bounds=[(0.5, None)] * 12,
                         options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if result.success and result.fun < -best_log_norm:
            best_roots = np.sort(result.x)
            best_log_norm = -result.fun
            
        # Strategy 2: Try from seed roots if time permits
        if time.time() - start_time < deadline - 120:
            result2 = minimize(objective, seed_roots, jac=objective_grad,
                             method='L-BFGS-B',
                             bounds=[(0.5, None)] * 12,
                             options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result2.success and result2.fun < -best_log_norm:
                best_roots = np.sort(result2.x)
                best_log_norm = -result2.fun
        
        # Strategy 3: Polish with tighter tolerances
        if time.time() - start_time < deadline - 60:
            result3 = minimize(objective, best_roots, jac=objective_grad,
                             method='L-BFGS-B',
                             bounds=[(0.5, None)] * 12,
                             options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-12})
            
            if result3.success and result3.fun < -best_log_norm:
                best_roots = np.sort(result3.x)
                best_log_norm = -result3.fun
                
        # Strategy 4: Nelder-Mead for local refinement if time permits
        if time.time() - start_time < deadline - 30:
            result4 = minimize(objective, best_roots, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result4.fun < -best_log_norm:
                best_roots = np.sort(result4.x)
                best_log_norm = -result4.fun
                
    except Exception:
        pass
    
    # Final verification: Ensure roots are positive and sorted
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != 12:
        best_roots = seed_roots.copy()
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
