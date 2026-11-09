"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_bound(a0, a4, a8):
    """
    Compute the Heisenberg uncertainty product sigma_x * sigma_p for the given Hermite coefficients.
    """
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 4790016.0
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    N_0 = np.sqrt(np.pi)
    N_4 = np.sqrt(384 * np.pi)
    N_8 = np.sqrt(10321920 * np.pi)
    N_12 = np.sqrt(4096 * 479001600 * np.pi)
    
    c_0 = a0 * N_0
    c_4 = a4 * N_4
    c_8 = a8 * N_8
    c_12 = a12 * N_12
    
    c = np.array([c_0, c_4, c_8, c_12])
    
    norm_sq = np.sum(c**2)
    if norm_sq < 1e-12:
        return 1e10
        
    # Indices in c correspond to n = 0, 4, 8, 12
    # Map index i to n_i
    n_vals = np.array([0, 4, 8, 12])
    
    # Matrix M_x for <x^2> and M_p for <p^2>
    # Diagonal: n + 0.5
    # Off-diagonal (n, m) where |n-m|=2: 0.5 * sqrt(min(n,m)*(min(n,m)+1))
    # Sign for M_p is negative for off-diagonal
    
    M_x = np.zeros((4, 4))
    M_p = np.zeros((4, 4))
    
    for i in range(4):
        n_i = n_vals[i]
        diag = n_i + 0.5
        M_x[i, i] = diag
        M_p[i, i] = diag
        
        for j in range(i + 1, 4):
            n_j = n_vals[j]
            if abs(n_i - n_j) == 2:
                n_low = min(n_i, n_j)
                val = 0.5 * np.sqrt(n_low * (n_low + 1))
                M_x[i, j] = val
                M_x[j, i] = val
                M_p[i, j] = -val
                M_p[j, i] = -val
            # If |n_i - n_j| > 2, term is 0

    x2_expect = c @ M_x @ c / norm_sq
    p2_expect = c @ M_p @ c / norm_sq
    
    # Variance is <x^2> - <x>^2. <x>=0 for even functions.
    var_x = x2_expect
    var_p = p2_expect
    
    # Heisenberg product
    if var_x < 0 or var_p < 0:
        return 1e10
    prod = np.sqrt(var_x * var_p)
    
    return prod

def construct():
    """
    Find optimal Hermite coefficients to minimize the upper bound C4 (Heisenberg product).
    """
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_bound = _compute_bound(*x0)
    best_x = x0.copy()
    
    # Multi-start optimization
    np.random.seed(42)
    n_starts = 25
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        if i == 0:
            x_start = x0.copy()
        else:
            # Random perturbation
            scales = np.abs(x0) + 1e-6
            # Scale perturbations to explore wider space if needed
            # Use a wider exploration range
            exploration_factor = 0.1 + 0.9 * (i / n_starts)
            x_start = x0 + np.random.randn(3) * scales * exploration_factor * 10.0
        
        try:
            result = minimize(
                _compute_bound,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if np.isfinite(result.fun) and result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
                
                # Polish with Nelder-Mead if L-BFGS-B gets stuck or for local refinement
                if time.time() < deadline - 5:
                    try:
                        result2 = minimize(
                            _compute_bound,
                            best_x,
                            method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                        )
                        if np.isfinite(result2.fun) and result2.fun < best_bound:
                            best_bound = result2.fun
                            best_x = result2.x.copy()
                    except Exception:
                        pass
        except Exception:
            continue
            
    # Final high-precision polish pass
    if time.time() < deadline - 2:
        try:
            result_final = minimize(
                _compute_bound,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            if np.isfinite(result_final.fun) and result_final.fun < best_bound:
                best_bound = result_final.fun
                best_x = result_final.x.copy()
        except Exception:
            pass
            
    return best_x
# EVOLVE-BLOCK-END
