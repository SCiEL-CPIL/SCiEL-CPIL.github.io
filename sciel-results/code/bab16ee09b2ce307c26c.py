"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses SDE/Langevin dynamics with temperature annealing in log-space.
    Returns a list of 12 positive root locations.
    """
    start_time = time.time()
    DEADLINE = 600.0
    SAFETY_MARGIN = 30.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the Laguerre basis transformation matrix
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    w = np.array([k+1 for k in range(13)], dtype=float)

    def compute_coefficients_from_roots(roots):
        """Compute Laguerre coefficients from roots."""
        p = np.array([1.0])
        for r in roots:
            p = np.convolve(p, np.array([1.0, -r]))
        c = A_T_inv @ p
        return c

    def cost_from_roots(roots):
        """Compute the upper bound on C4 from the 12 roots."""
        roots = np.array(roots)
        if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
            return 1e10
        c = compute_coefficients_from_roots(roots)
        norm_sq = np.dot(c**2, w)
        if norm_sq <= 0 or not np.isfinite(norm_sq):
            return 1e10
        return 1.0 / norm_sq

    def log_cost(log_roots):
        """Cost in log-space."""
        roots = np.exp(log_roots)
        return cost_from_roots(roots)

    # SDE/Langevin dynamics with temperature annealing
    np.random.seed(12345)
    
    best_value = 1e10
    best_roots = None
    best_log_roots = None

    # Initialize with multiple starting points
    initial_configs = []
    # Geometric sequence
    initial_configs.append(np.log(np.array([4.0 * (2.0 ** i) for i in range(12)])))
    # Logarithmically spaced
    initial_configs.append(np.log(np.linspace(0.1, 100, 12)))
    # Uniformly spaced in log
    initial_configs.append(np.random.uniform(-1.0, 3.0, 12))
    # Clustered around moderate values
    initial_configs.append(np.log(np.linspace(1.0, 20, 12)))

    # SDE parameters
    T0 = 5.0       # Initial temperature
    T_min = 0.01   # Minimum temperature
    cooling_rate = 0.99995
    step_size_base = 0.1  # Base step size for Langevin updates
    n_sweeps = 20000      # Number of sweeps per restart

    for init_idx, log_roots_init in enumerate(initial_configs):
        if not time_ok():
            break
        
        log_roots = log_roots_init.copy()
        T = T0
        local_best = log_cost(log_roots)
        local_best_roots = np.exp(log_roots).copy()
        
        for sweep in range(n_sweeps):
            if not time_ok():
                break
            
            # Compute gradient numerically (central difference)
            grad = np.zeros(12)
            h = 1e-6
            for i in range(12):
                lp = log_roots.copy()
                lm = log_roots.copy()
                lp[i] += h
                lm[i] -= h
                grad[i] = (log_cost(lp) - log_cost(lm)) / (2 * h)
            
            # Langevin update: x_{t+1} = x_t - dt * grad + sqrt(2*T*dt) * noise
            dt = 0.01
            noise = np.random.normal(0, 1, 12)
            log_roots = log_roots - dt * grad + np.sqrt(2 * T * dt) * noise * step_size_base
            
            # Keep roots in reasonable range in log-space
            log_roots = np.clip(log_roots, -5.0, 10.0)
            
            # Evaluate current state
            current_cost = log_cost(log_roots)
            if np.isfinite(current_cost) and current_cost < local_best:
                local_best = current_cost
                local_best_roots = np.exp(log_roots).copy()
            
            # Cool down
            T = max(T * cooling_rate, T_min)
        
        # Update global best
        if local_best < best_value:
            best_value = local_best
            best_roots = local_best_roots.copy()
            best_log_roots = log_roots.copy()
    
    # Final polish with Nelder-Mead if we have a good candidate
    if best_roots is not None and time_ok():
        def final_cost(log_roots):
            roots = np.exp(log_roots)
            if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
                return 1e10
            return cost_from_roots(roots)
        
        try:
            result = minimize(final_cost, best_log_roots, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if np.isfinite(result.fun) and result.fun < best_value:
                roots_final = np.exp(result.x)
                if np.all(roots_final > 0) and np.all(np.isfinite(roots_final)):
                    val = cost_from_roots(roots_final)
                    if val < best_value:
                        best_value = val
                        best_roots = roots_final.copy()
        except Exception:
            pass
    
    # Fallback
    if best_roots is None:
        best_roots = np.array([4.0 * (2.0 ** i) for i in range(12)])
    
    # Ensure roots are positive and finite
    best_roots = np.maximum(best_roots, 1e-10)
    best_roots = best_roots[np.isfinite(best_roots)]
    
    while len(best_roots) < 12:
        best_roots = np.append(best_roots, best_roots[-1] * 1.1)
    
    return list(best_roots[:12])
# EVOLVE-BLOCK-END
