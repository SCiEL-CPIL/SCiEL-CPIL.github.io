"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 620.0  # safety margin before 630s kill

def _compute_c4_bound(r):
    """
    Compute the C4 upper bound for the Cohn-Goncalves Laguerre construction.
    
    For m=12 double roots with dimension n=1, the relevant polynomial is
    P(x) = prod_i (x - r_i)^2.
    
    The C4 bound is related to the integral:
    I = ∫_0^∞ |P(x)|^2 * x * e^{-x} dx / (∫_0^∞ x * e^{-x} dx)^2
    or a similar normalized quantity.
    
    We use Gauss-Laguerre quadrature for efficiency.
    """
    r = np.asarray(r, dtype=np.float64)
    if len(r) != 12:
        return 1e12
    if np.any(r <= 0) or np.any(np.isnan(r)) or np.any(np.isinf(r)):
        return 1e12
    
    # Use generalized Gauss-Laguerre quadrature with alpha=1
    # Weight: x^1 * e^{-x}
    from scipy.special import roots_genlaguerre
    
    # For degree 24 polynomial (P is degree 24), we need at least 13 nodes
    # Use 50 nodes for high accuracy
    n_quad = 50
    try:
        nodes, weights = roots_genlaguerre(n_quad, 1.0)
    except Exception:
        # Fallback to standard Laguerre
        from numpy.polynomial.laguerre import laggauss
        nodes_std, weights_std = laggauss(n_quad)
        nodes = nodes_std
        weights = weights_std * nodes_std  # approximate x^1 weight
    
    # Compute log|P(x)| at quadrature nodes
    # P(x) = prod_i (x - r_i)^2
    log_abs_P = np.zeros_like(nodes)
    for ri in r:
        diff = nodes - ri
        # Use log1p for numerical stability when diff is small
        log_abs_P += 2.0 * np.log(np.abs(diff) + 1e-300)
    
    # The integral is ∫ |P(x)|^2 * x * e^{-x} dx
    # = sum_j weights_j * |P(x_j)|^2
    # In log space: log(integrand) = 2*log_abs_P + log(weights)
    log_integrand = 2.0 * log_abs_P + np.log(weights + 1e-300)
    
    if np.any(np.isnan(log_integrand)) or np.any(np.isinf(log_integrand)):
        return 1e12
    
    # Normalize to avoid overflow
    max_log = np.max(log_integrand)
    if max_log > 700:  # would overflow
        # Scale down
        scaled = np.exp(log_integrand - max_log)
        integral = np.sum(scaled) * math.exp(max_log)
    else:
        integral = np.sum(np.exp(log_integrand))
    
    if not np.isfinite(integral) or integral <= 0:
        return 1e12
    
    # The C4 bound is proportional to this integral
    # Normalize by a reference value (the seed's value)
    # For the seed [4, 8, ..., 48], compute the reference
    return integral

def _seed_reference():
    """Compute the C4 bound for the seed configuration."""
    seed_r = np.array([4.0 * (i + 1) for i in range(12)])
    return _compute_c4_bound(seed_r)

def _run_optimization(initial_r, max_iter=2000):
    """Run scipy optimization with time guard."""
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() - _START_TIME > _TIME_LIMIT:
            raise TimeoutError()
        val = _compute_c4_bound(x)
        return val if np.isfinite(val) else 1e12
    
    try:
        result = minimize(
            obj,
            initial_r,
            method='L-BFGS-B',
            bounds=[(0.1, None)] * 12,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.success or result.fun < 1e10:
            return result.x, result.fun
        else:
            return initial_r, obj(initial_r)
    except TimeoutError:
        return initial_r, obj(initial_r)
    except Exception:
        return initial_r, obj(initial_r)

def _get_laguerre_zeros():
    """Get the zeros of L_12(x)."""
    try:
        from scipy.special import roots_laguerre
        zeros, _ = roots_laguerre(12)
        return zeros.copy()
    except Exception:
        pass
    
    # Fallback: compute via numpy
    try:
        n_lag = 12
        c = []
        for k in range(n_lag + 1):
            val = ((-1) ** k) * math.comb(n_lag + k, k) / math.factorial(k)
            c.append(val)
        c_rev = list(reversed(c))
        roots = np.roots(c_rev)
        real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0])
        if len(real_roots) == 12:
            return np.array(real_roots)
    except Exception:
        pass
    
    return None

def construct():
    """
    Returns list of 12 positive double-root locations for the Laguerre
    combination (m=12, n=1) that minimizes the C4 upper bound.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_r = None
    best_val = float('inf')
    
    # Get starting points
    starts = []
    
    # 1. Seed: [4, 8, ..., 48]
    starts.append(np.array([4.0 * (i + 1) for i in range(12)]))
    
    # 2. Zeros of L_12(x)
    lag_zeros = _get_laguerre_zeros()
    if lag_zeros is not None:
        starts.append(lag_zeros)
    
    # 3. Zeros of L_12(x) with different scaling
    if lag_zeros is not None:
        for scale in [0.5, 1.5, 2.0]:
            starts.append(lag_zeros * scale)
    
    # 4. Random starts
    np.random.seed(42)
    for trial in range(8):
        r = np.sort(np.random.uniform(1.0, 60.0, 12))
        starts.append(r)
    
    # 5. Different spacings
    for spacing in [2.0, 3.0, 5.0, 6.0]:
        r = spacing * np.arange(1, 13)
        starts.append(r)
    
    # Evaluate all starting points first to find the best one
    for start in starts:
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
        val = _compute_c4_bound(start)
        if val < best_val:
            best_val = val
            best_r = start.copy()
    
    # Run optimization from the best starting point
    if best_r is not None and time.time() - _START_TIME < _TIME_LIMIT - 60:
        remaining = _TIME_LIMIT - time.time() - 60
        max_iter = int(min(3000, max(1000, remaining * 200)))
        r_opt, val = _run_optimization(best_r, max_iter=max_iter)
        if val < best_val:
            best_val = val
            best_r = r_opt.copy()
    
    # Run optimization from a few other good starting points
    if time.time() - _START_TIME < _TIME_LIMIT - 30:
        # Sort starts by their objective value and try top few
        start_vals = []
        for start in starts:
            if time.time() - _START_TIME > _TIME_LIMIT - 30:
                break
            val = _compute_c4_bound(start)
            start_vals.append((val, start))
        
        start_vals.sort(key=lambda x: x[0])
        
        for val, start in start_vals[:5]:
            if time.time() - _START_TIME > _TIME_LIMIT - 30:
                break
            if val >= best_val:
                break
            remaining = _TIME_LIMIT - time.time() - 30
            max_iter = int(min(2000, max(500, remaining * 100)))
            r_opt, val_opt = _run_optimization(start, max_iter=max_iter)
            if val_opt < best_val:
                best_val = val_opt
                best_r = r_opt.copy()
    
    # Final polish: try a very tight optimization from the best point
    if best_r is not None and time.time() - _START_TIME < _TIME_LIMIT - 10:
        try:
            from scipy.optimize import minimize
            def obj_polish(x):
                if time.time() - _START_TIME > _TIME_LIMIT:
                    raise TimeoutError()
                return _compute_c4_bound(x)
            
            result = minimize(
                obj_polish,
                best_r,
                method='L-BFGS-B',
                bounds=[(0.1, None)] * 12,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.fun < best_val:
                best_r = result.x.copy()
                best_val = result.fun
        except Exception:
            pass
    
    # Ensure validity
    if best_r is None:
        return [4.0 * (i + 1) for i in range(12)]
    
    best_r = np.maximum(best_r, 0.1)
    best_r = np.sort(best_r)
    
    return [float(x) for x in best_r]
# EVOLVE-BLOCK-END
