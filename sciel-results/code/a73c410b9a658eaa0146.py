"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_c3_value(h):
    """
    Compute the exact C3 upper bound for a given height vector h.
    Metric: max_k |conv(h, h)[k]| / (sum(|h|))^2
    """
    n_fft = 2**int(np.ceil(np.log2(2*N-1)))
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    conv = C[:2*N-1]
    max_abs = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h < 1e-15:
        return float('inf')
    return max_abs / (sum_abs_h**2)


def _c3_upper_bound(h):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    Uses continuous relaxation with tanh mapping and L-BFGS-B optimization.
    """
    start_time = time.time()
    deadline = 115.0  # Leave 35s margin for safety
    
    def objective(x):
        h = np.tanh(x)
        n_fft = 2**int(np.ceil(np.log2(2*N-1)))
        H = np.fft.rfft(h, n=n_fft)
        C = np.fft.irfft(H * H, n=n_fft)
        conv = C[:2*N-1]
        
        # Smooth max with adaptive temperature
        eps = 1e-10
        abs_conv = np.sqrt(conv**2 + eps)
        T = 1.0
        max_abs = np.max(abs_conv)
        if max_abs > 1e-15:
            log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
            log_max = T * log_sum_exp
        else:
            log_max = 0.0
        
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        value = log_max / (sum_abs_h**2)
        return value
    
    best_h = np.ones(N)
    best_value = _compute_c3_value(best_h)
    
    # Multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    # Try several different starting points
    initializations = [
        np.full(N, 5.0),           # All positive
        np.full(N, -5.0),          # All negative
        np.full(N, 0.0),           # Zero
        rng.normal(0, 2, N),       # Random Gaussian
        rng.normal(0, 5, N),       # Larger random
        np.where(rng.random(N) > 0.5, 3.0, -3.0),  # Random binary-like
    ]
    
    for x0 in initializations:
        if time.time() - start_time > deadline:
            break
        
        try:
            # First pass: moderate precision
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False})
            h_candidate = np.tanh(res.x)
            val_candidate = _compute_c3_value(h_candidate)
            
            if val_candidate < best_value:
                best_value = val_candidate
                best_h = h_candidate.copy()
                
            # If we have time, do a polish pass from this solution
            if time.time() - start_time < deadline - 5.0:
                x_polish = res.x.copy()
                res2 = minimize(objective, x_polish, method='L-BFGS-B', 
                               options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
                h_polished = np.tanh(res2.x)
                val_polished = _compute_c3_value(h_polished)
                
                if val_polished < best_value:
                    best_value = val_polished
                    best_h = h_polished.copy()
                    
        except Exception:
            continue
    
    # Final: try a few more random restarts if time permits
    if time.time() - start_time < deadline - 10.0:
        for _ in range(5):
            if time.time() - start_time > deadline - 5.0:
                break
            x0 = rng.normal(0, 3, N)
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False})
                h_candidate = np.tanh(res.x)
                val_candidate = _compute_c3_value(h_candidate)
                
                if val_candidate < best_value:
                    best_value = val_candidate
                    best_h = h_candidate.copy()
                    
                    # Polish if improved
                    if time.time() - start_time < deadline - 5.0:
                        x_polish = res.x.copy()
                        res2 = minimize(objective, x_polish, method='L-BFGS-B', 
                                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
                        h_polished = np.tanh(res2.x)
                        val_polished = _compute_c3_value(h_polished)
                        
                        if val_polished < best_value:
                            best_value = val_polished
                            best_h = h_polished.copy()
            except Exception:
                continue
    
    return best_h


def construct():
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END
