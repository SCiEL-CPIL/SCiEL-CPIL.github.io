"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_c3_bound(h):
    """Compute the autocorrelation upper bound C3 for step heights h.
    C3 relates to the max absolute autocorrelation at nonzero lags,
    normalized by the zero-lag value."""
    h = np.asarray(h, dtype=np.float64)
    # Autocorrelation via FFT
    n = len(h)
    f = np.fft.fft(h, n=2*n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    # Zero lag
    acf0 = acf[0]
    if abs(acf0) < 1e-15:
        return np.max(np.abs(acf[1:]))
    # Peak sidelobe level (normalized)
    peak = np.max(np.abs(acf[1:])) / abs(acf0)
    return peak

def _objective(h):
    """Objective: the C3 upper bound value."""
    return _compute_c3_bound(h)

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with random signed initialization
    np.random.seed(42)
    best_h = np.random.choice([-1.0, 1.0], size=N).astype(np.float64)
    best_val = _compute_c3_bound(best_h)
    
    # Try a few different initializations
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        if trial == 0:
            h0 = best_h.copy()
        else:
            h0 = np.random.choice([-1.0, 1.0], size=N).astype(np.float64)
            # Also try random continuous
            if trial == 3:
                h0 = np.random.randn(N).astype(np.float64)
        
        # Gradient descent with L-BFGS-B
        try:
            res = minimize(
                _objective,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
        except Exception:
            pass
        
        # Local search: flip individual elements
        if time.time() > deadline - 20:
            break
        
        h_local = best_h.copy()
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            current_val = _compute_c3_bound(h_local)
            # Try flipping each element
            for i in range(N):
                if time.time() > deadline - 15:
                    break
                old = h_local[i]
                h_local[i] = -old
                new_val = _compute_c3_bound(h_local)
                if new_val < current_val - 1e-12:
                    current_val = new_val
                    improved = True
                else:
                    h_local[i] = old
        
        if _compute_c3_bound(h_local) < best_val:
            best_val = _compute_c3_bound(h_local)
            best_h = h_local.copy()
    
    # Final polish: fine-tune with more iterations
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
