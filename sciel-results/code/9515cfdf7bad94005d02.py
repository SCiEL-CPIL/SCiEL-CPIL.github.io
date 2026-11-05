"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

def _autoconv(h):
    """Full autoconvolution of h with itself."""
    return np.convolve(h, h, mode='full')

def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    return abs(2.0 * N * np.max(ac) / (s * s))

def _soft_max(x, t=0.01):
    """Soft max approximation using log-sum-exp."""
    x_max = np.max(x)
    return x_max + t * np.log(np.sum(np.exp((x - x_max) / t)))

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Strategy: try to make max(autoconv) negative and close to zero
    # Use a base signal with negative autocorrelation properties
    
    # Initialize with alternating sign pattern windowed
    x = np.arange(N)
    # Create a signal that has negative autocorrelation: 
    # A high-frequency oscillation windowed
    base_freq = 0.45  # close to Nyquist for negative autocorrelation
    h0 = np.cos(2 * np.pi * base_freq * x / N * N)  # cos(2*pi*base_freq*x)
    # Apply a window to reduce edge effects
    window = np.hanning(N)
    h0 = h0 * window
    # Normalize sum to 1
    s0 = np.sum(h0)
    if abs(s0) > 1e-10:
        h0 = h0 / s0
    
    # Check initial value
    val0 = _value(h0)
    if val0 < best_val:
        best_h = h0.copy()
        best_val = val0
    
    # Optimize: minimize soft_max(autoconv(h)) subject to sum(h)=1
    # Use parameterization: h = h0 + delta, with sum constraint handled by optimization
    
    def objective(params):
        h = params.copy()
        # Enforce sum(h) = 1 by projecting
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
        # The evaluator uses sum(h)^2 in denominator, so we can scale
        # But for optimization, let's fix sum(h)=1
        h = h / s
        ac = _autoconv(h)
        # We want max(ac) to be as negative as possible (or close to 0 from below)
        # Minimize soft_max(ac)
        t = 0.05
        return _soft_max(ac, t)
    
    # Start from h0
    x0 = h0.copy()
    
    # Use L-BFGS-B for optimization
    try:
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            h_opt = result.x / np.sum(result.x)
            val_opt = _value(h_opt)
            if val_opt < best_val:
                best_h = h_opt.copy()
                best_val = val_opt
    except Exception:
        pass
    
    # Try different initializations if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        # Try a few different base frequencies
        for freq in [0.4, 0.45, 0.5, 0.55, 0.6]:
            if time.time() > deadline - 40:
                break
            h_trial = np.cos(2 * np.pi * freq * np.arange(N)) * np.hanning(N)
            s_trial = np.sum(h_trial)
            if abs(s_trial) < 1e-10:
                continue
            h_trial = h_trial / s_trial
            val_trial = _value(h_trial)
            if val_trial < best_val:
                best_h = h_trial.copy()
                best_val = val_trial
            
            # Quick optimization from this start
            try:
                res = minimize(
                    objective,
                    h_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
                )
                h_opt2 = res.x / np.sum(res.x)
                val_opt2 = _value(h_opt2)
                if val_opt2 < best_val:
                    best_h = h_opt2.copy()
                    best_val = val_opt2
            except Exception:
                pass
    
    # Final polish: try to reduce the max autoconv entry further
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Local refinement: minimize max(ac) directly with smaller t
        def objective_fine(params):
            h = params / np.sum(params)
            ac = _autoconv(h)
            t = 0.01
            return _soft_max(ac, t)
        
        try:
            res_fine = minimize(
                objective_fine,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            h_fine = res_fine.x / np.sum(res_fine.x)
            val_fine = _value(h_fine)
            if val_fine < best_val:
                best_h = h_fine.copy()
                best_val = val_fine
        except Exception:
            pass
    
    # Ensure finite and sum not zero
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
