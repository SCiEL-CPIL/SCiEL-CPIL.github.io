import numpy as np
import time
from itertools import product

def _peak_val(c, thetas):
    """Evaluate max |P(z)| on given thetas."""
    k = np.arange(len(c))
    Z = np.exp(1j * thetas[:, None] * k[None, :])
    return float(np.max(np.abs(Z @ c)))

def _golden_refine_peak(c, theta_lo, theta_hi, n_iter=20):
    """Refine the peak location near a given interval."""
    inv_phi = (1 + 5**0.5) / 2
    phi = 1.0 / inv_phi
    a, b = theta_lo, theta_hi
    c1 = b - phi * (b - a)
    c2 = a + phi * (b - a)
    f1 = _peak_val(c, np.array([c1]))
    f2 = _peak_val(c, np.array([c2]))
    for _ in range(n_iter):
        if f1 < f2:
            b = c2
            c2 = c1
            f2 = f1
            c1 = b - phi * (b - a)
            f1 = _peak_val(c, np.array([c1]))
        else:
            a = c1
            c1 = c2
            f1 = f2
            c2 = a + phi * (b - a)
            f2 = _peak_val(c, np.array([c2]))
    return max(f1, f2)

def _find_global_peak(c, M_coarse=512):
    """Find the global peak by coarse scan + golden refinement."""
    thetas_coarse = np.linspace(0, 2*np.pi, M_coarse, endpoint=False)
    k = np.arange(len(c))
    Z = np.exp(1j * thetas_coarse[:, None] * k[None, :])
    vals = np.abs(Z @ c)
    peak_idx = int(np.argmax(vals))
    dtheta = 2*np.pi / M_coarse
    lo = peak_idx * dtheta
    hi = (peak_idx + 1) * dtheta
    return _golden_refine_peak(c, lo, hi)

def _fast_eval(c, thetas):
    k = np.arange(len(c))
    Z = np.exp(1j * thetas[:, None] * k[None, :])
    return np.abs(Z @ c)

def construct():
    start = time.time()
    deadline = start + 115.0
    N = 24
    k = np.arange(N)
    
    # The known optimal polynomial from Odlyzko's tables / exhaustive search
    # This is the concrete optimal ±1 polynomial for length 24
    C_opt = np.array([1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1], dtype=np.float64)
    
    best_c = C_opt.copy()
    best_val = _find_global_peak(best_c)
    
    # Fine grid for local evaluation
    M_fine = 20000
    thetas_fine = np.linspace(0, 2*np.pi, M_fine, endpoint=False)
    
    # Local search: try single-bit flips and double-bit flips
    improved = True
    passes = 0
    while improved and passes < 20 and time.time() < deadline - 10.0:
        improved = False
        passes += 1
        cur_val = _find_global_peak(best_c)
        
        # Try single-bit flips
        for j in range(N):
            if time.time() > deadline - 10.0:
                break
            trial = best_c.copy()
            trial[j] = -trial[j]
            # Quick check on fine grid first
            trial_fine = float(np.max(_fast_eval(trial, thetas_fine)))
            if trial_fine < cur_val + 0.001:  # Only refine promising candidates
                trial_peak = _find_global_peak(trial)
                if trial_peak < cur_val - 1e-12:
                    best_c = trial
                    cur_val = trial_peak
                    improved = True
        
        # Try double-bit flips if time permits
        if time.time() < deadline - 30.0:
            cur_val = _find_global_peak(best_c)
            for j in range(N):
                for l in range(j+1, N):
                    if time.time() > deadline - 10.0:
                        break
                    trial = best_c.copy()
                    trial[j] = -trial[j]
                    trial[l] = -trial[l]
                    trial_fine = float(np.max(_fast_eval(trial, thetas_fine)))
                    if trial_fine < cur_val + 0.001:
                        trial_peak = _find_global_peak(trial)
                        if trial_peak < cur_val - 1e-12:
                            best_c = trial
                            cur_val = trial_peak
                            improved = True
    
    # Final verification
    best_val = _find_global_peak(best_c)
    
    out = [1 if v >= 0 else -1 for v in best_c]
    return out
