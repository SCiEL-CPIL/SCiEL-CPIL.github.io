import numpy as np
from scipy.optimize import minimize
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 110  # safety margin before 120s kill

def objective(x):
    """Minimize the target value."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    global START_TIME, TIME_LIMIT
    if time.time() - START_TIME > TIME_LIMIT:
        return np.ones(N)
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Start with a random initialization
    rng = np.random.default_rng(42)
    
    # Try several random starts
    for trial in range(3):
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        
        # Initialize: mostly positive with some structure
        # Use a low-frequency dominant signal
        freqs = np.fft.rfftfreq(N)
        # Weight low frequencies more heavily
        weights = 1.0 / (1.0 + 50.0 * freqs)
        weights[0] = 1.0  # DC
        
        # Random spectral profile
        spec_mag = rng.random(N // 2 + 1) * weights
        spec_phase = rng.uniform(-np.pi, np.pi, N // 2 + 1)
        spec = spec_mag * np.exp(1j * spec_phase)
        spec[0] = spec_mag[0]  # Keep DC real
        
        h_init = np.fft.irfft(spec, n=N)
        h_init = h_init - np.mean(h_init)  # Remove DC to avoid sum=0 issues? No, keep some DC
        # Actually, let's add back some DC
        h_init = h_init + 0.5
        
        val_init = objective(h_init)
        if val_init < best_val:
            best_val = val_init
            best_h = h_init.copy()
        
        # Local optimization
        try:
            result = minimize(
                objective,
                h_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
        except Exception:
            pass
        
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
    
    # Final check
    final_val = objective(best_h)
    if final_val < best_val:
        best_val = final_val
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h
