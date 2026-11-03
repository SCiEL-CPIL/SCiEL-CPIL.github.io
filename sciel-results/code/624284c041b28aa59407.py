import numpy as np
import time
from scipy.optimize import minimize

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def compute_acf(h):
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        acf = np.fft.irfft(H * np.conj(H), n=n_fft)[:N]
        return acf
    
    def objective(h):
        h = np.maximum(h, 0)
        energy = np.sum(h**2)
        if energy < 1e-12:
            return 1e6
        h = h / np.sqrt(energy)
        acf = compute_acf(h)
        R0 = acf[0]
        if R0 < 1e-12:
            return 1e6
        # Smooth max of acf[1:] / R0
        tau = 50.0
        x = acf[1:] / R0
        max_x = np.max(x)
        # Use log-sum-exp for stability
        shifted = x - max_x
        log_sum = max_x + np.log(np.sum(np.exp(tau * shifted))) / tau
        return log_sum
    
    # Initialize
    x = np.linspace(0, 1, N)
    h0 = 0.5 * (1 + np.cos(2 * np.pi * x))
    h0[h0 < 0] = 0
    h0 = h0 / np.sqrt(np.sum(h0**2))
    
    best_h = h0.copy()
    best_score = objective(h0)
    
    # Try L-BFGS-B
    try:
        result = minimize(objective, h0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'maxfun': 5000})
        if time.time() < deadline:
            h_opt = np.maximum(result.x, 0)
            energy = np.sum(h_opt**2)
            if energy > 1e-12:
                h_opt = h_opt / np.sqrt(energy)
                score = objective(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt
    except Exception:
        pass
    
    # If time remains, try other initializations
    if time.time() < deadline:
        # Try uniform
        h1 = np.ones(N) / np.sqrt(N)
        score1 = objective(h1)
        if score1 < best_score:
            best_score = score1
            best_h = h1
    
    if time.time() < deadline:
        # Try random
        rng = np.random.RandomState(42)
        h2 = rng.rand(N)
        h2 = h2 / np.sqrt(np.sum(h2**2))
        score2 = objective(h2)
        if score2 < best_score:
            best_score = score2
            best_h = h2
    
    return list(best_h)
