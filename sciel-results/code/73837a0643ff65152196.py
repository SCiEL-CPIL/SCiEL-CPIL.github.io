"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using Zadoff-Chu sequences modulated by chirps and windowed.
    """
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    def evaluate(h):
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2 * N * m / (s * s))
        return val

    t = np.arange(N, dtype=np.float64)
    
    def get_h(params):
        mu, alpha, win_param, dc = params
        # Zadoff-Chu like sequence
        zc = np.exp(-1j * np.pi * mu * t * (t + 1) / N)
        # Chirp modulation
        chirp = np.exp(1j * np.pi * alpha * t * t / N)
        signal = zc * chirp
        
        # Windowing: Gaussian or Cosine
        if win_param < 1.0:
            n_norm = (t - (N-1)/2.0) / ((N-1)/2.0)
            hann = 0.5 * (1 + np.cos(np.pi * n_norm))
            window = (1 - win_param) * 1.0 + win_param * hann
        else:
            sigma = win_param - 1.0 + 0.1
            n_norm = (t - (N-1)/2.0) / ((N-1)/2.0)
            window = np.exp(-0.5 * (n_norm / (sigma/2.0))**2)
        
        h_real = np.real(signal) * window + dc
        return h_real

    # Use Differential Evolution for global search
    # Bounds: mu, alpha, win_param, dc
    bounds = [
        (0, 20),       # mu
        (-1, 1),       # alpha
        (0, 3),        # win_param
        (-2, 2)        # dc
    ]
    
    np.random.seed(42)
    
    try:
        res_de = differential_evolution(
            lambda p: evaluate(get_h(p)),
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            polish=False
        )
        
        best_candidate_val = res_de.fun
        best_params = res_de.x
        best_candidate = get_h(best_params).copy()
        
        # Local refinement with Powell
        res_local = minimize(
            lambda p: evaluate(get_h(p)),
            best_params,
            method='Powell',
            options={'maxiter': 500, 'maxfev': 5000}
        )
        
        if res_local.fun < best_candidate_val:
            best_candidate_val = res_local.fun
            best_params = res_local.x
            best_candidate = get_h(best_params).copy()
            
    except Exception:
        pass

    if best_candidate is not None and best_candidate_val < best_val:
        return best_candidate
    return best_h
# EVOLVE-BLOCK-END
