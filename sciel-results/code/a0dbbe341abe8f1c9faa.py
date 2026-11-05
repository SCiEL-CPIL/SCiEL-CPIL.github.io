"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using a parameterized complex chirp signal modulated by a window.
    """
    np.random.seed(42)
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
        mu, alpha, phi, win_param, dc = params
        # Complex signal: Zadoff-Chu like + Chirp + Phase
        phase = np.pi * (mu * t * (t + 1) / N + alpha * t * t / N) + phi
        signal = np.exp(1j * phase)
        
        # Windowing: Generalized Cosine
        # win_param in [0, 1] maps to a generalized cosine window
        n_norm = (t - (N-1)/2.0) / ((N-1)/2.0) # -1 to 1
        hann = 0.5 * (1 + np.cos(np.pi * n_norm))
        window = (1 - win_param) * 1.0 + win_param * hann
        
        h_real = np.real(signal) * window + dc
        return h_real

    best_candidate = None
    best_candidate_val = float('inf')
    best_params = None
    
    # Coarse grid search
    mu_vals = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    alpha_vals = np.linspace(-0.5, 0.5, 11)
    phi_vals = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    win_vals = [0.0, 0.5, 1.0]
    dc_vals = [-0.2, 0.0, 0.2]
    
    for mu in mu_vals:
        for alpha in alpha_vals:
            for phi in phi_vals:
                for w in win_vals:
                    for d in dc_vals:
                        h_test = get_h([mu, alpha, phi, w, d])
                        val = evaluate(h_test)
                        if val < best_candidate_val:
                            best_candidate_val = val
                            best_candidate = h_test.copy()
                            best_params = [mu, alpha, phi, w, d]
    
    # Refine best candidates
    if best_candidate is not None:
        # Local optimization
        for _ in range(5):
            if _ == 0:
                x0 = np.array(best_params)
            else:
                x0 = np.array(best_params) + np.random.normal(0, 0.1, 5)
            
            res = minimize(
                lambda p: evaluate(get_h(p)),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < best_candidate_val:
                best_candidate_val = res.fun
                best_candidate = get_h(res.x).copy()
                best_params = res.x

    if best_candidate is not None and best_candidate_val < best_val:
        return best_candidate
    return best_h
# EVOLVE-BLOCK-END
