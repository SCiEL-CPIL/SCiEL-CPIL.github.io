"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

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
    t_centered = t - (N - 1) / 2.0

    def get_h(params):
        mu, alpha, beta, gamma, win_param, dc = params
        # Phase: quadratic + linear + constant
        # ZC-like: pi * mu * t*(t+1)/N
        # Chirp: pi * alpha * t^2 / N
        # Linear: beta * t
        # Const: gamma
        phase = np.pi * mu * t * (t + 1) / N + np.pi * alpha * t * t / N + beta * t + gamma
        signal = np.exp(1j * phase)
        
        # Windowing
        if win_param < 1.0:
            # Cosine window interpolation
            n_norm = t_centered / ((N - 1) / 2.0)
            hann = 0.5 * (1 + np.cos(np.pi * n_norm))
            window = (1 - win_param) * 1.0 + win_param * hann
        else:
            # Gaussian window
            sigma = (win_param - 1.0) + 0.1
            n_norm = t_centered / ((N - 1) / 2.0)
            window = np.exp(-0.5 * (n_norm / (sigma / 2.0))**2)
        
        h_real = np.real(signal) * window + dc
        return h_real

    # Define bounds for differential evolution
    # mu: 1 to 20
    # alpha: -1 to 1
    # beta: -pi to pi
    # gamma: -pi to pi
    # win: 0 to 3
    # dc: -2 to 2
    bounds = [
        (1.0, 20.0),
        (-1.0, 1.0),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (0.0, 3.0),
        (-2.0, 2.0)
    ]

    def objective(params):
        h = get_h(params)
        return evaluate(h)

    best_candidate = None
    best_candidate_val = float('inf')
    
    # Run Differential Evolution
    try:
        res = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            tol=1e-5,
            seed=42,
            polish=True,
            popsize=15
        )
        if res.fun < best_candidate_val:
            best_candidate_val = res.fun
            best_candidate = get_h(res.x).copy()
    except Exception:
        pass

    # Fallback grid search if DE fails or doesn't improve
    if best_candidate is None or best_candidate_val >= best_val:
         mu_vals = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
         alpha_vals = np.linspace(-0.5, 0.5, 5)
         beta_vals = [0.0, np.pi/4, np.pi/2]
         gamma_vals = [0.0, np.pi/2]
         win_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
         dc_vals = [0.0, 0.5, 1.0]
         
         for mu in mu_vals:
             for alpha in alpha_vals:
                 for beta in beta_vals:
                     for gamma in gamma_vals:
                         for w in win_vals:
                             for d in dc_vals:
                                 h_test = get_h([mu, alpha, beta, gamma, w, d])
                                 val = evaluate(h_test)
                                 if val < best_candidate_val:
                                     best_candidate_val = val
                                     best_candidate = h_test.copy()

    if best_candidate is not None and best_candidate_val < best_val:
        return best_candidate
    return best_h
# EVOLVE-BLOCK-END
