"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using a real-valued signal constructed from the real part of a product of
    a Zadoff-Chu-like complex exponential and a chirp, modulated by a raised-cosine
    window, with an optimized DC offset.
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
    n_norm = (t - (N-1)/2.0) / ((N-1)/2.0)

    def get_h_ac(params):
        """
        Returns the AC component (without DC) of the signal.
        params: [mu, alpha, beta, win_param]
        mu: Zadoff-Chu parameter
        alpha: Chirp rate
        beta: Global phase
        win_param: Window shape parameter (0=rect, 1=full raised cosine)
        """
        mu, alpha, beta, win_param = params
        
        # Zadoff-Chu like sequence
        zc = np.exp(-1j * np.pi * mu * t * (t + 1) / N)
        # Chirp modulation
        chirp = np.exp(1j * np.pi * alpha * t * t / N)
        # Global phase
        phase = np.exp(1j * beta)
        
        signal = zc * chirp * phase
        
        # Raised cosine window
        # win_param in [0, 1]
        # 0 -> rectangular
        # 1 -> full raised cosine (Hann)
        # w(t) = 1 - 0.5*win_param + 0.5*win_param*cos(pi*n_norm)
        # This is smooth and differentiable with respect to win_param
        window = 1.0 - 0.5 * win_param + 0.5 * win_param * np.cos(np.pi * n_norm)
        
        h_ac = np.real(signal) * window
        return h_ac

    def optimize_dc(x):
        """Finds the optimal DC offset c to add to x to minimize the value."""
        S_x = np.sum(x)
        C_x = np.convolve(x, x)
        U = np.convolve(x, np.ones(N))
        T = np.convolve(np.ones(N), np.ones(N))
        
        def obj(c):
            conv_c = C_x + 2*c*U + c**2*T
            m = np.max(conv_c)
            s = S_x + N*c
            if abs(s) < 1e-12:
                return float('inf')
            val = abs(2 * N * m / (s * s))
            return val
        
        # Use multiple local minima searches for DC offset
        # The function can be multimodal
        best_c = 0.0
        best_obj = obj(0.0)
        
        # Try a grid of starting points
        for c_start in np.linspace(-3, 3, 13):
            res = minimize_scalar(obj, bounds=(c_start-0.5, c_start+0.5), method='bounded', options={'xatol': 1e-8})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x
        
        return x + best_c

    def get_h_full(params):
        x = get_h_ac(params)
        h = optimize_dc(x)
        return h

    def objective(params):
        try:
            h = get_h_full(params)
            return evaluate(h)
        except Exception:
            return float('inf')

    # Multi-start local optimization with different initializations
    # Parameters: mu, alpha, beta, win_param
    # Bounds:
    # mu: 0 to 25
    # alpha: -2 to 2
    # beta: 0 to 2*pi
    # win_param: 0 to 1
    
    np.random.seed(12345)
    
    # Number of starts - balance between exploration and time
    n_starts = 80
    
    best_val_found = float('inf')
    best_params_found = None
    best_h_found = None
    
    for i in range(n_starts):
        # Mix of random and structured starts
        if i < 20:
            # Structured starts: vary mu systematically
            mu = i * 1.25
            alpha = np.random.uniform(-1, 1)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0.2, 1.0)
        elif i < 40:
            # Random mu in wider range
            mu = np.random.uniform(0, 25)
            alpha = np.random.uniform(-2, 2)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0, 1)
        elif i < 60:
            # Focus on narrow windows (often better for suppressing sidelobes)
            mu = np.random.uniform(0, 25)
            alpha = np.random.uniform(-2, 2)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0.5, 1.0)
        else:
            # Fully random
            mu = np.random.uniform(0, 30)
            alpha = np.random.uniform(-3, 3)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0, 1)
        
        x0 = np.array([mu, alpha, beta, win_param])
        
        # Powell method for local optimization
        res = minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': 300, 'maxfev': 3000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if res.fun < best_val_found:
            best_val_found = res.fun
            best_params_found = res.x.copy()
            best_h_found = get_h_full(best_params_found).copy()
            
            # Early termination if we get a very good value
            if best_val_found < 1.05:
                break
    
    if best_params_found is not None and best_val_found < best_val:
        return best_h_found
    return best_h
# EVOLVE-BLOCK-END
