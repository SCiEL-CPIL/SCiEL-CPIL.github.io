"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using Zadoff-Chu sequences with chirp modulation and optimized windowing.
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
        Returns the AC component of the signal.
        params: [mu, alpha, beta, win_width, win_alpha]
        mu: Zadoff-Chu parameter (0 to 20)
        alpha: Chirp rate (-2 to 2)
        beta: Global phase (0 to 2*pi)
        win_width: Window transition width (0.1 to 1.0)
        win_alpha: Window steepness (1 to 10)
        """
        mu, alpha, beta, win_width, win_alpha = params
        
        # Zadoff-Chu like sequence
        zc = np.exp(-1j * np.pi * mu * t * (t + 1) / N)
        # Chirp modulation
        chirp = np.exp(1j * np.pi * alpha * t * t / N)
        # Global phase
        phase = np.exp(1j * beta)
        
        signal = zc * chirp * phase
        
        # Smooth window with adjustable width and steepness
        # Use a smoothstep-like window
        edge = win_width
        left_edge = -1.0 + edge
        right_edge = 1.0 - edge
        
        window = np.ones(N)
        # Left transition
        mask_left = n_norm < left_edge
        if np.any(mask_left):
            x = (n_norm[mask_left] - (-1.0)) / edge
            x = np.clip(x, 0, 1)
            window[mask_left] = x**win_alpha / (x**win_alpha + (1-x)**win_alpha)
        
        # Right transition
        mask_right = n_norm > right_edge
        if np.any(mask_right):
            x = (1.0 - n_norm[mask_right]) / edge
            x = np.clip(x, 0, 1)
            window[mask_right] = x**win_alpha / (x**win_alpha + (1-x)**win_alpha)
        
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
        
        best_c = 0.0
        best_obj = obj(0.0)
        
        # Try multiple starting points
        for c_start in np.linspace(-2, 2, 9):
            res = minimize_scalar(obj, bounds=(c_start-0.3, c_start+0.3), method='bounded', options={'xatol': 1e-8})
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

    # Systematic search with multiple strategies
    np.random.seed(42)
    
    best_val_found = float('inf')
    best_params_found = None
    best_h_found = None
    
    # Strategy 1: Vary mu systematically with random other params
    for mu in [1, 2, 3, 5, 7, 10, 13, 17]:
        for _ in range(5):
            alpha = np.random.uniform(-1.5, 1.5)
            beta = np.random.uniform(0, 2 * np.pi)
            win_width = np.random.uniform(0.15, 0.5)
            win_alpha = np.random.uniform(2, 8)
            
            x0 = np.array([mu, alpha, beta, win_width, win_alpha])
            res = minimize(
                objective,
                x0,
                method='Powell',
                options={'maxiter': 200, 'maxfev': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res.fun < best_val_found:
                best_val_found = res.fun
                best_params_found = res.x.copy()
                best_h_found = get_h_full(best_params_found).copy()
                if best_val_found < 1.05:
                    break
        if best_val_found < 1.05:
            break
    
    # Strategy 2: Random exploration
    if best_val_found >= 1.05:
        for _ in range(30):
            mu = np.random.uniform(0.5, 20)
            alpha = np.random.uniform(-2, 2)
            beta = np.random.uniform(0, 2 * np.pi)
            win_width = np.random.uniform(0.1, 0.6)
            win_alpha = np.random.uniform(1.5, 10)
            
            x0 = np.array([mu, alpha, beta, win_width, win_alpha])
            res = minimize(
                objective,
                x0,
                method='Powell',
                options={'maxiter': 200, 'maxfev': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res.fun < best_val_found:
                best_val_found = res.fun
                best_params_found = res.x.copy()
                best_h_found = get_h_full(best_params_found).copy()
                if best_val_found < 1.05:
                    break
    
    # Strategy 3: Focus on narrow windows
    if best_val_found >= 1.05:
        for _ in range(20):
            mu = np.random.uniform(1, 15)
            alpha = np.random.uniform(-1, 1)
            beta = np.random.uniform(0, 2 * np.pi)
            win_width = np.random.uniform(0.05, 0.2)
            win_alpha = np.random.uniform(3, 15)
            
            x0 = np.array([mu, alpha, beta, win_width, win_alpha])
            res = minimize(
                objective,
                x0,
                method='Powell',
                options={'maxiter': 200, 'maxfev': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res.fun < best_val_found:
                best_val_found = res.fun
                best_params_found = res.x.copy()
                best_h_found = get_h_full(best_params_found).copy()
                if best_val_found < 1.05:
                    break
    
    if best_params_found is not None and best_val_found < best_val:
        return best_h_found
    return best_h
# EVOLVE-BLOCK-END
