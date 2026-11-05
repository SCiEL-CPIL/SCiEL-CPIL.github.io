"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using a real-valued signal from Zadoff-Chu * chirp * window + DC offset.
    Improved: faster DC optimization, more window types, better search.
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
    
    # Precompute window shapes for different types
    # win_type: 0=Hann, 1=Gaussian, 2=Blackman-Harris, 3=Flat-top
    def get_window(win_type, win_param):
        if win_type < 0.5:
            # Hann window
            w = 0.5 + 0.5 * np.cos(np.pi * n_norm * win_param)
            w = np.clip(w, 0, None)
        elif win_type < 1.5:
            # Gaussian window
            sigma = 0.5 / (1.0 + win_param)
            w = np.exp(-0.5 * (n_norm / sigma) ** 2)
        elif win_type < 2.5:
            # Blackman-Harris window (approximated with parameter control)
            # Standard BH: 0.35875 - 0.48829*cos(pi*x) + 0.14128*cos(2*pi*x) - 0.01168*cos(3*pi*x)
            # We can scale the argument to control width
            x = n_norm * win_param
            w = 0.35875 - 0.48829 * np.cos(np.pi * x) + 0.14128 * np.cos(2 * np.pi * x) - 0.01168 * np.cos(3 * np.pi * x)
            w = np.clip(w, 0, None)
        else:
            # Flat-top window (approximation)
            x = n_norm * win_param
            w = 1.0 - 0.42 * np.cos(np.pi * x) + 0.05 * np.cos(2 * np.pi * x)
            w = np.clip(w, 0, None)
        return w

    def get_h_ac(params):
        """
        Returns the AC component (without DC) of the signal.
        params: [mu, alpha, beta, win_type, win_param]
        """
        mu, alpha, beta, win_type, win_param = params
        
        # Zadoff-Chu like sequence
        zc = np.exp(-1j * np.pi * mu * t * (t + 1) / N)
        # Chirp modulation
        chirp = np.exp(1j * np.pi * alpha * t * t / N)
        # Global phase
        phase = np.exp(1j * beta)
        
        signal = zc * chirp * phase
        
        w = get_window(win_type, win_param)
        
        h_ac = np.real(signal) * w
        return h_ac

    def optimize_dc_fast(x):
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
        
        # Try key candidate points
        candidates = [0.0, -S_x/N, -0.5*S_x/N, 0.5*S_x/N, -1.0*S_x/N, 1.0*S_x/N]
        
        # Add grid points around 0 and -S_x/N
        for c in np.linspace(-2, 2, 11):
            candidates.append(c)
        
        # Remove duplicates and sort
        candidates = list(set(candidates))
        
        for c_start in candidates:
            # Quick local refinement
            res = minimize_scalar(obj, bounds=(c_start-0.5, c_start+0.5), method='bounded', options={'xatol': 1e-6})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x
        
        return x + best_c

    def get_h_full(params):
        x = get_h_ac(params)
        h = optimize_dc_fast(x)
        return h

    def objective(params):
        try:
            h = get_h_full(params)
            return evaluate(h)
        except Exception:
            return float('inf')

    np.random.seed(42)
    
    # Number of starts
    n_starts = 80
    
    best_val_found = float('inf')
    best_params_found = None
    best_h_found = None
    
    start_time = time.time()
    max_time = 100
    
    for i in range(n_starts):
        if time.time() - start_time > max_time:
            break
            
        # Structured starts focusing on different window types and parameters
        if i < 20:
            # Hann windows with varying width
            mu = i * 1.5
            alpha = np.random.uniform(-1.5, 1.5)
            beta = np.random.uniform(0, 2 * np.pi)
            win_type = 0.0
            win_param = np.random.uniform(0.5, 1.5)
        elif i < 40:
            # Gaussian windows
            mu = np.random.uniform(0, 15)
            alpha = np.random.uniform(-2, 2)
            beta = np.random.uniform(0, 2 * np.pi)
            win_type = 1.0
            win_param = np.random.uniform(0, 1.5)
        elif i < 60:
            # Blackman-Harris windows
            mu = np.random.uniform(0, 15)
            alpha = np.random.uniform(-2, 2)
            beta = np.random.uniform(0, 2 * np.pi)
            win_type = 2.0
            win_param = np.random.uniform(0.5, 1.5)
        else:
            # Flat-top and random mix
            mu = np.random.uniform(0, 20)
            alpha = np.random.uniform(-2.5, 2.5)
            beta = np.random.uniform(0, 2 * np.pi)
            win_type = np.random.choice([0, 1, 2, 3])
            win_param = np.random.uniform(0.5, 1.5)
        
        x0 = np.array([mu, alpha, beta, win_type, win_param])
        
        # Powell method for local optimization
        res = minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': 150, 'maxfev': 1500, 'xatol': 1e-5, 'fatol': 1e-7}
        )
        
        if res.fun < best_val_found:
            best_val_found = res.fun
            best_params_found = res.x.copy()
            best_h_found = get_h_full(best_params_found).copy()
            
            # Early termination
            if best_val_found < 1.02:
                break
    
    if best_params_found is not None and best_val_found < best_val:
        return best_h_found
    return best_h
# EVOLVE-BLOCK-END
