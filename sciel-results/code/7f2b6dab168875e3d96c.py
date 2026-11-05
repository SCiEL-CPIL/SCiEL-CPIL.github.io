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
    n_norm_sq = n_norm ** 2

    def get_h_ac(params):
        """
        Returns the AC component (without DC) of the signal.
        params: [mu, alpha, beta, win_type, win_param]
        mu: Zadoff-Chu parameter
        alpha: Chirp rate
        beta: Global phase
        win_type: 0=Hann, 1=Gaussian, 2=Exponential
        win_param: Window shape parameter
        """
        mu, alpha, beta, win_type, win_param = params
        
        # Zadoff-Chu like sequence
        zc = np.exp(-1j * np.pi * mu * t * (t + 1) / N)
        # Chirp modulation
        chirp = np.exp(1j * np.pi * alpha * t * t / N)
        # Global phase
        phase = np.exp(1j * beta)
        
        signal = zc * chirp * phase
        
        # Different window types
        if win_type < 0.5:
            # Hann window: w = 0.5 + 0.5*cos(pi*n_norm), win_param controls width
            # Effective width controlled by win_param in [0.3, 1.0]
            w = 0.5 + 0.5 * np.cos(np.pi * n_norm * win_param)
            w = np.clip(w, 0, None)
        elif win_type < 1.5:
            # Gaussian window: w = exp(-0.5 * (n_norm / sigma)^2)
            sigma = 0.5 / (1.0 + win_param)  # win_param=0 -> sigma=0.5, win_param=1 -> sigma=0.25
            w = np.exp(-0.5 * (n_norm / sigma) ** 2)
        else:
            # Exponential decay from center: w = exp(-|n_norm| / tau)
            tau = 1.0 / (1.0 + win_param)  # win_param=0 -> tau=1, win_param=1 -> tau=0.5
            w = np.exp(-np.abs(n_norm) / tau)
        
        h_ac = np.real(signal) * w
        return h_ac

    def optimize_dc_fast(x):
        """Finds the optimal DC offset c to add to x to minimize the value.
        Uses a faster approach with fewer evaluations."""
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
        
        # Try key candidate points + a few local refinements
        # The optimal c often makes the peak convolve value minimal
        # Try c values that zero out or minimize the max
        candidates = [0.0, -S_x/N, -0.5*S_x/N, 0.5*S_x/N, -1.0*S_x/N, 1.0*S_x/N]
        
        # Add grid points
        for c in np.linspace(-2, 2, 9):
            candidates.append(c)
        
        for c_start in candidates:
            # Quick local refinement
            res = minimize_scalar(obj, bounds=(c_start-0.3, c_start+0.3), method='bounded', options={'xatol': 1e-6})
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

    # Multi-start local optimization with different initializations
    # Parameters: mu, alpha, beta, win_type, win_param
    # Bounds:
    # mu: 0 to 20
    # alpha: -3 to 3
    # beta: 0 to 2*pi
    # win_type: 0 to 2 (discrete, but we'll use continuous)
    # win_param: 0 to 1
    
    np.random.seed(42)
    
    # Number of starts - balance between exploration and time
    n_starts = 60
    
    best_val_found = float('inf')
    best_params_found = None
    best_h_found = None
    
    start_time = time.time()
    max_time = 100  # seconds, leave buffer for timeout
    
    for i in range(n_starts):
        if time.time() - start_time > max_time:
            break
            
        # Mix of random and structured starts
        if i < 15:
            # Structured starts: vary mu systematically, focus on Hann windows
            mu = i * 1.3
            alpha = np.random.uniform(-1.5, 1.5)
            beta = np.random.uniform(0, 2 * np.pi)
            win_type = 0.0  # Hann
            win_param = np.random.uniform(0.4, 1.0)
        elif i < 30:
            # Random with mixed window types
            mu = np.random.uniform(0, 20)
            alpha = np.random.uniform(-2.5, 2.5)
            beta = np.random.uniform(0, 2 * np.pi)
            win_type = np.random.uniform(0, 2)
            win_param = np.random.uniform(0, 1)
        elif i < 45:
            # Focus on Gaussian windows with various widths
            mu = np.random.uniform(0, 15)
            alpha = np.random.uniform(-2, 2)
            beta = np.random.uniform(0, 2 * np.pi)
            win_type = 1.0  # Gaussian
            win_param = np.random.uniform(0, 1)
        else:
            # Fully random
            mu = np.random.uniform(0, 25)
            alpha = np.random.uniform(-3, 3)
            beta = np.random.uniform(0, 2 * np.pi)
            win_type = np.random.uniform(0, 2)
            win_param = np.random.uniform(0, 1)
        
        x0 = np.array([mu, alpha, beta, win_type, win_param])
        
        # Powell method for local optimization
        res = minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': 200, 'maxfev': 2000, 'xatol': 1e-5, 'fatol': 1e-7}
        )
        
        if res.fun < best_val_found:
            best_val_found = res.fun
            best_params_found = res.x.copy()
            best_h_found = get_h_full(best_params_found).copy()
            
            # Early termination if we get a very good value
            if best_val_found < 1.02:
                break
    
    if best_params_found is not None and best_val_found < best_val:
        return best_h_found
    return best_h
# EVOLVE-BLOCK-END
