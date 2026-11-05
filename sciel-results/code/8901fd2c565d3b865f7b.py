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

    def optimize_dc(x, prev_c=0.0):
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
        best_c = prev_c
        best_obj = obj(prev_c)
        
        # Try a grid of starting points, including the previous best
        starts = [prev_c]
        # Add grid points around 0 and prev_c
        for c_start in np.linspace(-5, 5, 21):
            starts.append(c_start)
            
        for c_start in starts:
            # Bounds around the start point
            res = minimize_scalar(obj, bounds=(c_start-1.0, c_start+1.0), method='bounded', options={'xatol': 1e-10})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x
        
        return x + best_c, best_c

    def get_h_full(params, prev_c=0.0):
        x = get_h_ac(params)
        h, new_c = optimize_dc(x, prev_c)
        return h, new_c

    def objective(params):
        try:
            # We don't have access to prev_c easily in the stateless objective for Powell,
            # so we use a default 0.0 here. The final polishing will handle refinement.
            h, _ = get_h_full(params, 0.0)
            return evaluate(h)
        except Exception:
            return float('inf')

    # Multi-start local optimization with different initializations
    # Parameters: mu, alpha, beta, win_param
    # Bounds:
    # mu: 0 to 40
    # alpha: -4 to 4
    # beta: 0 to 2*pi
    # win_param: 0 to 1
    
    np.random.seed(12345)
    
    # Number of starts - balance between exploration and time
    n_starts = 200
    
    best_val_found = float('inf')
    best_params_found = None
    best_h_found = None
    best_c_found = 0.0
    
    for i in range(n_starts):
        # Mix of random and structured starts
        if i < 50:
            # Structured starts: vary mu systematically
            mu = i * 0.8
            alpha = np.random.uniform(-2, 2)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0.2, 1.0)
        elif i < 100:
            # Random mu in wider range
            mu = np.random.uniform(0, 40)
            alpha = np.random.uniform(-4, 4)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0, 1)
        elif i < 150:
            # Focus on narrow windows (often better for suppressing sidelobes)
            mu = np.random.uniform(0, 40)
            alpha = np.random.uniform(-4, 4)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0.5, 1.0)
        else:
            # Fully random
            mu = np.random.uniform(0, 60)
            alpha = np.random.uniform(-6, 6)
            beta = np.random.uniform(0, 2 * np.pi)
            win_param = np.random.uniform(0, 1)
        
        x0 = np.array([mu, alpha, beta, win_param])
        
        # Powell method for local optimization
        res = minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': 200, 'maxfev': 2000, 'xatol': 1e-7, 'fatol': 1e-9}
        )
        
        if res.fun < best_val_found:
            best_val_found = res.fun
            best_params_found = res.x.copy()
            # Refine DC offset with the found parameters
            h_candidate, c_candidate = get_h_full(best_params_found, 0.0)
            best_h_found = h_candidate.copy()
            best_c_found = c_candidate
            
            # Early termination if we get a very good value
            if best_val_found < 1.02:
                break
    
    # Polishing step: refine the best candidate found
    if best_params_found is not None and best_val_found < best_val:
        # Run a tighter optimization on the best parameters
        # We can also try to optimize the DC offset more precisely if needed, 
        # but optimize_dc is already quite tight.
        
        # Let's do one more local refinement using Nelder-Mead on the best params
        def objective_refine(params):
            try:
                h, _ = get_h_full(params, best_c_found)
                return evaluate(h)
            except Exception:
                return float('inf')
                
        res_refine = minimize(
            objective_refine,
            best_params_found,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if res_refine.fun < best_val_found:
            best_val_found = res_refine.fun
            best_params_found = res_refine.x.copy()
            best_h_found, best_c_found = get_h_full(best_params_found, best_c_found)

        if best_val_found < best_val:
            return best_h_found
            
    return best_h
# EVOLVE-BLOCK-END
