import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def fast_convolve(h):
        fft_size = 1
        while fft_size < 2 * N - 1:
            fft_size *= 2
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = fast_convolve(h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    # Evaluate seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    t = np.linspace(0, 1, N, endpoint=False)
    
    # Strategy: Optimize a parameterized model h = A * cos(2*pi*f*t + chirp*t^2 + phase) + DC
    # We want to minimize the value.
    
    def model(params):
        A, f, chirp, phase, dc = params
        h = A * np.cos(2 * np.pi * f * t + 0.5 * chirp * t * t + phase) + dc
        return h

    def objective(params):
        h = model(params)
        val, valid = evaluate(h)
        if not valid:
            return 1e10
        return val

    # Grid search for good starting points
    best_params = None
    best_val = best_value
    
    # Focus on frequencies that might create constructive interference in sum but destructive in peak?
    # Actually, we just want to find any basin.
    
    for f in np.arange(1, 20, 1):
        for chirp in np.arange(-100, 101, 20):
            for dc in np.arange(-2.0, 2.1, 0.5):
                for A in [1.0]:
                    for phase in [0.0, np.pi/2]:
                        if time.time() - start_time > DEADLINE - 10:
                            break
                        params = [A, f, chirp, phase, dc]
                        val = objective(params)
                        if val < best_val:
                            best_val = val
                            best_params = params.copy()
                    if time.time() - start_time > DEADLINE - 10:
                        break
                if time.time() - start_time > DEADLINE - 10:
                    break
            if time.time() - start_time > DEADLINE - 10:
                break
        if time.time() - start_time > DEADLINE - 10:
            break

    if best_params is not None:
        # Local refinement using L-BFGS-B
        try:
            res = minimize(objective, best_params, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success or res.fun < best_val:
                h_opt = model(res.x)
                val_opt, valid = evaluate(h_opt)
                if valid and val_opt < best_value:
                    best_value = val_opt
                    best_h = h_opt
        except Exception:
            pass

    # Try random Fourier synthesis
    rng = np.random.RandomState(42)
    for _ in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break
        n_comp = rng.randint(2, 10)
        freqs = rng.randint(1, 50, n_comp)
        phases = rng.uniform(0, 2 * np.pi, n_comp)
        amps = rng.uniform(0.5, 2.0, n_comp)
        h_fourier = np.zeros(N)
        for f, p, a in zip(freqs, phases, amps):
            h_fourier += a * np.cos(2 * np.pi * f * t + p)
        
        # Add DC to ensure S != 0 and potentially large
        # We want S large.
        # If h_fourier has mean 0, S=0.
        # Add DC = mean(h_fourier) * k?
        # Let's just try a few DC offsets
        for dc in [0.0, 1.0, -1.0, 2.0, -2.0]:
            h_try = h_fourier + dc
            val_try, valid = evaluate(h_try)
            if valid and val_try < best_value:
                best_value = val_try
                best_h = h_try

    return best_h
