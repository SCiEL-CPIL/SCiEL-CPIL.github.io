"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _get_fft_params(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return size, fsize

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    size, fsize = _get_fft_params(N)
    
    # Baseline: All ones
    h_base = np.ones(N)
    best_h = h_base.copy()
    best_val = _score(best_h)
    
    # We will search over binary sequences {+1, -1}
    # Since the objective is scale-invariant, the magnitude doesn't matter, only the pattern.
    # Actually, for signed step functions, any non-zero scalar multiple is equivalent.
    # So we can just work with +/- 1.

    rng = random.Random(42)
    
    # Try a few initial patterns
    inits = []
    inits.append(np.ones(N))
    inits.append(np.ones(N) * -1)
    
    # Random
    for _ in range(10):
        inits.append(np.array([1 if rng.random() > 0.5 else -1 for _ in range(N)], dtype=np.float64))
        
    # Structured: Chirp-like
    x = np.arange(N)
    inits.append(np.sign(np.cos(np.pi * x**2 / N)))
    inits.append(np.sign(np.cos(np.pi * x / N)))
    
    best_h = inits[0].copy()
    best_val = _score(best_h)
    
    for h_init in inits:
        if time.time() > deadline - 30:
            break
            
        # Simulated Annealing / Stochastic Local Search
        # We need to update FFT efficiently.
        # H = FFT(h)
        # If we flip h[i] -> -h[i], delta_h = -2*h[i] at index i.
        # delta_H = delta_h * exp(-j * 2*pi * i * k / fsize)
        # This is a rank-1 update to H.
        
        h = h_init.copy()
        # Normalize to +/- 1 for consistency, though it doesn't strictly matter for the ratio
        # unless we hit numerical issues.
        h = np.sign(h)
        if np.sum(h) == 0:
             # Avoid zero sum
             h[0] = 1.0
             if h[0] == 1.0: h[0] = -1.0 # Just to be safe, pick a non-zero
             # Actually just ensure it's not all cancelling. 
             # For N=400, random +/-1 almost surely has non-zero sum.
        
        H = np.fft.rfft(h, fsize)
        
        # Precompute twiddle factors for each index i
        # twiddles[i] = exp(-j * 2*pi * i * k / fsize) for k in 0..fsize/2
        # We can compute this on the fly or precompute.
        # Precomputing for N=400 and fsize=1024 is 400 * 512 complex numbers. ~200k floats. Fine.
        k_indices = np.arange(fsize // 2 + 1)
        # twiddles[i, k] = exp(-j * 2*pi * i * k / fsize)
        # This is a Vandermonde-like matrix.
        # twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), k_indices) / fsize)
        # This might be memory heavy if N was larger, but 400*512 is small.
        twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), k_indices) / fsize)
        
        current_val = _score(h)
        
        # Annealing schedule
        # Start T high, decrease
        T_start = 1.0
        T_end = 0.01
        alpha = 0.999 # Cooling factor per step?
        
        # Estimate steps possible
        # _score involves FFT (fsize log fsize) and max.
        # Update H is O(fsize).
        # Let's run a fixed number of steps or until time.
        
        max_steps = 20000
        steps_per_iter = 0
        T = T_start
        
        # Optimize: we can check the score of the *new* state.
        # We need to update H, compute AC, check score.
        # If accepted, keep H and h.
        # If rejected, revert H and h.
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
                
            i = rng.randint(0, N - 1)
            
            # Flip h[i]
            old_val = h[i]
            new_val = -old_val
            h[i] = new_val
            
            # Update H
            delta = -2 * old_val
            H += delta * twiddles[i]
            
            # Compute new score
            ac = np.fft.irfft(H * H, fsize)
            new_ac_max = np.max(np.abs(ac))
            s = np.sum(h)
            if abs(s) < 1e-9:
                new_score = 1e10
            else:
                new_score = new_ac_max / (s * s)
            
            # Acceptance criterion
            delta_score = new_score - current_val
            if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
                current_val = new_score
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                # Revert
                h[i] = old_val
                H -= delta * twiddles[i]
            
            # Cool down
            T *= 0.9995 # Slow cooling
            if T < T_end:
                T = T_end
                
    # Final local search on the best found binary sequence
    # Try single flips to see if we can improve further (greedy hill climb)
    h = best_h.copy()
    H = np.fft.rfft(h, fsize)
    current_val = _score(h)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old_val = h[i]
            new_val = -old_val
            h[i] = new_val
            delta = -2 * old_val
            H += delta * twiddles[i]
            
            ac = np.fft.irfft(H * H, fsize)
            new_ac_max = np.max(np.abs(ac))
            s = np.sum(h)
            if abs(s) < 1e-9:
                new_score = 1e10
            else:
                new_score = new_ac_max / (s * s)
            
            if new_score < current_val:
                current_val = new_score
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
                improved = True
            else:
                h[i] = old_val
                H -= delta * twiddles[i]
                
    return best_h

# EVOLVE-BLOCK-END
