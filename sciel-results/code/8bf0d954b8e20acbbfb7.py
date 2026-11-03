import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    """Get the next power of 2 size for FFT of linear convolution of length n."""
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp.
    Returns a scalar.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    # Clip to avoid overflow in exp
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    """
    Smooth objective for optimization.
    Returns the value to be minimized.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _generate_candidate(rng, strategy=0):
    """
    Generate a diverse candidate vector.
    strategy: 
    0: Random Gaussian
    1: Random Bernoulli (dense)
    2: Random Bernoulli (sparse)
    3: Block signs
    4: Low-pass filtered noise
    5: Cosine window with random signs
    """
    h = np.zeros(N)
    
    if strategy == 0:
        h = rng.standard_normal(N)
    elif strategy == 1:
        # Dense Bernoulli
        h = rng.choice([1.0, -1.0], N)
    elif strategy == 2:
        # Sparse Bernoulli (20% non-zero)
        mask = rng.random(N) < 0.2
        h[mask] = rng.choice([1.0, -1.0], mask.sum())
    elif strategy == 3:
        # Block signs: divide into ~10 blocks
        num_blocks = 10
        block_size = N // num_blocks
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size if i < num_blocks - 1 else N
            sign = rng.choice([1.0, -1.0])
            h[start:end] = sign
    elif strategy == 4:
        # Low-pass filtered noise
        noise = rng.standard_normal(N)
        H = np.fft.rfft(noise)
        # Keep only low frequencies
        cutoff = 10
        H[cutoff:] = 0
        h = np.fft.irfft(H, N)
        # Optional: sign flip based on zero crossings? No, let optimizer handle it.
        # But ensure it's not zero
        if np.sum(np.abs(h)) < 1e-6:
             h = rng.standard_normal(N)
    elif strategy == 5:
        # Cosine window envelope, random signs
        x = np.arange(N)
        env = np.cos(np.pi * x / N) ** 2
        signs = rng.choice([1.0, -1.0], N)
        h = env * signs
    else:
        h = rng.standard_normal(N)
        
    return h

def _simulated_annealing(h_init, deadline, rng, T0=1.0, cooling=0.999, min_T=1e-6):
    """
    Simulated Annealing to refine a candidate.
    Uses multi-bit flips and exact score evaluation.
    """
    h = h_init.copy()
    # Normalize initial state
    s = np.sum(h)
    if abs(s) < 1e-9:
        h = rng.standard_normal(N)
        s = np.sum(h)
    h = h / s
    
    current_score = _score(h)
    best_score = current_score
    best_h = h.copy()
    
    T = T0
    i = 0
    while time.time() < deadline and T > min_T:
        i += 1
        # Generate neighbor: flip a random number of signs
        # The number of flips can vary. Early on, more flips for exploration.
        # n_flips = max(1, int(N * T * 0.05)) # Scale with T?
        # Let's try a fixed small number or random between 1 and 5
        n_flips = rng.integers(1, 6)
        indices = rng.choice(N, n_flips, replace=False)
        
        # Save original values
        orig_vals = h[indices].copy()
        h[indices] = -h[indices]
        
        # Evaluate
        new_score = _score(h)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Probability of accepting worse solution
            # delta = new_score - current_score
            # p = exp(-delta / T)
            # Note: Score is positive.
            # If T is small, p is small.
            if T > 0:
                delta = new_score - current_score
                # Clamp delta to avoid overflow in exp if T is very small
                # If delta is large, exp(-large/T) is 0.
                if delta / T < 700: # exp(-700) is ~ 1e-304, effectively 0
                    p = np.exp(-delta / T)
                    if rng.random() < p:
                        current_score = new_score
                    else:
                        # Revert
                        h[indices] = orig_vals
                else:
                    # Revert
                    h[indices] = orig_vals
            else:
                # Revert
                h[indices] = orig_vals
        
        # Cool down
        T *= cooling
        
        # If we are stuck for a while, we might want to restart or increase T?
        # For now, simple cooling.
        
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin for 120s limit
    
    best_score = np.inf
    best_h = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    strategies = [0, 1, 2, 3, 4, 5]
    
    # Generate initial candidates
    num_initial = 20
    candidates = []
    for i in range(num_initial):
        if time.time() > deadline - 20:
            break
        strat = strategies[i % len(strategies)]
        h = _generate_candidate(rng, strat)
        s = np.sum(h)
        if abs(s) < 1e-9:
            continue
        h = h / s
        val = _score(h)
        candidates.append((val, h))
        
    candidates.sort(key=lambda x: x[0])
    
    # Run SA on top candidates
    # Allocate time based on rank
    # Top 5 get more time
    for i, (init_val, h_init) in enumerate(candidates[:8]):
        if time.time() > deadline - 5:
            break
        
        # Give better candidates more time?
        # Or just split remaining time.
        remaining = deadline - time.time()
        # Simple heuristic: give 10% of remaining time to each of the top few,
        # but ensure we don't overshoot.
        # Let's just give a fixed chunk of time to each, say 5-10 seconds.
        local_deadline = min(deadline - 2.0, time.time() + 8.0)
        
        h_opt, score_opt = _simulated_annealing(h_init, local_deadline, rng)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Final polish on the best found
    if time.time() < deadline - 5.0 and best_score < np.inf:
        final_deadline = deadline
        h_polished, score_polished = _simulated_annealing(best_h, final_deadline, rng, T0=0.1, cooling=0.9999)
        
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished
            
    # Ensure valid return
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
