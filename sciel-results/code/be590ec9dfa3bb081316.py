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
    6: Chirp
    7: Golden Ratio
    """
    h = np.zeros(N)
    t = np.arange(N)
    
    if strategy == 0:
        h = rng.standard_normal(N)
    elif strategy == 1:
        h = rng.choice([1.0, -1.0], N)
    elif strategy == 2:
        mask = rng.random(N) < 0.2
        h[mask] = rng.choice([1.0, -1.0], mask.sum())
    elif strategy == 3:
        num_blocks = 10
        block_size = N // num_blocks
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size if i < num_blocks - 1 else N
            sign = rng.choice([1.0, -1.0])
            h[start:end] = sign
    elif strategy == 4:
        noise = rng.standard_normal(N)
        H = np.fft.rfft(noise)
        cutoff = 10
        H[cutoff:] = 0
        h = np.fft.irfft(H, N)
        if np.sum(np.abs(h)) < 1e-6:
             h = rng.standard_normal(N)
    elif strategy == 5:
        x = np.arange(N)
        env = np.cos(np.pi * x / N) ** 2
        signs = rng.choice([1.0, -1.0], N)
        h = env * signs
    elif strategy == 6:
        # Chirp
        freq = 1.0 + rng.random() * 2.0
        phase = 2 * np.pi * freq * (t**2) / N
        h = np.sign(np.cos(phase))
        h[h == 0] = 1.0
    elif strategy == 7:
        # Golden Ratio
        phi = (1 + np.sqrt(5)) / 2
        h = np.sign(np.cos(2 * np.pi * phi * t))
        h[h == 0] = 1.0
    else:
        h = rng.standard_normal(N)
        
    return h

def _local_search_block(h, deadline, max_iters=10):
    """
    Local search using block flips.
    Tries to flip blocks of size 1, 3, 5, 7.
    """
    h = h.copy()
    # Binarize to start local search on discrete domain
    h = np.sign(h)
    h[h == 0] = 1.0
    
    best_score = _score(h)
    block_sizes = [1, 3, 5, 7]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        # Shuffle order of block sizes to avoid bias
        sizes_to_try = block_sizes.copy()
        np.random.shuffle(sizes_to_try)
        
        for bs in sizes_to_try:
            if time.time() > deadline:
                break
            # Try a random subset of start positions to keep it fast
            # If we try all, it's O(N) per size. With 4 sizes, 4N evals.
            # N=400, 4*400=1600 evals. Each eval ~1ms. 1.6s per pass.
            # Too slow for many passes. 
            # Strategy: Try random flips instead of exhaustive search.
            num_flips = 50
            for _ in range(num_flips):
                if time.time() > deadline:
                    break
                start = np.random.randint(0, N - bs + 1)
                h[start:start+bs] = -h[start:start+bs]
                new_val = _score(h)
                if new_val < best_score:
                    best_score = new_val
                    improved = True
                else:
                    h[start:start+bs] = -h[start:start+bs] # revert
                    
        if not improved:
            # If no improvement, try a few more random single-bit flips
            for _ in range(20):
                if time.time() > deadline:
                    break
                j = np.random.randint(0, N)
                h[j] = -h[j]
                new_val = _score(h)
                if new_val < best_score:
                    best_score = new_val
                    improved = True
                else:
                    h[j] = -h[j]
            
            if not improved:
                break
                
    return h, best_score

def _gradient_refine(h, deadline):
    """
    Gradient descent on smooth objective.
    """
    h = h.copy()
    # Normalize to sum=1 to prevent scale drift
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, _score(h)
    h = h / s
    
    best_score = _score(h)
    
    # Multi-stage beta
    for beta in [50, 100, 200]:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _objective_smooth(x, beta=beta)
        
        try:
            res = minimize(obj, h, method='L-BFGS-B', 
                           options={'maxiter': 20, 'disp': False, 'gtol': 1e-5})
            
            if res.x is not None and np.all(np.isfinite(res.x)):
                h_grad = res.x
                s = np.sum(h_grad)
                if abs(s) > 1e-9:
                    h_grad = h_grad / s
                    val = _score(h_grad)
                    if val < best_score:
                        best_score = val
                        h = h_grad
        except Exception:
            pass
            
    return h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse candidates
    candidates = []
    strategies = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # Generate 30 candidates
    for i in range(30):
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
        
    # Sort by score
    candidates.sort(key=lambda x: x[0])
    
    # 2. Refine top candidates
    # We have ~120s. 
    # Gradient refine takes ~1-2s per candidate (20 iters L-BFGS).
    # Local search takes ~2-5s per candidate.
    # Let's refine top 5 with Gradient, then Top 10 with Local Search.
    
    top_k_grad = 5
    top_k_local = 10
    
    # Gradient Refine Top 5
    for i in range(min(top_k_grad, len(candidates))):
        if time.time() > deadline - 30:
            break
        init_val, h_init = candidates[i]
        grad_deadline = min(deadline - 20, time.time() + 5.0)
        h_opt, score_opt = _gradient_refine(h_init, grad_deadline)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
        # Add the refined result to the pool for local search
        # But we need to be careful not to spend too much time.
        
    # Now, take the best candidate so far and the top few from initial list
    # and run Local Search on them.
    
    # Re-evaluate best_h and top initial candidates
    pool = [(best_score, best_h)]
    # Add top 4 initial candidates that weren't in top 5 grad (or just top 5 overall)
    for i in range(min(5, len(candidates))):
        pool.append(candidates[i])
        
    # Sort pool
    pool.sort(key=lambda x: x[0])
    
    # Local Search on top 5 of the pool
    for i in range(min(5, len(pool))):
        if time.time() > deadline - 10:
            break
        init_val, h_init = pool[i]
        local_deadline = min(deadline - 5, time.time() + 15.0)
        h_opt, score_opt = _local_search_block(h_init, local_deadline, max_iters=5)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    # 3. Final Polish
    if time.time() < deadline - 2:
        # One more local search on the very best
        final_deadline = deadline - 1
        h_opt, score_opt = _local_search_block(best_h, final_deadline, max_iters=10)
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
