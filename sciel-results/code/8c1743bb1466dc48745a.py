import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = np.convolve(h, h, 'full')
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _objective_spectral(h, beta=50.0):
    """
    Smooth objective using the spectral representation.
    We want to minimize max|ac[k]| / (sum h)^2.
    
    Using the spectrum Ac = |H|^2, the autoconvolution ac = irfft(Ac).
    We use a soft-max approximation of max|ac|.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    
    # Compute spectrum
    fsize = 1
    target = 2 * N - 1
    while fsize < target:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    Ac = H * np.conjugate(H)
    ac = np.fft.irfft(Ac, fsize)
    
    # Only the first 2N-1 values are the linear autoconvolution
    ac_lin = ac[:2*N-1]
    
    # Soft max of |ac_lin|
    abs_ac = np.abs(ac_lin)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_max = m + log_sum / beta
    
    return soft_max / (s * s)

def _generate_candidate(rng, strategy=0):
    """
    Generate a diverse candidate vector.
    """
    h = np.zeros(N)
    
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
        # Cosine wave with random phase
        freq = rng.uniform(1, 5)
        phase = rng.uniform(0, 2*np.pi)
        h = np.cos(2*np.pi*freq*np.arange(N)/N + phase)
    elif strategy == 7:
        # Two-tone signal
        f1 = rng.uniform(1, 3)
        f2 = rng.uniform(3, 7)
        p1 = rng.uniform(0, 2*np.pi)
        p2 = rng.uniform(0, 2*np.pi)
        h = np.cos(2*np.pi*f1*np.arange(N)/N + p1) + np.cos(2*np.pi*f2*np.arange(N)/N + p2)
    else:
        h = rng.standard_normal(N)
        
    return h

def _local_search_refine(h, deadline, max_iters=30):
    """
    Hybrid local search:
    1. Gradient descent on smooth spectral objective.
    2. Sign flipping local search.
    """
    h = h.copy()
    best_score = _score(h)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # 1. Gradient Descent Step on spectral objective
        try:
            beta = 50.0 + i * 10.0
            def obj(x):
                return _objective_spectral(x, beta=beta)
            
            res = minimize(obj, h, method='L-BFGS-B', 
                           options={'maxiter': 10, 'disp': False, 'gtol': 1e-5})
            
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

        # 2. Sign Flipping Local Search (greedy)
        improved = False
        # Try flipping each element once
        for j in range(N):
            if time.time() > deadline:
                break
            orig = h[j]
            h[j] = -orig
            new_val = _score(h)
            if new_val < best_score:
                best_score = new_val
                improved = True
            else:
                h[j] = orig
        
        if not improved:
            # Try a few random flips
            for _ in range(10):
                j = np.random.randint(0, N)
                orig = h[j]
                h[j] = -orig
                new_val = _score(h)
                if new_val < best_score:
                    best_score = new_val
                    improved = True
                else:
                    h[j] = orig
            if not improved:
                break

    return h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin for 150s limit
    
    best_score = np.inf
    best_h = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    strategies = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # 1. Initial Phase: Generate and screen candidates
    num_initial_candidates = 60
    
    candidates = []
    for i in range(num_initial_candidates):
        if time.time() > deadline - 40:
            break
        strat = strategies[i % len(strategies)]
        h = _generate_candidate(rng, strat)
        s = np.sum(h)
        if abs(s) < 1e-9:
            continue
        h = h / s
        val = _score(h)
        candidates.append((val, h))
        
    # Sort by initial score
    candidates.sort(key=lambda x: x[0])
    
    # Refine the top K candidates
    K = 12
    for i, (init_val, h_init) in enumerate(candidates[:K]):
        if time.time() > deadline - 15:
            break
        
        local_deadline = min(deadline - 5.0, time.time() + 6.0)
        
        h_opt, score_opt = _local_search_refine(h_init, local_deadline)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # 2. Final Phase: Intensive polishing
    if time.time() < deadline - 5.0 and best_score < np.inf:
        final_deadline = deadline
        h_polished, score_polished = _local_search_refine(best_h, final_deadline, max_iters=50)
        
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished
    
    # Ensure valid return
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
