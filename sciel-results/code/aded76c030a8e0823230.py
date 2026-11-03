import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs(ac, beta=100.0):
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _generate_zadoff_chu(rng, N, root=1):
    """Generate Zadoff-Chu-like sequence."""
    n = np.arange(N)
    # Zadoff-Chu: exp(-j*pi*root*k*(k+1)/N) for even N
    # Use a phase that gives good autocorrelation properties
    phase = -np.pi * root * n * (n + 1) / N
    h = np.cos(phase)
    # Add small random perturbation to break symmetry
    h += rng.standard_normal(N) * 0.1
    return h

def _generate_frank_zadoff(rng, N, m=4):
    """Frank-Zadoff-Chu code construction."""
    n = np.arange(N)
    # Frank code: product of two chirps
    phase1 = np.pi * n**2 / N
    phase2 = np.pi * (n % m) * n / N
    phase = phase1 + phase2
    h = np.cos(phase)
    h += rng.standard_normal(N) * 0.05
    return h

def _generate_candidate(rng, strategy=0):
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
    elif strategy == 8:
        # Zadoff-Chu with random root
        root = rng.integers(1, N)
        h = _generate_zadoff_chu(rng, N, root)
    elif strategy == 9:
        # Frank-Zadoff-Chu
        m = rng.choice([2, 4, 8])
        h = _generate_frank_zadoff(rng, N, m)
    elif strategy == 10:
        # Multi-chirp superposition
        num_chirps = rng.integers(2, 6)
        h = np.zeros(N)
        for _ in range(num_chirps):
            freq = rng.random() * 3.0 + 0.5
            phase = 2 * np.pi * freq * (t**2) / N
            h += np.cos(phase)
    else:
        h = rng.standard_normal(N)
        
    return h

def _local_search_block(h, deadline, max_iters=10):
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1.0
    
    best_score = _score(h)
    block_sizes = [1, 3, 5, 7]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        sizes_to_try = block_sizes.copy()
        np.random.shuffle(sizes_to_try)
        
        for bs in sizes_to_try:
            if time.time() > deadline:
                break
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
                    h[start:start+bs] = -h[start:start+bs]
                    
        if not improved:
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
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, _score(h)
    h = h / s
    
    best_score = _score(h)
    
    for beta in [50, 100, 200]:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _objective_smooth(x, beta=beta)
        
        try:
            res = minimize(obj, h, method='L-BFGS-B', 
                           options={'maxiter': 30, 'disp': False, 'gtol': 1e-6})
            
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

def _phase_optimize(h, deadline):
    """
    Optimize phases in frequency domain to minimize PAPR-like objective.
    Represent h as real part of complex sequence, optimize phases.
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, _score(h)
    h = h / s
    
    best_score = _score(h)
    
    # Get FFT
    fsize = _get_fft_size(N)
    H = np.fft.rfft(h, fsize)
    
    # Work with phases only, keep magnitudes fixed
    mags = np.abs(H)
    mags[mags < 1e-10] = 1e-10  # avoid log(0)
    
    for beta in [30, 50, 80]:
        if time.time() > deadline:
            break
        
        phases = np.angle(H)
        
        def phase_obj(phases):
            H_new = mags * np.exp(1j * phases)
            h_new = np.fft.irfft(H_new, fsize)[:N]
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-9:
                return 1e10
            ac = _autoconv_fft(h_new)
            sm = _soft_max_abs(ac, beta)
            return sm / (s_new * s_new)
        
        try:
            res = minimize(phase_obj, phases, method='L-BFGS-B',
                           options={'maxiter': 20, 'disp': False, 'gtol': 1e-5})
            if res.x is not None and np.all(np.isfinite(res.x)):
                H_new = mags * np.exp(1j * res.x)
                h_new = np.fft.irfft(H_new, fsize)[:N]
                s_new = np.sum(h_new)
                if abs(s_new) > 1e-9:
                    h_new = h_new / s_new
                    val = _score(h_new)
                    if val < best_score:
                        best_score = val
                        h = h_new
        except Exception:
            pass
            
    return h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_score = np.inf
    best_h = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse candidates including Zadoff-Chu and Frank-Zadoff
    candidates = []
    strategies = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    num_candidates = 40
    for i in range(num_candidates):
        if time.time() > deadline - 25:
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
    
    # 2. Refine top candidates with multiple methods
    top_k = 8
    
    # Phase optimization on top candidates
    for i in range(min(top_k, len(candidates))):
        if time.time() > deadline - 30:
            break
        init_val, h_init = candidates[i]
        phase_deadline = min(deadline - 20, time.time() + 8.0)
        h_opt, score_opt = _phase_optimize(h_init, phase_deadline)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
        
        # Add to pool
        candidates.append((score_opt, h_opt.copy()))
    
    # Gradient refine on top candidates
    for i in range(min(top_k, len(candidates))):
        if time.time() > deadline - 20:
            break
        init_val, h_init = candidates[i]
        grad_deadline = min(deadline - 10, time.time() + 6.0)
        h_opt, score_opt = _gradient_refine(h_init, grad_deadline)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
    
    # 3. Local search on best candidates
    pool = [(best_score, best_h)]
    for i in range(min(5, len(candidates))):
        pool.append(candidates[i])
    pool.sort(key=lambda x: x[0])
    
    for i in range(min(5, len(pool))):
        if time.time() > deadline - 8:
            break
        init_val, h_init = pool[i]
        local_deadline = min(deadline - 3, time.time() + 12.0)
        h_opt, score_opt = _local_search_block(h_init, local_deadline, max_iters=8)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    # 4. Final polish
    if time.time() < deadline - 2:
        final_deadline = deadline - 1
        h_opt, score_opt = _local_search_block(best_h, final_deadline, max_iters=12)
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
