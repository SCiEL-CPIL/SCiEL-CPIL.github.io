import numpy as np
import time

N = 400
_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

# Precompute FFT size for autoconvolution
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=_FFT_SIZE)
    C = np.fft.irfft(F * F, n=_FFT_SIZE)
    return C[:2 * N - 1]

def _score(f):
    """Compute max(|autoconv|) / (sum(f))^2. Lower is better."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return float('inf')
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def construct():
    """
    Minimizes max(|autoconv(f)|) / (sum(f))^2 using multi-start local search
    with FFT-accelerated scoring and structured initializations.
    """
    deadline = _start_time + _TIME_LIMIT
    
    best_f = np.ones(N, dtype=np.int8)
    best_score = _score(best_f)
    
    # Helper: try a candidate and update best
    def try_candidate(f):
        nonlocal best_f, best_score
        sc = _score(f)
        if sc < best_score:
            best_score = sc
            best_f = f.copy()
    
    # Phase 1: Structured initializations
    # 1.1 Thue-Morse
    f_tm = np.zeros(N, dtype=np.int8)
    f_tm[0] = 1
    for i in range(1, N):
        n = i // 2
        r = i % 2
        f_tm[i] = f_tm[n] if r == 0 else -f_tm[n]
    try_candidate(f_tm)
    
    # 1.2 Alternating
    f_alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
    try_candidate(f_alt)
    
    # 1.3 Periodic patterns
    for period in [3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200]:
        if time.time() > deadline:
            break
        # Try various phase shifts and duty cycles
        for num_neg in range(0, period, max(1, period // 8)):
            if time.time() > deadline:
                break
            pattern = np.ones(period, dtype=np.int8)
            if num_neg > 0:
                neg_pos = np.linspace(0, period - 1, num_neg, dtype=int)
                neg_pos = np.array(sorted(set(neg_pos.tolist())))
                if len(neg_pos) > 0:
                    pattern[neg_pos] = -1
            f_per = np.tile(pattern, N // period + 1)[:N]
            try_candidate(f_per)
    
    # 1.4 Random ±1 sequences
    rng = np.random.default_rng(42)
    for i in range(2000):
        if time.time() > deadline:
            break
        f_rand = rng.choice([-1, 1], size=N).astype(np.int8)
        try_candidate(f_rand)
    
    # Phase 2: Local search from best candidate
    # Use incremental updates for speed
    def local_search(f, max_passes=10, time_budget=10.0):
        f = f.copy().astype(np.float64)
        S = np.sum(f)
        if abs(S) < 1e-6:
            return f, float('inf')
        
        C = _autoconv_fft(f)
        score = np.max(np.abs(C)) / (S * S)
        
        start = time.time()
        
        for _pass in range(max_passes):
            if time.time() - start > time_budget:
                break
            if time.time() > deadline:
                break
            
            improved = False
            # Try flipping each bit
            for i in range(N):
                if time.time() > deadline:
                    break
                
                v = f[i]
                # Flip f[i]
                f[i] = -v
                new_S = S - 2 * v
                
                # Compute new autoconvolution incrementally
                # C_new[k] = C[k] - 4 * v * f[k-i] for k != 2i, C_new[2i] = C[2i]
                # Actually: C[k] = sum_j f[j]f[k-j]
                # After flipping f[i] -> -f[i]:
                # delta_C[k] = -4 * f[i] * f[k-i] (for k != 2i)
                # delta_C[2i] = 0 (the term f[i]*f[i] becomes f[i]*f[i] still, but sign changes twice)
                # Wait: C[k] contains terms f[i]*f[k-i] and f[k-i]*f[i] if i != k-i
                # If we flip f[i] to -f[i]:
                # New term: (-f[i])*f[k-i] = -f[i]*f[k-i]
                # Old term: f[i]*f[k-i]
                # Change: -2*f[i]*f[k-i]
                # But there are two such terms (j=i and j=k-i) if i != k-i
                # So total change: -4*f[i]*f[k-i]
                # If i == k-i (i.e., k=2i), the term is f[i]*f[i], which becomes (-f[i])*(-f[i]) = f[i]*f[i], no change.
                
                C_new = C.copy()
                k_start = max(0, i)
                k_end = min(2 * N - 1, i + N)
                
                idx = np.arange(k_start, k_end)
                f_idx = idx - i
                
                # Mask out center where k = 2i
                center_mask = (idx == 2 * i)
                delta = -4 * v * f[f_idx]
                delta[center_mask] = 0
                
                C_new[idx] += delta
                
                if new_S == 0:
                    new_score = float('inf')
                else:
                    new_score = np.max(np.abs(C_new)) / (new_S * new_S)
                
                if new_score < score - 1e-10:
                    # Accept
                    S = new_S
                    C = C_new
                    score = new_score
                    improved = True
                else:
                    # Revert
                    f[i] = v
        
            if not improved:
                break
        
        return f, score
    
    # Run local search from the best candidate found so far
    if time.time() < deadline - 5.0:
        f_opt, sc_opt = local_search(best_f.astype(np.float64), max_passes=20, time_budget=30.0)
        if sc_opt < best_score:
            best_score = sc_opt
            best_f = f_opt.astype(np.int8)
    
    # Phase 3: Multi-start local search from random restarts
    if time.time() < deadline - 10.0:
        for restart in range(50):
            if time.time() > deadline - 10.0:
                break
            
            # Random start
            f_rand = rng.choice([-1, 1], size=N).astype(np.float64)
            f_opt, sc_opt = local_search(f_rand, max_passes=5, time_budget=2.0)
            
            if sc_opt < best_score:
                best_score = sc_opt
                best_f = f_opt.astype(np.int8)
    
    # Phase 4: Final local search from best
    if time.time() < deadline - 5.0:
        f_opt, sc_opt = local_search(best_f.astype(np.float64), max_passes=30, time_budget=35.0)
        if sc_opt < best_score:
            best_score = sc_opt
            best_f = f_opt.astype(np.int8)
    
    return best_f.astype(np.int8)

# EVOLVE-BLOCK-END
