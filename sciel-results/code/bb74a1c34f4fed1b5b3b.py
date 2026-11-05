import numpy as np
import time

N = 400

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return np.max(np.abs(acf[1:N]))

def _local_search(h, time_limit):
    """
    Fast local search using greedy single-flip moves.
    Uses FFT-based score evaluation.
    """
    h = h.copy()
    n = len(h)
    current_score = _score(h)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        best_new_score = current_score
        best_pos = -1
        
        # Try all single flips, pick the best one
        for j in range(n):
            h[j] = -h[j]
            new_score = _score(h)
            h[j] = -h[j]  # revert
            
            if new_score < best_new_score - 1e-9:
                best_new_score = new_score
                best_pos = j
        
        if best_pos != -1:
            h[best_pos] = -h[best_pos]
            current_score = best_new_score
            improved = True
            
    return h, current_score

def _swap_local_search(h, time_limit):
    """
    Local search using pairwise swaps (swap signs at two positions).
    This expands the neighborhood beyond single flips.
    """
    h = h.copy()
    n = len(h)
    current_score = _score(h)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        best_new_score = current_score
        best_i = -1
        best_j = -1
        
        # Try a subset of swaps to keep it fast
        # Random subset of positions to swap
        rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
        positions = rng.choice(n, size=min(50, n), replace=False)
        
        for idx_i in range(len(positions)):
            i = positions[idx_i]
            for idx_j in range(idx_i + 1, len(positions)):
                j = positions[idx_j]
                
                # Swap signs at i and j
                h[i], h[j] = -h[i], -h[j]
                new_score = _score(h)
                h[i], h[j] = -h[i], -h[j]  # revert
                
                if new_score < best_new_score - 1e-9:
                    best_new_score = new_score
                    best_i = i
                    best_j = j
        
        if best_i != -1:
            h[best_i], h[best_j] = -h[best_i], -h[best_j]
            current_score = best_new_score
            improved = True
            
    return h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    # Strategy:
    # 1. Generate diverse initial candidates:
    #    a) Chaotic logistic map sequences (deterministic aperiodicity)
    #    b) Random sequences
    #    c) Quasi-periodic structures
    #    d) Thue-Morse and other combinatorial sequences
    # 2. Use fast local search (greedy flipping + swap search) to refine candidates.
    # 3. Use random restarts if time permits.
    
    candidates = []
    
    # 1. Chaotic logistic map sequences
    # The logistic map x_{n+1} = r * x_n * (1 - x_n) generates deterministic aperiodic sequences
    # for r near the edge of chaos (3.57 - 4.0). These have broadband spectral properties.
    rng = np.random.default_rng(42)
    
    # Known good chaotic parameters
    r_candidates = [3.57, 3.6, 3.7, 3.8, 3.9, 3.95, 3.99, 4.0]
    x0_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for r in r_candidates:
        for x0 in x0_candidates:
            x = np.empty(N)
            x[0] = x0
            for i in range(1, N):
                x[i] = r * x[i-1] * (1 - x[i-1])
            signs = np.sign(x)
            signs[signs == 0] = 1.0
            candidates.append(signs)
    
    # 2. Random sequences
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 3. Quasi-periodic structures
    for trial in range(5):
        phases = rng.uniform(0, 2 * np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2 * np.pi * k * t + phases[k-1])
        candidates.append(np.sign(val))
    
    # 4. Thue-Morse sequence (known good for low autocorrelation)
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    tm = np.array([bin(i).count('1') % 2 for i in range(N)])
    tm_signs = np.where(tm == 0, 1.0, -1.0)
    candidates.append(tm_signs)
    
    # 5. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2 * np.pi * np.arange(N) / period))
        candidates.append(pattern)
    
    # 6. Block structure
    block_size = 10
    for i in range(4):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b * block_size:(b + 1) * block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
    
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top candidates with local search
    # Use single-flip local search first, then swap local search
    for i, (s, h) in enumerate(scored_candidates[:8]):
        if time.time() > deadline - 30:
            break
        
        # Single-flip local search
        time_limit = deadline - 20
        h_refined, s_refined = _local_search(h, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
        
        # If time permits, do swap local search on the best result
        if time.time() < deadline - 10 and best_h is not None:
            h_swap, s_swap = _swap_local_search(best_h, deadline - 5)
            if s_swap < best_score:
                best_score = s_swap
                best_h = h_swap.copy()
    
    # If time remains, do a few random restarts with local search
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(deadline - 5, time.time() + 5)
        h_refined, s_refined = _local_search(h_init, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
