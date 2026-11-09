import numpy as np
import time

N = 400

def compute_score(h):
    """
    Compute the maximum normalized autocorrelation sidelobe.
    h: np.array of shape (N,) with values +/- 1.
    """
    # np.correlate(h, h, 'full') gives length 2N-1
    # We want lags 0 to N-1.
    # R[k] = sum h[j] h[j+k]
    # np.correlate output index N-1+k corresponds to lag k.
    R = np.correlate(h, h, mode='full')
    R_sidelobes = R[N-1:N-1+N]
    R_sidelobes[0] = 0 # Ignore peak at lag 0
    return np.max(np.abs(R_sidelobes)) / N

def local_search(h, current_score, max_iters=100):
    """
    Perform greedy local search. Flip bit i if it improves score.
    Uses np.correlate for fast evaluation.
    """
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(N):
            # Try flipping i
            h[i] *= -1
            new_score = compute_score(h)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                h[i] *= -1 # Revert
    return h, current_score

def iterated_local_search(h_init, deadline, rng):
    """
    Run ILS: Local search, then perturb, then local search again.
    """
    h = h_init.copy()
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Initial local search
    h, score = local_search(h, best_score)
    if score < best_score:
        best_score = score
        best_h = h.copy()
    
    while time.time() < deadline:
        # Perturbation: Flip K random bits
        # K can vary, e.g., 5 to 20
        K = rng.integers(5, 20)
        indices = rng.choice(N, size=K, replace=False)
        h[indices] *= -1
        
        # Local search from perturbed state
        h, score = local_search(h, compute_score(h))
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Random Start
    h_rand = rng.choice([-1.0, 1.0], N)
    candidates.append(h_rand)
    
    # 2. Thue-Morse Sequence
    def generate_thue_morse(length):
        seq = np.array([0])
        while len(seq) < length:
            inv = 1 - seq
            seq = np.concatenate([seq, inv])
        return (1 - 2 * seq[:length]).astype(np.float64)

    h_tm = generate_thue_morse(N)
    candidates.append(h_tm)

    # 3. M-Sequence Start (Length 511, truncated to 400)
    def generate_mseq_length(m, taps, length):
        state = 1
        seq = []
        for i in range(length):
            seq.append(1.0 if (state & 1) else -1.0)
            fb = 0
            for t in taps:
                fb ^= (state >> t) & 1
            state = (state >> 1) | (fb << (m - 1))
        return np.array(seq)
        
    # Taps for x^9+x^4+1: bits 0 and 4
    h_mseq = generate_mseq_length(9, [0, 4], 511)
    h_mseq = h_mseq[:N]
    candidates.append(h_mseq)
    
    # 4. All Ones (Baseline)
    h_ones = np.ones(N)
    candidates.append(h_ones)
    
    best_h = None
    best_score = float('inf')
    
    # Allocate time to each candidate
    # We have ~115s. 4 candidates.
    # ILS is fast, so we can give more time to the first few or just split evenly.
    # Splitting evenly is safe.
    time_per_candidate = (deadline - time.time()) / len(candidates)
    
    for idx, h in enumerate(candidates):
        if time.time() > deadline:
            break
            
        h_work = h.copy()
        local_deadline = time.time() + time_per_candidate
        
        h_opt, score = iterated_local_search(h_work, local_deadline, rng)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
