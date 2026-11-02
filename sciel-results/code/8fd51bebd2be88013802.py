"""Improved program: block-flip simulated annealing + FFT-based delta evaluation for C3 bound."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3_bound(seq):
    """Compute the autoconvolution upper bound C3 (sum of |sidelobe autocorrelation|)."""
    n = len(seq)
    if n < 2:
        return 0.0
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    f = np.fft.fft(seq, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return float(np.sum(np.abs(acf[1:])))

def _delta_c3_flip(seq, idx, n_fft, f):
    """Compute change in C3 bound after flipping sign at index idx.
    
    Uses the fact that flipping h_i to -h_i changes the autocorrelation at lag k
    by -2*h_i*(h_{i+k} + h_{i-k} - delta_{k,0}*h_i) for the relevant terms.
    For efficiency, we recompute via FFT when block size is small, or use
    a direct approach for single flips.
    """
    n = len(seq)
    # For single flip, the change in autocorrelation:
    # acf[k] = sum_i h_i * h_{i+k}
    # After flipping h_idx to -h_idx:
    # acf_new[k] = acf[k] - 2*h_idx*h_{idx+k} (if idx+k < n) 
    #            + 2*h_idx*h_{idx-k} (if idx-k >= 0)  [for the reverse term when k>0]
    # Actually more carefully:
    # acf[k] = sum_{i=0}^{n-1-k} h_i * h_{i+k}
    # The terms involving h_idx are:
    #   i = idx: h_idx * h_{idx+k} (if idx+k < n)
    #   i+k = idx => i = idx-k: h_{idx-k} * h_idx (if idx-k >= 0)
    # So delta_acf[k] = -2 * h_idx * (h_{idx+k} if idx+k<n else 0) 
    #                    -2 * h_idx * (h_{idx-k} if idx-k>=0 else 0)
    # But wait, when k=0: acf[0] = sum h_i^2, flipping doesn't change it.
    # For k>0: the two terms above.
    
    old_h = seq[idx]
    # We'll compute the new acf sidelobe sum directly for the affected lags
    # This is O(n) per flip which is acceptable for 400 elements
    
    # Actually, let's just do a quick FFT-based check for small changes
    # For a single flip, we can update the FFT in O(n)
    new_seq = seq.copy()
    new_seq[idx] = -new_seq[idx]
    
    # Quick acf computation
    f_new = np.fft.fft(new_seq, n=n_fft)
    acf_new = np.fft.ifft(f_new * np.conj(f_new)).real[:n]
    return float(np.sum(np.abs(acf_new[1:])))

def _anneal_block(seq, deadline, rng, min_block=2, max_block=32, n_iterations=50000):
    """Simulated annealing with block-flip moves."""
    n = len(seq)
    cur_seq = seq.copy()
    
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    
    cur_score = _compute_c3_bound(cur_seq)
    best_seq = cur_seq.copy()
    best_score = cur_score
    
    T = 50.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    it = 0
    while it < n_iterations and time.time() < deadline:
        # Choose block size
        block_size = int(rng.integers(min_block, max_block + 1))
        if block_size > n:
            block_size = n
        # Choose random start position
        start = int(rng.integers(0, n - block_size + 1))
        
        # Flip the block
        old_vals = cur_seq[start:start+block_size].copy()
        cur_seq[start:start+block_size] = -old_vals
        
        new_score = _compute_c3_bound(cur_seq)
        
        delta = new_score - cur_score
        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_seq = cur_seq.copy()
        else:
            # Revert
            cur_seq[start:start+block_size] = old_vals
        
        T = max(T_min, T * cooling_rate)
        it += 1
        
        # Periodic time check
        if it % 100 == 0 and time.time() > deadline:
            break
    
    return best_seq, best_score

def _greedy_polish(seq, deadline):
    """Greedy local search: try all single flips, keep improvements. Repeat until no improvement."""
    n = len(seq)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    
    cur_seq = seq.copy()
    cur_score = _compute_c3_bound(cur_seq)
    
    improved = True
    passes = 0
    while improved and time.time() < deadline and passes < 5:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > deadline:
                break
            cur_seq[i] = -cur_seq[i]
            new_score = _compute_c3_bound(cur_seq)
            if new_score < cur_score:
                cur_score = new_score
                improved = True
            else:
                cur_seq[i] = -cur_seq[i]
    
    return cur_seq, cur_score

def _random_walk_sequence(n, rng, seed):
    """Generate a random ±1 sequence."""
    r = np.random.default_rng(seed)
    return r.choice([-1.0, 1.0], size=n)

def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound C3."""
    t_start = time.time()
    deadline = t_start + 130  # Safety margin before 150s limit
    
    rng = np.random.default_rng(12345)
    
    best_seq = None
    best_score = float('inf')
    
    # Generate candidate initial sequences
    candidates = []
    
    # All ones (baseline)
    candidates.append(np.ones(N, dtype=np.float64))
    
    # Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Random sequences with different seeds
    for s in [1, 7, 42, 99, 2024, 31337]:
        candidates.append(_random_walk_sequence(N, rng, s))
    
    # Legendre-like sequence
    def _legendre(n):
        # Find smallest prime >= n
        p = n
        def is_prime(k):
            if k < 2:
                return False
            for i in range(2, int(k**0.5)+1):
                if k % i == 0:
                    return False
            return True
        while not is_prime(p):
            p += 1
        qr = set((k*k) % p for k in range(1, p))
        seq = np.ones(p)
        for k in range(1, p):
            if k not in qr:
                seq[k] = -1.0
        return seq[:n]
    try:
        candidates.append(_legendre(N))
    except:
        pass
    
    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline - 60:
            break
        sc = _compute_c3_bound(cand)
        if sc < best_score:
            best_score = sc
            best_seq = cand.copy()
    
    if best_seq is None:
        best_seq = np.ones(N, dtype=np.float64)
        best_score = _compute_c3_bound(best_seq)
    
    # Run multiple annealing rounds with different parameters
    remaining = deadline - time.time()
    if remaining > 10:
        # First anneal: larger blocks
        time_left = deadline - time.time()
        if time_left > 20:
            seq, sc = _anneal_block(best_seq, t_start + time_left - 5, rng, 
                                     min_block=4, max_block=32, n_iterations=30000)
            if sc < best_score:
                best_score = sc
                best_seq = seq
        
        # Second anneal: smaller blocks for refinement
        time_left = deadline - time.time()
        if time_left > 10:
            seq, sc = _anneal_block(best_seq, t_start + time_left - 3, rng,
                                     min_block=2, max_block=8, n_iterations=40000)
            if sc < best_score:
                best_score = sc
                best_seq = seq
    
    # Final greedy polish
    time_left = deadline - time.time()
    if time_left > 5:
        seq, sc = _greedy_polish(best_seq, t_start + time_left - 1)
        if sc < best_score:
            best_score = sc
            best_seq = seq
    
    return best_seq
# EVOLVE-BLOCK-END
