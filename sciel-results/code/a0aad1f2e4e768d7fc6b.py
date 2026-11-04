"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_peak(h):
    """Compute max absolute value of linear autoconvolution of step function h."""
    n = len(h)
    # Linear autoconvolution via FFT
    size = 2 * n
    f = np.zeros(size)
    f[:n] = h
    F = np.fft.fft(f)
    conv = np.fft.ifft(F * F).real[:n]
    return np.max(np.abs(conv))

def _thue_morse(n):
    """Generate Thue-Morse sequence mapped to +1/-1."""
    # Efficient vectorized generation
    # t[i] = parity of popcount(i)
    # 0 -> +1, 1 -> -1
    idx = np.arange(n)
    # Use bit manipulation to count set bits
    # Since n=400, we can just do a loop or use a lookup table
    # Vectorized approach:
    # For i in 0..n-1, popcount(i) % 2
    # This is the Thue-Morse sequence
    # We can compute it recursively or via bit tricks
    # Simple loop is fast enough for 400
    t = np.empty(n, dtype=np.float64)
    for i in range(n):
        # Count set bits
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        t[i] = 1.0 if c % 2 == 0 else -1.0
    return t

def _random_sequence(n, seed=None):
    """Generate random +/-1 sequence."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.choice([-1.0, 1.0], size=n)

def _local_search(h, time_limit=10.0):
    """
    Fast local search using incremental updates where possible,
    but here we use full FFT recompute for simplicity and correctness,
    with a time limit.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _autoconv_peak(best_h)
    start_time = time.time()
    
    # We'll do a simple hill-climbing with random restarts within the time limit
    # To be faster, we can try flipping multiple bits at once or use a more clever update,
    # but for now, let's do single-bit flips with a time check.
    
    # Since N=400, FFT is ~O(N log N) which is very fast.
    # We can do many iterations.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try flipping each bit
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            trial = best_h.copy()
            trial[i] *= -1
            new_score = _autoconv_peak(trial)
            if new_score < best_score - 1e-12:
                best_h = trial
                best_score = new_score
                improved = True
                # Don't break immediately, keep looking for more improvements in this pass
                # But if we want to be faster, we could break.
                # Let's continue the pass to find multiple improvements.
        
        if not improved:
            # If no improvement in a full pass, try a random restart or accept a worse move?
            # For simplicity, let's do a random perturbation if stuck.
            # Pick a few random bits to flip
            k = np.random.randint(1, 5)
            idxs = np.random.choice(n, size=k, replace=False)
            trial = best_h.copy()
            trial[idxs] *= -1
            new_score = _autoconv_peak(trial)
            if new_score < best_score:
                best_h = trial
                best_score = new_score
            # Else, we are stuck. Let's break or continue with random restarts.
            # To avoid infinite loops, let's break if we've been stuck for a while.
            # For now, just break and return.
            break
            
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_h = None
    best_score = np.inf
    
    # Strategy:
    # 1. Try Thue-Morse (both polarities)
    # 2. Try several random initializations
    # 3. Run local search on each
    # 4. Return the best found
    
    # 1. Thue-Morse
    tm = _thue_morse(N)
    for signs in [tm, -tm]:
        if time.time() > deadline - 5.0:
            break
        h, score = _local_search(signs, time_limit=10.0)
        if score < best_score:
            best_score = score
            best_h = h
    
    # 2. Random initializations
    # We have about 120s left. Each local search takes ~10s.
    # We can do ~10 random restarts.
    for i in range(10):
        if time.time() > deadline - 5.0:
            break
        h_init = _random_sequence(N)
        h, score = _local_search(h_init, time_limit=10.0)
        if score < best_score:
            best_score = score
            best_h = h
            
    # If we still have time, do more random restarts with shorter time limits
    while time.time() < deadline - 5.0:
        h_init = _random_sequence(N)
        # Use shorter time limit for more restarts
        remaining = deadline - 5.0 - time.time()
        tl = min(5.0, remaining / 2.0)
        if tl < 0.1:
            break
        h, score = _local_search(h_init, time_limit=tl)
        if score < best_score:
            best_score = score
            best_h = h

    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
