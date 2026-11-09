import numpy as np
import time

N = 400

def max_sidelobe(h):
    """
    Compute the maximum normalized autocorrelation sidelobe.
    h: np.array of shape (N,) with values +/- 1.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.fft(f)
    R = np.real(np.fft.ifft(F * np.conj(F)))
    R = R[:N]
    R[0] = 0
    # Normalized by N (peak power)
    return np.max(np.abs(R)) / N

def greedy_local_search(h, deadline, rng):
    """
    Optimize h using Greedy Local Search.
    Repeatedly flips the bit that provides the largest improvement, or any improving bit,
    until no single-bit flip improves the score or time runs out.
    """
    current_score = max_sidelobe(h)
    best_h = h.copy()
    best_score = current_score
    h_work = h.copy()
    
    # To speed up, we can pre-allocate a buffer for the FFT input if we were doing it manually,
    # but np.fft.fft handles this. The main cost is the FFT itself.
    
    # We will loop until no improvement is found or time runs out.
    # To avoid infinite loops in case of numerical noise (unlikely with +/- 1), 
    # we track the number of no-improvement steps.
    
    no_improvement_count = 0
    max_no_improvement_steps = 1000 # Safety valve
    
    while time.time() < deadline and no_improvement_count < max_no_improvement_steps:
        improved = False
        # We can try all indices or random indices. 
        # Trying all indices is O(N) FFTs per iteration. N=400. 
        # If we do 100 iterations, that's 40,000 FFTs. 
        # 1024-point FFT is fast. Let's estimate: ~1ms per FFT. 40k * 1ms = 40s. 
        # This might be too slow for a single run if we have multiple restarts.
        
        # Strategy: Randomly sample a subset of indices to flip, or just pick one random index?
        # Standard greedy: find the BEST flip. This requires N evaluations.
        # Faster greedy: find ANY improving flip. This requires on average much less than N evaluations.
        
        # Let's try to find ANY improving flip.
        # We will iterate through indices in a random order.
        indices = rng.permutation(N)
        
        for i in indices:
            if time.time() >= deadline:
                break
            
            # Flip
            h_work[i] *= -1
            new_score = max_sidelobe(h_work)
            
            if new_score < current_score:
                # Accept
                current_score = new_score
                improved = True
                # Update best if this is the new global best
                if new_score < best_score:
                    best_score = new_score
                    best_h = h_work.copy()
                
                # Since we accepted, the landscape changed. 
                # We could restart the index loop or continue.
                # Restarting is safer to ensure we don't miss a better move due to order.
                # But for speed, let's just break and restart the outer loop with new random order.
                break
            else:
                # Reject, revert
                h_work[i] *= -1
        
        if not improved:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            
    return best_h, best_score

def generate_thue_morse(length):
    """
    Generate Thue-Morse sequence mapped to +/- 1.
    """
    seq = np.array([0])
    while len(seq) < length:
        inv = 1 - seq
        seq = np.concatenate([seq, inv])
    # Map 0 -> 1, 1 -> -1
    return (1 - 2 * seq[:length]).astype(np.float64)

def generate_mseq_length(m, taps, length):
    """
    Generate M-Sequence.
    """
    state = 1
    seq = []
    for i in range(length):
        seq.append(1.0 if (state & 1) else -1.0)
        fb = 0
        for t in taps:
            fb ^= (state >> t) & 1
        state = (state >> 1) | (fb << (m - 1))
    return np.array(seq)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = float('inf')
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random
    candidates.append(rng.choice([-1.0, 1.0], N))
    
    # 2. Thue-Morse
    candidates.append(generate_thue_morse(N))
    
    # 3. M-Sequence (x^9+x^4+1)
    # Note: M-seq length 511 is prime, so it's a good candidate for low autocorr.
    # We truncate to 400.
    h_mseq = generate_mseq_length(9, [0, 4], 511)
    candidates.append(h_mseq[:N])
    
    # 4. Another M-Sequence variant or random
    candidates.append(rng.choice([-1.0, 1.0], N))
    
    # Run Greedy Local Search on each candidate
    # Allocate time evenly
    num_candidates = len(candidates)
    time_per_candidate = (deadline - time.time()) / num_candidates
    
    for idx, h in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Leave a small buffer for the last candidate to ensure we don't exceed deadline
        local_deadline = time.time() + time_per_candidate
        # But if it's the last one, use the remaining time
        if idx == num_candidates - 1:
            local_deadline = deadline - 1.0 # 1s buffer for return
            
        h_work = h.copy()
        h_opt, score = greedy_local_search(h_work, local_deadline, rng)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
    
    # If we have time left, do more random restarts
    if best_h is not None and time.time() < deadline - 5.0:
        while time.time() < deadline - 2.0:
            h_rand = rng.choice([-1.0, 1.0], N)
            h_opt, score = greedy_local_search(h_rand, deadline - 2.0, rng)
            if score < best_score:
                best_score = score
                best_h = h_opt
    
    if best_h is None:
        return np.ones(N)
        
    return best_h
