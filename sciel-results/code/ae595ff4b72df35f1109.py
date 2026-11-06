"""Optimized additive basis construction using FFT-based evaluation and local search."""
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill
    
    n = 128
    # Target sums: [128, 256], inclusive
    # A subset of [0, 128]
    
    # We'll work with a bitset/indicator vector of length 129 (indices 0..128)
    # Sum of two elements a+b where a,b in A, a<=b (unordered, allow equal)
    # For sum s in [128, 256], representation count = number of pairs (a,b) with a<=b, a+b=s, a,b in A
    
    # Using FFT: the autocorrelation of indicator vector v (length 129) gives
    # c[s] = sum_{a} v[a] * v[s-a] for all s.
    # But this counts ordered pairs. For unordered with equal allowed:
    #   if s is even: count_unordered = (c[s] + v[s//2]) // 2
    #   if s is odd:  count_unordered = c[s] // 2
    
    # Actually let's be more careful. c[s] = sum_{a=0}^{128} v[a]*v[s-a] counts ordered pairs (a,b) with a+b=s.
    # Each unordered pair {a,b} with a!=b contributes 2 to c[s].
    # Each pair {a,a} with 2a=s contributes 1 to c[s].
    # So: unordered_count = (c[s] + (1 if s even and v[s//2] else 0)) // 2
    
    # Let's verify: if a!=b, ordered has (a,b) and (b,a), so 2 ordered -> 1 unordered.
    # If a==b, ordered has (a,a) once, so 1 ordered -> 1 unordered.
    # c[s] = 2*(# pairs a<b with a+b=s) + (# pairs a=a with 2a=s)
    # unordered = # pairs a<b + # pairs a=a = (c[s] - I(s even, v[s//2]))/2 + I(s even, v[s//2])
    #          = (c[s] + I(s even, v[s//2]))/2
    
    # We want to minimize the max representation count over s in [128, 256].
    
    # Strategy: start with a good initial set, then do local search (add/remove/swap elements)
    # to minimize the peak multiplicity.
    
    # Initial set: try a few different starting points
    # The seed was range(64,129) which is 65 elements.
    # Let's think about what's optimal. We need to cover [128,256] with pair sums.
    # If A = {a1, ..., ak}, then pair sums range from 2*min(A) to 2*max(A).
    # We need 2*max(A) >= 256, so max(A) >= 128. Since A subset of [0,128], max(A)=128.
    # We need 2*min(A) <= 128, so min(A) <= 64.
    
    # Let's try a random initialization with some structure, then local search.
    
    def compute_peak(v):
        """Compute the peak representation multiplicity for target sums [128,256]."""
        # v is a boolean array of length 129
        # Use FFT for convolution
        # Pad to appropriate size
        size = 257  # need sums up to 256
        # Use next power of 2 for FFT
        fft_size = 1
        while fft_size < 2 * size:
            fft_size *= 2
        
        v_float = v.astype(np.float64)
        # Zero-pad
        v_padded = np.zeros(fft_size)
        v_padded[:len(v_float)] = v_float
        
        # Autocorrelation via FFT
        V = np.fft.fft(v_padded)
        auto = np.fft.ifft(V * np.conj(V)).real
        auto = np.round(auto).astype(np.int64)
        
        peak = 0
        for s in range(128, 257):
            c = auto[s]
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and v[half]:
                    count = (c + 1) // 2
                else:
                    count = c // 2
            else:
                count = c // 2
            if count > peak:
                peak = count
        return peak
    
    def is_valid(v):
        """Check that every sum in [128,256] has at least one representation."""
        size = 257
        fft_size = 1
        while fft_size < 2 * size:
            fft_size *= 2
        
        v_float = v.astype(np.float64)
        v_padded = np.zeros(fft_size)
        v_padded[:len(v_float)] = v_float
        
        V = np.fft.fft(v_padded)
        auto = np.fft.ifft(V * np.conj(V)).real
        auto = np.round(auto).astype(np.int64)
        
        for s in range(128, 257):
            c = auto[s]
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and v[half]:
                    count = (c + 1) // 2
                else:
                    count = c // 2
            else:
                count = c // 2
            if count == 0:
                return False
        return True
    
    # Initial solution: try the seed
    v = np.zeros(129, dtype=bool)
    v[64:129] = True  # range(64, 129)
    
    # Verify seed is valid
    if not is_valid(v):
        # Try a different init
        v = np.zeros(129, dtype=bool)
        v[0:129:2] = True  # even numbers
        if not is_valid(v):
            v = np.zeros(129, dtype=bool)
            v[0:129:1] = True  # all numbers (worst case)
    
    current_peak = compute_peak(v)
    best_v = v.copy()
    best_peak = current_peak
    
    # Local search: try adding/removing/swapping elements
    # Use simulated annealing or simple hill climbing with random restarts
    
    import random
    random.seed(42)
    
    # Precompute for speed: we'll recompute peak after each move
    # To speed up, we can track which sums are affected
    
    # Let's do a more efficient local search
    # For each element x in [0,128], try toggling it and see if peak decreases
    
    # First, let's try to reduce peak by removing elements that contribute to high-peak sums
    # Then add elements to fix any coverage gaps
    
    improved = True
    max_iter = 50000
    iter_count = 0
    
    while improved and iter_count < max_iter and (time.time() - start_time) < deadline:
        improved = False
        # Try removing elements
        for x in range(129):
            if not v[x]:
                continue
            if (time.time() - start_time) > deadline:
                break
            
            v[x] = False
            if is_valid(v):
                new_peak = compute_peak(v)
                if new_peak < current_peak:
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_v = v.copy()
                    improved = True
                else:
                    v[x] = True
            else:
                v[x] = True
        
        if (time.time() - start_time) > deadline:
            break
        
        # Try adding elements
        for x in range(129):
            if v[x]:
                continue
            if (time.time() - start_time) > deadline:
                break
            
            v[x] = True
            new_peak = compute_peak(v)
            if new_peak < current_peak:
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_v = v.copy()
                improved = True
            else:
                v[x] = False
        
        iter_count += 1
    
    # If we still have time, do random perturbations and restart
    if (time.time() - start_time) < deadline - 5:
        # Random restarts
        for restart in range(20):
            if (time.time() - start_time) > deadline - 2:
                break
            
            # Random initialization
            v_rand = np.zeros(129, dtype=bool)
            # Start with a base that covers the range
            v_rand[64:129] = True
            # Randomly toggle some elements
            for _ in range(30):
                x = random.randint(0, 128)
                v_rand[x] = not v_rand[x]
            
            if is_valid(v_rand):
                peak_rand = compute_peak(v_rand)
                if peak_rand < best_peak:
                    best_peak = peak_rand
                    best_v = v_rand.copy()
                    # Continue local search from here
                    v = v_rand.copy()
                    current_peak = peak_rand
                    for _ in range(500):
                        if (time.time() - start_time) > deadline - 1:
                            break
                        improved2 = False
                        for x in range(129):
                            if (time.time() - start_time) > deadline - 1:
                                break
                            if v[x]:
                                v[x] = False
                                if is_valid(v):
                                    np_ = compute_peak(v)
                                    if np_ < current_peak:
                                        current_peak = np_
                                        if current_peak < best_peak:
                                            best_peak = current_peak
                                            best_v = v.copy()
                                        improved2 = True
                                    else:
                                        v[x] = True
                                else:
                                    v[x] = True
                            else:
                                v[x] = True
                                np_ = compute_peak(v)
                                if np_ < current_peak:
                                    current_peak = np_
                                    if current_peak < best_peak:
                                        best_peak = current_peak
                                        best_v = v.copy()
                                    improved2 = True
                                else:
                                    v[x] = False
                        if not improved2:
                            break
    
    # Return the best valid solution
    result = [i for i in range(129) if best_v[i]]
    if not result:
        result = list(range(64, 129))
    return result
