import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 20s safety margin
    deadline = start_time + 130.0
    
    def generate_powerful(limit):
        """
        Generate all powerful numbers <= limit.
        Powerful numbers are of the form u^2 * v^3 where gcd(u, v) is not strictly required 
        to be 1 for the set definition, but we can generate them uniquely by iterating 
        cubes and squares.
        """
        pw = set()
        max_v = int(limit ** (1/3)) + 2
        
        cubes = []
        for v in range(1, max_v + 1):
            c = v ** 3
            if c > limit:
                break
            cubes.append(c)
            
        for c in cubes:
            max_u = isqrt(limit // c)
            for u in range(1, max_u + 1):
                pw.add(u * u * c)
        
        return sorted(pw)

    # Strategy:
    # 1. Generate powerful numbers up to a "full" limit (e.g., 10^9) for exact verification.
    # 2. Use a "coarse" FFT on a smaller subset (e.g., up to 10^7) to find top candidates N.
    # 3. For each top candidate N, count exact pairs using the full set.
    
    # Full limit for powerful numbers
    # Memory for set of 100k ints is small.
    # Time to generate 10^9 powerful numbers: 
    # v^3 <= 10^9 => v <= 1000.
    # For v=1, u <= 31622.
    # Total count is approx 100k. Generation is fast.
    
    full_limit = 100_000_000  # 10^8 is safe and fast. 10^9 might be slower.
    # Let's try 200_000_000 (2*10^8) to match inspiration 1's range but with a different verification.
    # Actually, let's stick to 10^8 for the full set to ensure speed, and use the FFT to guide us to N up to 2*10^8.
    # Wait, if full_limit is 10^8, max N is 2*10^8.
    # Current program searches N up to 160M (2*80M).
    # If we use full_limit = 200M, max N is 400M.
    
    if time.time() > deadline - 100:
        full_limit = 50_000_000
    else:
        full_limit = 200_000_000
        
    try:
        powerful_full = generate_powerful(full_limit)
    except Exception:
        full_limit = 50_000_000
        powerful_full = generate_powerful(full_limit)
        
    K_full = len(powerful_full)
    powerful_set_full = set(powerful_full)
    
    # Coarse FFT limit
    # We want the FFT to be fast. 10^7 is good.
    coarse_limit = 10_000_000
    
    # Check if we have time for a larger coarse limit
    if time.time() < deadline - 100:
        coarse_limit = 20_000_000
        
    powerful_coarse = [p for p in powerful_full if p <= coarse_limit]
    K_coarse = len(powerful_coarse)
    
    if K_coarse == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Create indicator array for coarse FFT
    size = 2 * coarse_limit + 1
    arr = np.zeros(size, dtype=np.float32)
    arr[powerful_coarse] = 1.0
    
    fft_arr = np.fft.rfft(arr)
    conv = np.fft.irfft(fft_arr * fft_arr, n=size)
    
    # Find top candidates
    top_k = 200
    if len(conv) > top_k:
        top_indices = np.argpartition(conv, -top_k)[-top_k:]
    else:
        top_indices = np.arange(len(conv))
        
    top_indices = top_indices[np.argsort(conv[top_indices])[::-1]]
    
    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]
    
    # Verify top candidates using the FULL set of powerful numbers
    # We iterate over powerful_full up to N/2
    
    for idx in top_indices:
        if time.time() > deadline - 5:
            break
            
        N_cand = int(idx)
        
        # The coarse FFT only considered powerful numbers <= coarse_limit.
        # The true N_cand might have pairs involving powerful numbers > coarse_limit.
        # But N_cand <= 2 * coarse_limit.
        # If a > coarse_limit, then b = N_cand - a < coarse_limit.
        # So we must check all a in powerful_full up to N_cand // 2.
        
        exact_pairs = []
        count = 0
        
        # Optimization: Only check a up to N_cand // 2
        # powerful_full is sorted.
        # Find the index where powerful_full > N_cand // 2
        # We can use bisect, but simple loop with break is fine if K is small.
        # K_full is ~200k. 200 candidates * 200k = 40M checks. Might be slow in Python.
        # Let's use numpy for the check?
        # Or just break early.
        
        half_N = N_cand // 2
        # Find cutoff index in powerful_full
        # Since powerful_full is sorted, we can find the max index where p <= half_N
        # But we also need b = N_cand - a to be in powerful_set_full.
        
        # To speed up, we can precompute a numpy array of powerful_full
        # and use a set for lookup.
        
        for a in powerful_full:
            if a > half_N:
                break
            b = N_cand - a
            if b in powerful_set_full:
                count += 1
                exact_pairs.append([a, b])
                
        if count > best_count:
            best_count = count
            best_N = N_cand
            best_pairs = exact_pairs
            
        # Early termination if conv value is too low
        # conv[idx] is the count from the coarse set.
        # The true count is >= conv[idx] (since coarse set is subset of full set? No, coarse set is subset of full set, so any pair in coarse is in full. But there might be more pairs in full).
        # Actually, conv[idx] counts pairs (a,b) with a,b <= coarse_limit.
        # The true count includes pairs where one or both are > coarse_limit.
        # So true count >= conv[idx].
        # If conv[idx] < best_count, it's still possible that true count > best_count.
        # So we can't break early based on conv[idx] < best_count.
        # We must check all top_k candidates.
        
    return {'N': best_N, 'pairs': best_pairs}
