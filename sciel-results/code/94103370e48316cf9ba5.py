import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Sieve primes up to 10^6
    limit = 10**6
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]

    def compute_phi_segment(start, end):
        """
        Compute phi(n) for n in [start, end] using segmented approach with NumPy.
        Returns a numpy array of phi values.
        """
        size = end - start + 1
        if size <= 0:
            return np.array([], dtype=np.int64)
        
        # Initialize phi and remaining factor arrays
        phi = np.arange(start, end + 1, dtype=np.int64)
        rem = phi.copy()
        
        # Process each prime factor
        for p in primes:
            if p * p > end:
                break
            
            # Find the first multiple of p in [start, end]
            if start % p == 0:
                first = start
            else:
                first = start + (p - start % p)
            
            if first > end:
                continue
                
            # Indices in the segment that are multiples of p
            # relative index = value - start
            idxs = np.arange(first - start, size, p)
            
            if len(idxs) == 0:
                continue
                
            # Check which of these have p as a factor in their remaining part
            # This is always true for the first pass, but for subsequent passes
            # (if we didn't remove all factors) it might not be. 
            # However, our logic removes all factors of p from rem.
            # So rem[idx] % p == 0 is the condition.
            
            mask = rem[idxs] % p == 0
            if not np.any(mask):
                continue
                
            sel = idxs[mask]
            
            # Update phi: phi(n) = phi(n) * (1 - 1/p) = phi(n) - phi(n)//p
            phi[sel] -= phi[sel] // p
            
            # Remove all factors of p from rem
            rem[sel] //= p
            while True:
                still_divisible = rem[sel] % p == 0
                if not np.any(still_divisible):
                    break
                sel = sel[still_divisible]
                rem[sel] //= p
        
        # Handle any remaining prime factor > 1
        # If rem[i] > 1, it is a prime factor
        mask = rem > 1
        if np.any(mask):
            phi[mask] -= phi[mask] // rem[mask]
            
        return phi

    def find_longest_distinct_run(phi_vals):
        """
        Find the longest subarray of phi_vals with all distinct elements.
        Returns (max_length, start_index).
        """
        n = len(phi_vals)
        if n == 0:
            return 0, 0
            
        last_seen = {}
        left = 0
        max_len = 0
        max_start = 0
        
        for right in range(n):
            val = phi_vals[right]
            
            if val in last_seen and last_seen[val] >= left:
                left = last_seen[val] + 1
                
            last_seen[val] = right
            cur_len = right - left + 1
            
            if cur_len > max_len:
                max_len = cur_len
                max_start = left
                
            # Optimization: if we already have a run of length 200, we can't do better
            if max_len >= 200:
                break
                
        return max_len, max_start

    best_k = 1
    best_s = 1

    # Phase 1: Small range to get a quick baseline
    # This also validates the logic
    if time.time() < deadline - 10:
        try:
            phi_vals = compute_phi_segment(1, 10000)
            k, s_offset = find_longest_distinct_run(phi_vals)
            if k > best_k:
                best_k = k
                best_s = 1 + s_offset
        except Exception:
            pass

    # Phase 2: Scan various ranges
    # We want to cover the space [1, 10^12] efficiently.
    # Large numbers might have different collision properties.
    # Let's try segments of size 50,000 to 100,000.
    
    seg_size = 50000
    
    # Candidate starting points
    # We'll scan a few specific regions:
    # 1. Low range (already done partially)
    # 2. Mid range
    # 3. High range near 10^12
    
    candidates = []
    
    # Uniform sampling across the range
    # 10^12 / 50000 = 20,000,000 segments. We can't do all.
    # Let's pick 100 random-ish points and some structured ones.
    
    # Structured points: powers of 10 and multiples
    for exp in range(6, 13):
        base = 10**exp
        if base <= 10**12:
            candidates.append(base)
            candidates.append(base + 10**5)
            candidates.append(base + 10**6)
            
    # Random sampling
    import random
    random.seed(42)
    for _ in range(50):
        s = random.randint(10**6, 10**12 - seg_size)
        candidates.append(s)
        
    # High end specific
    for i in range(10):
        s = 10**12 - seg_size - i * 10**6
        if s > 10**6:
            candidates.append(s)

    # Remove duplicates and sort (optional, but good for deterministic behavior)
    candidates = list(set(candidates))
    candidates.sort()
    
    # Filter valid candidates
    valid_candidates = [c for c in candidates if 1 <= c <= 10**12 - seg_size]
    
    for start in valid_candidates:
        if time.time() > deadline - 5:
            break
            
        end = start + seg_size - 1
        if end > 10**12:
            end = 10**12
            seg_size_actual = end - start + 1
        else:
            seg_size_actual = seg_size
            
        try:
            phi_vals = compute_phi_segment(start, end)
            k, s_offset = find_longest_distinct_run(phi_vals)
            
            if k > best_k:
                best_k = k
                best_s = start + s_offset
                # If we found a very long run, we might want to stop or refine
                if best_k >= 200:
                    # Can't get better than 200
                    break
        except Exception:
            continue

    # Phase 3: Refine around the best found start
    if best_k < 200 and time.time() < deadline - 10:
        # Try to extend the run found
        # The run is [best_s, best_s + best_k - 1]
        # Let's look at a wider window around it
        window_size = 1000
        scan_start = max(1, best_s - window_size // 2)
        scan_end = min(10**12, best_s + best_k + window_size // 2)
        
        try:
            phi_vals = compute_phi_segment(scan_start, scan_end)
            k, s_offset = find_longest_distinct_run(phi_vals)
            if k > best_k:
                best_k = k
                best_s = scan_start + s_offset
        except Exception:
            pass

    # Final validation
    if best_s < 1:
        best_s = 1
    if best_k < 1:
        best_k = 1
    if best_s + best_k - 1 > 10**12:
        best_k = 10**12 - best_s + 1
    if best_k > 200:
        best_k = 200

    return {'start': int(best_s), 'length': int(best_k)}
