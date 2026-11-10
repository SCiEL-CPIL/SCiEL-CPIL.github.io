import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
        
    # Sieve primes up to sqrt(10^12) = 10^6
    limit = 10**6 + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    best_k = 1
    best_s = 1
    
    # Block size for phi computation
    W = 200000
    
    def compute_phi_block(lo, hi):
        """Compute phi(n) for n in [lo, hi] using segmented sieve."""
        n = hi - lo + 1
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        rem = np.arange(lo, hi + 1, dtype=np.int64)
        
        max_p = int(math.isqrt(hi))
        idx_end = np.searchsorted(primes, max_p + 1)
        relevant_primes = primes[:idx_end]
        
        for p in relevant_primes:
            p = int(p)
            start = ((lo + p - 1) // p) * p
            if start > hi:
                continue
            indices = np.arange(start, hi + 1, p) - lo
            phi[indices] -= phi[indices] // p
            r = rem[indices].copy()
            while True:
                mask = r % p == 0
                if not np.any(mask):
                    break
                r[mask] //= p
            rem[indices] = r
        
        mask = rem > 1
        if np.any(mask):
            phi[mask] -= phi[mask] // rem[mask]
        return phi
    
    def find_best_run_in_phi(phi, lo):
        """Find longest run of distinct values in phi array."""
        last_seen = {}
        start_idx = 0
        local_best_k = 0
        local_best_s = lo
        
        for i in range(len(phi)):
            v = phi[i]
            if v in last_seen:
                if last_seen[v] >= start_idx:
                    start_idx = last_seen[v] + 1
            last_seen[v] = i
            k = i - start_idx + 1
            if k > local_best_k:
                local_best_k = k
                local_best_s = lo + start_idx
        
        return local_best_k, local_best_s
    
    # Strategy: Use random sampling to find promising regions
    # Long runs are more likely at larger values where phi values are more spread out
    
    # First, do a quick scan of some strategic regions
    search_regions = []
    
    # Region 1: Start from a moderate value
    search_regions.append((10**6, 10**6 + W))
    
    # Region 2: Larger values where collisions are less frequent
    search_regions.append((10**8, 10**8 + W))
    search_regions.append((10**9, 10**9 + W))
    search_regions.append((10**10, 10**10 + W))
    
    # Random samples in the upper range
    np.random.seed(42)
    for _ in range(20):
        if time_left() < 30:
            break
        start = np.random.randint(10**9, 10**12 - W)
        search_regions.append((start, start + W))
    
    # Process regions in order of likelihood (larger values first)
    search_regions.sort(key=lambda x: -x[0])
    
    for lo, hi in search_regions:
        if time_left() < 10:
            break
        if hi > 10**12:
            hi = 10**12
            lo = max(1, hi - W + 1)
        if lo < 1:
            lo = 1
        if lo >= hi:
            continue
            
        phi = compute_phi_block(lo, hi)
        local_k, local_s = find_best_run_in_phi(phi, lo)
        
        if local_k > best_k:
            best_k = local_k
            best_s = local_s
            
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
        
        # If we found a good run, try to extend it by searching nearby
        if best_k >= 50 and time_left() > 20:
            # Search around the best region found
            center = best_s + best_k // 2
            for offset in [-W//2, -W//4, W//4, W//2, -W, W]:
                if time_left() < 15:
                    break
                new_lo = center + offset
                new_hi = new_lo + W
                if new_lo < 1:
                    new_lo = 1
                if new_hi > 10**12:
                    new_hi = 10**12
                    new_lo = max(1, new_hi - W + 1)
                if new_lo >= new_hi:
                    continue
                
                phi = compute_phi_block(new_lo, new_hi)
                local_k, local_s = find_best_run_in_phi(phi, new_lo)
                
                if local_k > best_k:
                    best_k = local_k
                    best_s = local_s
                    if best_k == 200:
                        return {'start': best_s, 'length': best_k}
    
    # If we still have time, do a more thorough search
    if time_left() > 30 and best_k < 100:
        # Sequential scan from a good starting point
        lo = 10**7
        while lo <= 10**12 and time_left() > 15:
            hi = min(lo + W - 1, 10**12)
            phi = compute_phi_block(lo, hi)
            local_k, local_s = find_best_run_in_phi(phi, lo)
            
            if local_k > best_k:
                best_k = local_k
                best_s = local_s
                if best_k == 200:
                    return {'start': best_s, 'length': best_k}
            
            # Skip ahead if we're not finding good runs
            if local_k < 10:
                lo = hi + W * 10
            else:
                lo = hi + 1
    
    return {'start': best_s, 'length': best_k}
