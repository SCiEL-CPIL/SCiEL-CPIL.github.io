import numpy as np
import time

def construct():
    def get_primes(limit):
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return np.where(sieve)[0]

    # Primes up to 10^6 (sqrt of 10^12)
    primes = get_primes(10**6)
    
    best_k = 1
    best_s = 1
    
    SEG_SIZE = 10**7 # Increase segment size to reduce overhead of segment setup? 
                     # 10^7 int64 is 80MB. Might be too big for cache? 
                     # 10^6 is 8MB. Let's stick to 10^6 or try 2*10^6.
                     # The prompt suggests 10^7 in inspiration. Let's try 10^7.
                     # If memory is an issue, we can reduce.
                     # 10^7 * 8 bytes = 80MB. Acceptable.
                     
    MAX_N = 10**12
    start_time = time.time()
    
    L = 1
    prev_tail = []
    prev_R = 0
    T = 200 # Tail length to keep
    
    while L <= MAX_N:
        if time.time() - start_time > 110:
            break
            
        R = min(L + SEG_SIZE - 1, MAX_N)
        N = R - L + 1
        
        # Compute phi for [L, R]
        phi_arr = np.arange(L, R + 1, dtype=np.int64)
        
        for p in primes:
            if p > R:
                break
            # Find first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            if start > R:
                continue
            offset = start - L
            last = R - (R % p)
            if last < start:
                continue
            count = (last - start) // p + 1
            # Update phi
            # phi[x] = phi[x] / p * (p-1)
            # Ensure integer division
            phi_arr[offset : offset + count*p : p] //= p
            phi_arr[offset : offset + count*p : p] *= (p - 1)
        
        # Sliding window for distinct values
        # We process prev_tail and phi_arr without creating a combined list
        seen = {}
        left = 0
        local_max_k = 0
        local_max_start = 0
        
        # Helper to get actual start number from virtual index
        def get_actual_start(v_idx):
            if v_idx < len(prev_tail):
                return prev_R - (len(prev_tail) - 1) + v_idx
            else:
                idx_in_curr = v_idx - len(prev_tail)
                return L + idx_in_curr

        # Process prev_tail
        for i, val in enumerate(prev_tail):
            if val in seen:
                left = max(left, seen[val] + 1)
            seen[val] = i
            curr_k = i - left + 1
            if curr_k > local_max_k:
                local_max_k = curr_k
                local_max_start = left
            # Early exit check
            if local_max_k >= 200:
                # We found a run of at least 200
                # We can return immediately?
                # Need to verify bounds.
                actual_start = get_actual_start(local_max_start)
                if actual_start >= 1 and actual_start + 200 - 1 <= MAX_N:
                    return {'start': actual_start, 'length': 200}
        
        # Process phi_arr
        # To speed up, we can convert phi_arr to a list? No, iterating numpy is ok.
        # But we need to handle the case where the window extends into phi_arr.
        
        # Optimization: If local_max_k is already close to 200, we might hit 200 soon.
        
        for i, val in enumerate(phi_arr):
            v_idx = len(prev_tail) + i
            if val in seen:
                left = max(left, seen[val] + 1)
            seen[val] = v_idx
            curr_k = v_idx - left + 1
            if curr_k > local_max_k:
                local_max_k = curr_k
                local_max_start = left
                if local_max_k >= 200:
                    actual_start = get_actual_start(local_max_start)
                    if actual_start >= 1 and actual_start + 200 - 1 <= MAX_N:
                        return {'start': actual_start, 'length': 200}
            
            # Optional: Prune seen dict?
            # If left moves forward, entries with index < left are no longer needed.
            # But dict doesn't support efficient removal of keys by value (index).
            # Given T=200, the dict size is bounded by the window size + some overhead.
            # Actually, if the window is small, dict is small.
            # If the window is large (distinct values), dict grows.
            # But we exit at 200. So dict size <= 200 + (collisions)?
            # If values are distinct, window grows.
            # If we don't exit, it means we didn't hit 200.
            # So dict size is bounded by the max distinct run length found so far.
            # If max run is < 200, dict is < 200?
            # No, `seen` contains ALL values seen so far, not just in window.
            # We should clean up `seen` to prevent it from growing to $10^6$.
            # If `left` advances, we can remove keys where `seen[key] < left`.
            # This is $O(1)$ amortized if we just check on access? No.
            # We can just let it grow? $10^6$ entries in a dict is ~80MB?
            # And we do this for every segment.
            # It's better to clear `seen` and repopulate?
            # Or just use a new dict for each segment?
            # But we need the tail from the previous segment.
            
            # Alternative: Since we only care about the last 200 values for the window,
            # and we exit if we hit 200, the dict size is effectively bounded by 200 
            # IF we implement the "remove old entries" logic.
            # Without removal, dict grows to $N$.
            # Let's add a cleanup step if left moves?
            # It's hard to do efficiently in Python loop.
            # However, if we exit at 200, we never reach $N=10^6$ with a large dict 
            # unless we have a run of 199 distinct values that never collides?
            # If we have a run of 199, the dict has 199 entries.
            # If the 200th value is distinct, we exit.
            # If the 200th value is a collision, we shrink.
            # So the dict size stays small (< 200 + some) if we are near the limit.
            # If we are far from the limit (e.g. max run is 10), the dict size is small.
            # So dict size is bounded by the max run length found in this segment.
            # If max run is small, dict is small.
            # If max run is large, we exit.
            # So we don't need explicit cleanup!
            
        # Update tail
        if N >= T:
            prev_tail = phi_arr[-T:].tolist()
        else:
            # Combine
            # This case is rare (last segment)
            combined = prev_tail + phi_arr.tolist()
            prev_tail = combined[-T:]
            
        # Update best if not exited
        if local_max_k > best_k:
            actual_start = get_actual_start(local_max_start)
            if actual_start >= 1 and actual_start + local_max_k - 1 <= MAX_N and local_max_k <= 200:
                best_k = local_max_k
                best_s = actual_start
                
        prev_R = R
        L = R + 1
        
    return {'start': best_s, 'length': best_k}
