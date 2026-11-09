import numpy as np
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n in [1, N]
    # Using the recurrence:
    # If n = p * m and p does not divide m: sigma(n) = sigma(m) * (1 + p)
    # If n = p * m and p divides m: sigma(n) = sigma(m) + n
    # We need the smallest prime factor (SPF) or similar.
    # Actually, we can just iterate and use the property directly if we store the "power part".
    # But SPF sieve is standard.
    
    # SPF sieve
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    sigma = np.ones(N + 1, dtype=np.int64)
    # p_power_sum[n] stores sum of powers of the smallest prime factor p of n, up to the power in n.
    # i.e., if n = p^k * m, p_power_sum[n] = 1 + p + ... + p^k
    # p_power[n] stores p^k
    p_power_sum = np.zeros(N + 1, dtype=np.int64)
    p_power = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(2, N + 1):
        if time.time() > deadline:
            break # Fallback to seed if too slow
        p = spf[i]
        if p == 0:
            p = i # i is prime
        m = i // p
        if m % p == 0:
            # p divides m, so i = p^(k+1) * m'
            p_power[i] = p_power[m] * p
            p_power_sum[i] = p_power_sum[m] + p_power[i]
            sigma[i] = sigma[m] + i
        else:
            # p does not divide m, so i = p * m
            p_power[i] = p
            p_power_sum[i] = 1 + p
            sigma[i] = sigma[m] * (1 + p)
            
    # 2. Group by sigma value
    # We only care about n >= 1
    # sigma[0] is 0, sigma[1] is 1.
    
    # Sort indices 1..N by sigma value
    # To save time, we can just work with the array directly.
    # argsort of 10M elements takes ~1-2s.
    
    indices = np.arange(1, N + 1)
    sorted_indices = indices[np.argsort(sigma[1:], kind='stable')]
    sorted_sigmas = sigma[1:][np.argsort(sigma[1:], kind='stable')]
    
    # Find segment boundaries
    # diff of sorted_sigmas
    diff = np.diff(sorted_sigmas)
    change_points = np.where(diff != 0)[0] + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(sorted_sigmas)]))
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Process segments
    # We want to find pairs (a, b) in the same segment such that |a/b - 1.5| is min.
    # |a/b - 1.5| = |2a - 3b| / (2b)
    
    # To optimize, we can skip segments where the max possible b is too small to beat best_error.
    # Max b in a segment is the max value in sorted_indices[start:end].
    # Since sorted_indices is not sorted by value, we need max(sorted_indices[start:end]).
    
    for i in range(len(starts)):
        if time.time() > deadline:
            break
        s = starts[i]
        e = ends[i]
        if e - s < 2:
            continue
            
        # Get the segment of numbers
        seg = sorted_indices[s:e]
        
        # Check if this segment can improve best_error
        # Max possible b is max(seg)
        # Min possible error is 1 / (2 * max_b)
        max_b = np.max(seg)
        if 1.0 / (2.0 * max_b) >= best_error:
            continue
            
        # Sort the segment by value for binary search
        seg_sorted = np.sort(seg)
        
        # For each b in seg_sorted, find a in seg_sorted closest to 1.5 * b
        # We can iterate over b
        # To speed up, we can iterate over the larger b's first?
        # Or just iterate all.
        
        # Optimization: If we find an error of 1/(2b), that's the best possible for that b.
        # We can break early if we find |2a - 3b| == 1.
        
        for b in seg_sorted:
            # Target a is 1.5 * b
            target = 1.5 * b
            
            # Binary search in seg_sorted
            # bisect requires a list or array. np.array works with bisect?
            # bisect.bisect_left works on sequences.
            pos = bisect.bisect_left(seg_sorted, target)
            
            # Check pos and pos-1
            candidates = []
            if pos < len(seg_sorted):
                candidates.append(seg_sorted[pos])
            if pos > 0:
                candidates.append(seg_sorted[pos - 1])
                
            for a in candidates:
                if a == b:
                    continue
                # Calculate error
                # err = |a/b - 1.5| = |2a - 3b| / (2b)
                # Use integer arithmetic to avoid float precision issues in comparison?
                # |2a - 3b| is integer.
                num = abs(2 * a - 3 * b)
                if num == 0:
                    # Invalid per task (raw value 0)
                    continue
                err = num / (2.0 * b)
                
                if err < best_error:
                    best_error = err
                    best_a = int(a)
                    best_b = int(b)
                    
                    # If we achieved the theoretical minimum for this b, we can't do better for this b
                    # But we still need to check other b's.
                    # However, if num == 1, that's the smallest non-zero numerator.
                    # So for this b, we are done.
                    if num == 1:
                        break
                        
    # Ensure types are Python int
    return [int(best_a), int(best_b)]

if __name__ == "__main__":
    # Test
    pass
