import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy
    # This is O(N log N) but vectorized
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if time.time() > deadline - 10.0:
            # Fallback if sieve takes too long
            return [11, 6]

    # 2. Arithmetic Search for small k
    # We want to minimize |2a - 3b| / (2b)
    # Let k = 2a - 3b. We search for small |k|.
    # For a fixed k, a = (3b + k) / 2.
    # a is integer iff 3b + k is even.
    # We iterate k from 1 to K_MAX.
    
    best_err = float('inf')
    best_a = 11
    best_b = 6
    
    # We can vectorize the check for each k
    # Valid b range for a given k:
    # 1 <= (3b + k) / 2 <= N  =>  2 - k <= 3b <= 2N - k
    # b >= ceil((2 - k) / 3)
    # b <= floor((2N - k) / 3)
    # Also 1 <= b <= N
    
    K_MAX = 200  # Search for small errors
    
    for k in range(1, K_MAX + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Determine valid b range
        b_min = max(1, (2 - k + 2) // 3)  # ceil((2-k)/3)
        # Careful with integer division for negative/positive
        # If (2-k) is negative, ceil is negative, so b_min is 1.
        # If (2-k) is positive, ceil is (2-k+2)//3.
        # Actually, simpler: b_min = max(1, int(np.ceil((2 - k) / 3.0)))
        # But we want integer ops.
        # 3b >= 2 - k
        # b >= (2 - k) / 3
        
        if k <= 2:
            b_min = 1
        else:
            # 2 - k is negative, so b >= negative number, so b >= 1
            b_min = 1
            
        b_max = min(N, (2 * N - k) // 3)
        
        if b_min > b_max:
            continue
            
        # We need 3b + k to be even.
        # If k is odd, 3b must be odd => b must be odd.
        # If k is even, 3b must be even => b must be even.
        
        # Create array of valid b's
        # This can be large (up to N/3).
        # To save memory and time, we might want to process in chunks or just use the full array if N is 10^7.
        # N/3 is 3.3M elements. int64 is 8 bytes. 3.3M * 8 = 26 MB. This is fine.
        
        # Step is 2 to enforce parity
        start_b = b_min
        if k % 2 != 0:
            # k odd, b must be odd
            if start_b % 2 == 0:
                start_b += 1
        else:
            # k even, b must be even
            if start_b % 2 == 1:
                start_b += 1
        
        if start_b > b_max:
            continue
            
        b_vals = np.arange(start_b, b_max + 1, 2, dtype=np.int64)
        
        # Compute a
        a_vals = (3 * b_vals + k) // 2
        
        # Check bounds (should be within [1, N] by construction, but just in case)
        # a_vals >= 1 and a_vals <= N
        
        # Get sigma values
        sig_a = sigma[a_vals]
        sig_b = sigma[b_vals]
        
        # Find matches
        matches = sig_a == sig_b
        
        if np.any(matches):
            # Get indices of matches
            match_indices = np.where(matches)[0]
            # We want to minimize |k| / (2 * b) = |k| / (2 * b_vals[match_indices])
            # Since k is fixed, we want to maximize b.
            # So we just need the max b among the matches.
            max_b = b_vals[match_indices].max()
            max_a = a_vals[np.where(b_vals[match_indices] == max_b)[0][0]]
            
            # Calculate error
            # err = |k| / (2 * max_b)
            err = float(abs(k)) / (2.0 * float(max_b))
            
            if err < best_err:
                best_err = err
                best_a = int(max_a)
                best_b = int(max_b)
                
                # If we found a very small error, we might stop early?
                # Theoretical min error for N=10^7 is ~ 1/(2*10^7) = 5e-8.
                # If err < 1e-8, it's likely optimal or near optimal.
                if err < 1e-8:
                    # We can break out of k loop if we are confident
                    # But there might be a larger k with even larger b?
                    # No, larger k means larger numerator.
                    # So smaller k is always better if b is same.
                    # But if k=1 has no match, k=2 might have a match with larger b.
                    # We should probably continue to check smaller k's if we haven't checked them all?
                    # We are iterating k=1, 2, ... so we are checking smallest k first.
                    # If we find a match for k=1, it's likely the best unless there's a k=1 match with larger b?
                    # We already took max b for this k.
                    # So for a fixed k, we found the best pair.
                    # Since k is increasing, any future pair will have error >= k'/(2b) > k/(2b) if b is similar.
                    # However, b could be larger for larger k?
                    # b_max for k is ~ 2N/3.
                    # So b is bounded.
                    # Thus, smaller k is generally better.
                    # We can break if k=1 has a match?
                    # Not necessarily, if k=1 match has small b, and k=2 match has huge b.
                    # But k=1 error = 1/(2b1). k=2 error = 2/(2b2) = 1/b2.
                    # If b2 > 2*b1, then k=2 is better.
                    # So we cannot break immediately.
                    # But we can break if best_err is extremely small, e.g., < 1e-9.
                    pass

    # 3. Fallback / Verification with Group Search if needed
    # If the arithmetic search didn't find a good pair, or to be safe,
    # we can run a quick group search for the best pair found so far.
    # But given the time budget, and that arithmetic search is very fast for small k,
    # we might have time for a limited group search.
    
    # Let's do a quick check: if best_err is still inf or large, do group search.
    if best_err > 1e-6:
        # Perform group search
        # This is expensive, so we only do it if arithmetic search failed to find a good solution.
        
        # Get unique sigma values and their indices
        # This is O(N log N)
        if time.time() < deadline - 10.0:
            sorted_indices = np.argsort(sigma[1:], kind='stable')
            sorted_sigmas = sigma[1:][sorted_indices]
            
            i = 0
            n = len(sorted_sigmas)
            while i < n:
                if time.time() > deadline - 5.0:
                    break
                s = sorted_sigmas[i]
                j = i
                while j < n and sorted_sigmas[j] == s:
                    j += 1
                if j - i >= 2:
                    # Extract numbers
                    # Note: sorted_indices are 0-based, so add 1
                    nums = sorted_indices[i:j] + 1
                    nums_sorted = np.sort(nums)
                    nlen = len(nums_sorted)
                    
                    # For each b, find best a
                    # We can vectorize this?
                    # For each b, target a = 1.5 b.
                    # Use bisect on nums_sorted.
                    # This is O(M log M) in Python loop.
                    # If M is large, this is slow.
                    # But most groups are small.
                    
                    for idx_b in range(nlen):
                        b = int(nums_sorted[idx_b])
                        target_a = 1.5 * b
                        # Binary search
                        lo, hi = 0, nlen - 1
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if nums_sorted[mid] < target_a:
                                lo = mid + 1
                            else:
                                hi = mid
                        # Check lo and lo-1
                        for idx_a in (lo, lo - 1):
                            if 0 <= idx_a < nlen:
                                a = int(nums_sorted[idx_a])
                                if a == b:
                                    continue
                                # Check 2a == 3b
                                if 2 * a == 3 * b:
                                    continue
                                num = abs(2 * a - 3 * b)
                                den = 2 * b
                                err = num / den
                                if err < best_err:
                                    best_err = err
                                    best_a = a
                                    best_b = b
                i = j

    # Final check
    if best_a == 0 or best_b == 0:
        return [11, 6]
        
    return [int(best_a), int(best_b)]
