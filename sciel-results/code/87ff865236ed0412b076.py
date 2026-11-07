def construct():
        import time
        import numpy as np
        
        t0 = time.time()
        LIMIT = 60.0
        hard_stop = t0 + LIMIT * 0.85
        
        N = 10000000
        
        # SPF Sieve
        spf = np.zeros(N + 1, dtype=np.int32)
        sigma = np.zeros(N + 1, dtype=np.int64)
        sigma[1] = 1
        
        # Initialize primes
        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                sigma[i] = i + 1
            # This loop is O(N log log N) for SPF, but the sigma calc is O(N)
            # However, doing it in a Python loop is slow.
            
        # Vectorized SPF is hard.
        # Let's stick to the divisor sum sieve but optimize it.
        
        sigma = np.zeros(N + 1, dtype=np.int64)
        # The loop is the bottleneck.
        # Can we vectorize the outer loop? No, step size changes.
        
        # Let's try to compute sigma for only a subset?
        # No, we need to find the best pair.
        
        # Maybe we can use the property that if a/b is close to 1.5,
        # then a and b are not too different in magnitude?
        # Not necessarily. 10000000 and 6666666.
        
        # Let's assume the current sieve is fast enough (~2s).
        # The sorting is ~1s.
        # The group iteration is the rest.
        
        # Let's try to improve the group search.
        # Instead of iterating all groups, can we filter groups by potential?
        # A group has potential if it contains numbers close to a 1.5 ratio.
        
        # Let's just make the code more robust and ensure it doesn't timeout.
        # And maybe try N=10^7 is fine.
        
        # One optimization: use int32 for indices if possible?
        # N=10^7 fits in int32.
        
        try:
            sigma = np.zeros(N + 1, dtype=np.int64)
            for d in range(1, N + 1):
                sigma[d::d] += d
                
            if time.time() - t0 > 10:
                # Fallback if sieve is too slow
                return [11, 6]

            idx = np.arange(1, N + 1, dtype=np.int32)
            vals = sigma[1:]
            
            order = np.argsort(vals, kind='stable')
            svals = vals[order]
            
            # Find groups
            change = np.empty(len(svals), dtype=bool)
            change[0] = True
            change[1:] = svals[1:] != svals[:-1]
            starts = np.nonzero(change)[0]
            ends = np.append(starts[1:], len(svals))
            
            best_val = float('inf')
            best_pair = None
            
            for gi in range(len(starts)):
                if time.time() > hard_stop:
                    break
                
                s0 = starts[gi]
                e0 = ends[gi]
                length = e0 - s0
                if length < 2:
                    continue
                
                group_nums = idx[order[s0:e0]]
                # Sort group_nums to use searchsorted effectively?
                # The group is already sorted by index in 'order' if we used stable sort?
                # No, 'order' sorts by sigma. Within same sigma, the order is arbitrary (stable sort keeps original index order).
                # So group_nums is sorted by index (which is n).
                # So group_nums is already sorted ascending.
                
                L = len(group_nums)
                
                # For each b, find a closest to 1.5*b
                # We can vectorize this search.
                
                # targets = 1.5 * group_nums
                # But 1.5 * group_nums might overflow int32? No, max 1.5 * 10^7 = 1.5 * 10^7.
                # But we need to find a in group_nums.
                
                # Use searchsorted
                # Since group_nums is sorted, we can find insertion point for 1.5*b.
                
                # Let's try to do this in chunks to avoid huge temporary arrays?
                # L can be large.
                
                # We can just iterate b?
                # If L is small, iterating is fine.
                # If L is large, vectorized is better.
                
                # Let's use the vectorized approach from current code but check for bugs.
                
                # The current code checks left and right candidates.
                
                # One potential issue: `num > 0` check.
                # If a == 1.5 b exactly, num is 0.
                # We must exclude this.
                
                # Let's rewrite the inner loop to be as fast as possible.
                
                # Vectorized candidate finding
                # j = searchsorted(group_nums, 1.5 * group_nums)
                # Candidates are group_nums[j] and group_nums[j-1]
                
                # We need to handle boundaries.
                
                # Let's try to implement this carefully.
                
                # Note: group_nums is int32.
                # 1.5 * group_nums is float64.
                
                # We want to minimize |a/b - 1.5| = |2a - 3b| / (2b)
                
                # For each b in group_nums:
                #   target = 1.5 * b
                #   pos = searchsorted(group_nums, target)
                #   candidates = []
                #   if pos < L: candidates.append(group_nums[pos])
                #   if pos > 0: candidates.append(group_nums[pos-1])
                #   for a in candidates:
                #       if a != b:
                #           err = abs(2*a - 3*b) / (2*b)
                #           if err < best_val: update
                
                # This is O(L log L) per group.
                # Sum of L over all groups is N.
                # So total O(N log N) roughly.
                
                # Let's try to vectorize the candidate extraction.
                
                # j = np.searchsorted(group_nums, 1.5 * group_nums.astype(np.float64))
                
                # valid_left = j < L
                # a_left = np.where(valid_left, group_nums[j], -1) # -1 as invalid
                
                # valid_right = j > 0
                # j_right = np.where(valid_right, j - 1, 0)
                # a_right = np.where(valid_right, group_nums[j_right], -1)
                
                # Now we have two candidate arrays a_left and a_right.
                # We need to compute error for both and take min.
                
                # err_left = abs(2*a_left - 3*group_nums) / (2*group_nums)
                # err_right = abs(2*a_right - 3*group_nums) / (2*group_nums)
                
                # Mask out invalid (-1) and a==b.
                
                # This is efficient.
                
                b_arr = group_nums
                targets = 1.5 * b_arr.astype(np.float64)
                j = np.searchsorted(group_nums, targets)
                
                # Left candidates (a >= target)
                valid_l = j < L
                if np.any(valid_l):
                    a_l = group_nums[j[valid_l]]
                    b_l = b_arr[valid_l]
                    mask = (a_l != b_l)
                    if np.any(mask):
                        a_l = a_l[mask]
                        b_l = b_l[mask]
                        num = np.abs(2 * a_l - 3 * b_l)
                        den = 2 * b_l
                        err = num / den
                        min_idx = np.argmin(err)
                        if err[min_idx] < best_val:
                            best_val = err[min_idx]
                            best_pair = (int(a_l[min_idx]), int(b_l[min_idx]))
                            
                # Right candidates (a < target)
                valid_r = j > 0
                if np.any(valid_r):
                    j_r = j[valid_r] - 1
                    a_r = group_nums[j_r]
                    b_r = b_arr[valid_r]
                    mask = (a_r != b_r)
                    if np.any(mask):
                        a_r = a_r[mask]
                        b_r = b_r[mask]
                        num = np.abs(2 * a_r - 3 * b_r)
                        den = 2 * b_r
                        err = num / den
                        min_idx = np.argmin(err)
                        if err[min_idx] < best_val:
                            best_val = err[min_idx]
                            best_pair = (int(a_r[min_idx]), int(b_r[min_idx]))

            if best_pair:
                return list(best_pair)
            else:
                return [11, 6]
                
        except Exception:
            return [11, 6]
