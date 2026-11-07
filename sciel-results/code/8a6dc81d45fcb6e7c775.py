import time
import numpy as np

def construct():
    t0 = time.time()
    LIMIT = 60.0
    SAFETY = 0.85
    hard_stop = t0 + LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    N = 10000000
    fallback = [11, 6] # sigma(11)=12, sigma(6)=12. 11/6 = 1.833. |1.833 - 1.5| = 0.333.
    
    try:
        # 1. SPF Sieve
        spf = np.arange(N + 1, dtype=np.int32)
        # Only need to loop to sqrt(N)
        limit_spf = int(np.sqrt(N)) + 1
        for i in range(2, limit_spf):
            if spf[i] == i:
                # Vectorized update
                # Start from i*i
                # Indices: i*i, i*i + i, ...
                # We only update if spf[j] is still j (i.e., not marked by smaller prime)
                # Actually, if spf[j] == j, it means j is prime OR not visited by smaller primes?
                # No, if j is composite, it must have a factor < j.
                # If j is composite and not marked by any prime < i, then i is the smallest factor.
                # So checking spf[j] == j is correct.
                
                # Create indices
                start = i * i
                indices = np.arange(start, N + 1, i)
                # Mask: where spf[indices] == indices
                mask = spf[indices] == indices
                # Update
                spf[indices[mask]] = i
                
        if time_left() < 15.0:
            return fallback

        # 2. Compute Sigma
        sigma = np.ones(N + 1, dtype=np.int64)
        core = np.arange(N + 1, dtype=np.int32) # core[n] = n with all factors of spf[n] removed
        p_pow = np.ones(N + 1, dtype=np.int32)  # p_pow[n] = highest power of spf[n] dividing n
        sigma_p_pow = np.ones(N + 1, dtype=np.int64) # sigma_p_pow[n] = sigma(p_pow[n])
        
        # Initialize for n=1
        sigma[1] = 1
        core[1] = 1
        p_pow[1] = 1
        sigma_p_pow[1] = 1

        # Python loop for DP
        # This loop is the bottleneck. 10^7 iterations.
        # Optimize by local variables
        _sigma = sigma
        _core = core
        _p_pow = p_pow
        _sigma_p_pow = sigma_p_pow
        _spf = spf
        
        for n in range(2, N + 1):
            p = _spf[n]
            m = n // p
            if m % p == 0:
                # p divides m, so spf[m] == p
                _core[n] = _core[m]
                _p_pow[n] = _p_pow[m] * p
                _sigma_p_pow[n] = _sigma_p_pow[m] + _p_pow[n]
            else:
                # p does not divide m, so spf[m] != p (or m=1)
                # gcd(m, p) = 1
                _core[n] = m
                _p_pow[n] = p
                _sigma_p_pow[n] = 1 + p
            _sigma[n] = _sigma[_core[n]] * _sigma_p_pow[n]
            
            if (n & 0x3FFFF) == 0 and time_left() < 10.0:
                return fallback

        if time_left() < 5.0:
            return fallback

        # 3. Group by Sigma
        # We need indices 1..N
        idx = np.arange(1, N + 1, dtype=np.int32)
        vals = _sigma[1:]
        
        order = np.argsort(vals, kind='stable')
        svals = vals[order]
        
        # Find group boundaries
        M = order.size
        change = np.empty(M, dtype=bool)
        change[0] = True
        change[1:] = svals[1:] != svals[:-1]
        starts = np.nonzero(change)[0]
        ends = np.append(starts[1:], M)
        
        best_val = float('inf')
        best_pair = None
        
        # Iterate groups
        for gi in range(starts.size):
            if time_left() < 2.0:
                break
                
            s0 = int(starts[gi])
            e0 = int(ends[gi])
            length = e0 - s0
            if length < 2:
                continue
                
            # Extract and sort the group elements
            # idx[order[s0:e0]] gives the original numbers
            group_nums = np.sort(idx[order[s0:e0]])
            
            # Vectorized search for best pair in this group
            # We want to minimize |2a - 3b| / (2b) for a, b in group_nums, a != b
            targets = 1.5 * group_nums
            j_left = np.searchsorted(group_nums, targets, side='left')
            
            L = len(group_nums)
            
            local_best_val = float('inf')
            local_best_a = 0
            local_best_b = 0
            
            # Check candidates from the right (a >= target)
            valid_left = j_left < L
            if np.any(valid_left):
                a_cands = group_nums[j_left[valid_left]]
                b_cands = group_nums[valid_left]
                
                num = np.abs(2 * a_cands - 3 * b_cands)
                mask_valid = num > 0
                if np.any(mask_valid):
                    a_v = a_cands[mask_valid]
                    b_v = b_cands[mask_valid]
                    num_v = num[mask_valid]
                    den_v = 2 * b_v
                    val = num_v / den_v
                    
                    min_idx = np.argmin(val)
                    if val[min_idx] < local_best_val:
                        local_best_val = val[min_idx]
                        local_best_a = a_v[min_idx]
                        local_best_b = b_v[min_idx]
                        
            # Check candidates from the left (a < target)
            valid_right = j_left > 0
            if np.any(valid_right):
                j_right = j_left[valid_right] - 1
                a_cands = group_nums[j_right]
                b_cands = group_nums[valid_right]
                
                num = np.abs(2 * a_cands - 3 * b_cands)
                # Exclude a == b and num == 0
                mask_valid = (num > 0) & (a_cands != b_cands)
                if np.any(mask_valid):
                    a_v = a_cands[mask_valid]
                    b_v = b_cands[mask_valid]
                    num_v = num[mask_valid]
                    den_v = 2 * b_v
                    val = num_v / den_v
                    
                    min_idx = np.argmin(val)
                    if val[min_idx] < local_best_val:
                        local_best_val = val[min_idx]
                        local_best_a = a_v[min_idx]
                        local_best_b = b_v[min_idx]
            
            if local_best_val < best_val:
                best_val = local_best_val
                best_pair = (int(local_best_a), int(local_best_b))

        if best_pair is None:
            return fallback
            
        a, b = int(best_pair[0]), int(best_pair[1])
        if not (1 <= a <= N and 1 <= b <= N):
            return fallback
        if 2 * a == 3 * b:
            return fallback
            
        return [a, b]

    except Exception:
        return [11, 6]
