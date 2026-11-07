import time
import numpy as np
import bisect

def construct():
    t0 = time.time()
    LIMIT = 60.0
    SAFETY = 0.85
    hard_stop = t0 + LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    N = 10000000
    fallback = [11, 6]

    try:
        # 1. SPF Sieve
        spf = np.arange(N + 1, dtype=np.int32)
        limit_spf = int(np.sqrt(N)) + 1
        for i in range(2, limit_spf):
            if spf[i] == i:
                start = i * i
                indices = np.arange(start, N + 1, i)
                mask = spf[indices] == indices
                spf[indices[mask]] = i
                
        if time_left() < 15.0:
            return fallback

        # 2. Compute Sigma
        sigma = np.ones(N + 1, dtype=np.int64)
        core = np.arange(N + 1, dtype=np.int32)
        p_pow = np.ones(N + 1, dtype=np.int32)
        sigma_p_pow = np.ones(N + 1, dtype=np.int64)
        
        _sigma = sigma
        _core = core
        _p_pow = p_pow
        _sigma_p_pow = sigma_p_pow
        _spf = spf
        
        for n in range(2, N + 1):
            p = _spf[n]
            m = n // p
            if m % p == 0:
                _core[n] = _core[m]
                _p_pow[n] = _p_pow[m] * p
                _sigma_p_pow[n] = _sigma_p_pow[m] + _p_pow[n]
            else:
                _core[n] = m
                _p_pow[n] = p
                _sigma_p_pow[n] = 1 + p
            _sigma[n] = _sigma[_core[n]] * _sigma_p_pow[n]
            
            if (n & 0x3FFFF) == 0 and time_left() < 10.0:
                return fallback

        if time_left() < 5.0:
            return fallback

        # 3. Group by Sigma using Dictionary
        # This is O(N) average, potentially faster than O(N log N) sort
        # We only care about groups with size >= 2
        # To save memory/time, we can just store the numbers.
        
        # Using a dictionary of lists
        # Key: sigma value (int)
        # Value: list of n
        
        # Optimization: We can pre-allocate or use a dict.
        # Number of distinct sigmas is ~N.
        # Dict overhead is high.
        # Alternative: Sort is safer for memory if N is large.
        # Let's stick to sort but optimize the group iteration.
        
        # Actually, let's try the dictionary approach but only keep small groups?
        # No, we need all groups to find the best pair.
        
        # Let's stick to the sort approach but ensure it's fast.
        # The sort is the bottleneck.
        
        idx = np.arange(1, N + 1, dtype=np.int32)
        vals = _sigma[1:]
        
        order = np.argsort(vals, kind='stable')
        svals = vals[order]
        
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
                
            # Get the sorted numbers in this group
            # gn is already sorted because order is stable and idx is 1..N
            # Wait, idx is 1..N. order is indices into idx.
            # So idx[order[s0:e0]] are the numbers with that sigma.
            # Since idx is sorted 1..N, and order is stable sort of vals,
            # are the numbers in gn sorted?
            # No. `order` sorts `vals`. `idx[order]` gives the original indices.
            # The original indices are 1..N.
            # If we sort by sigma, the resulting list of indices is NOT sorted by value.
            # We need to sort the numbers in the group to use two-pointer.
            # The current code does: gn = np.sort(idx[order[s0:e0]])
            # This is O(K log K) per group.
            
            gn = np.sort(idx[order[s0:e0]])
            gn_list = gn.tolist()
            L = len(gn_list)
            
            # Two-pointer approach
            i_a = 0
            # We want to minimize |2a - 3b| / (2b)
            # For each b, find best a.
            # b iterates through gn_list.
            # Since gn_list is sorted, b increases.
            # Target 1.5b increases.
            
            for bi in range(L):
                b = gn_list[bi]
                # Optimization: if b is too small or too large?
                # a must be <= N.
                # 1.5 b <= N + margin?
                # If 1.5 b > N, the best a is N (max possible).
                # If 1.5 b < 1, best a is 1.
                
                target = 1.5 * b
                
                # Advance i_a
                while i_a < L and gn_list[i_a] < target:
                    i_a += 1
                    
                # Check i_a
                if i_a < L:
                    a = gn_list[i_a]
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)
                
                # Check i_a - 1
                if i_a > 0:
                    a = gn_list[i_a - 1]
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)

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
