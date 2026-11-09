import time

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):
        val_a = 2 ** a
        for b in range(0, 5):
            val_ab = val_a * (3 ** b)
            if val_ab > 128:
                break
            for c in range(0, 4):
                val_abc = val_ab * (5 ** c)
                if val_abc > 128:
                    break
                smooth.add(val_abc)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # Precompute incompatibility masks
    # incompat_mask[i] has bit j set if smooth_list[i] and smooth_list[j] are incompatible (one divides the other)
    incompat_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth_list[i], smooth_list[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                incompat_mask[i] |= (1 << j)
                incompat_mask[j] |= (1 << i)
    
    # Precompute prefix sums for pruning in backtracking
    # prefix_sum[i] = sum(smooth_list[0:i])
    prefix_sum = [0] * (n_smooth + 1)
    for i in range(n_smooth):
        prefix_sum[i + 1] = prefix_sum[i] + smooth_list[i]
        
    # total_sum = sum(smooth_list)
    # suffix_sum[i] = sum(smooth_list[i:])
    suffix_sum = [0] * (n_smooth + 1)
    for i in range(n_smooth - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + smooth_list[i]

    min_k_for_sum = {}
    
    # k=1
    for i in range(n_smooth):
        s = smooth_list[i]
        if s <= 128 and s not in min_k_for_sum:
            min_k_for_sum[s] = (1, [s])
            
    # k=2
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if (incompat_mask[i] >> j) & 1:
                continue
            s = smooth_list[i] + smooth_list[j]
            if s > 128:
                break # Since smooth_list is sorted, further j will be larger
            if s not in min_k_for_sum:
                min_k_for_sum[s] = (2, [smooth_list[i], smooth_list[j]])
                
    # k=3
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if (incompat_mask[i] >> j) & 1:
                continue
            s2 = smooth_list[i] + smooth_list[j]
            if s2 > 128:
                break
            for k in range(j+1, n_smooth):
                if (incompat_mask[i] >> k) & 1 or (incompat_mask[j] >> k) & 1:
                    continue
                s3 = s2 + smooth_list[k]
                if s3 > 128:
                    break
                if s3 not in min_k_for_sum:
                    min_k_for_sum[s3] = (3, [smooth_list[i], smooth_list[j], smooth_list[k]])
                    
    # k=4
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if (incompat_mask[i] >> j) & 1:
                continue
            s2 = smooth_list[i] + smooth_list[j]
            if s2 > 128:
                break
            for k in range(j+1, n_smooth):
                if (incompat_mask[i] >> k) & 1 or (incompat_mask[j] >> k) & 1:
                    continue
                s3 = s2 + smooth_list[k]
                if s3 > 128:
                    break
                for l in range(k+1, n_smooth):
                    if (incompat_mask[i] >> l) & 1 or (incompat_mask[j] >> l) & 1 or (incompat_mask[k] >> l) & 1:
                        continue
                    s4 = s3 + smooth_list[l]
                    if s4 > 128:
                        break
                    if s4 not in min_k_for_sum:
                        min_k_for_sum[s4] = (4, [smooth_list[i], smooth_list[j], smooth_list[k], smooth_list[l]])
                    
    # Fallback for uncovered numbers using backtracking
    # We search for minimal k starting from 5
    def find_antichain_sum(n, k):
        # Candidates are indices into smooth_list
        # We want to pick k distinct indices i1 < i2 < ... < ik
        # such that sum(smooth_list[ij]) == n and no pair is incompatible.
        
        def backtrack(start, rem_sum, rem_k, current_path, current_mask):
            if rem_k == 0:
                if rem_sum == 0:
                    return [smooth_list[idx] for idx in current_path]
                return None
            
            # Pruning:
            # 1. Not enough candidates left
            if n_smooth - start < rem_k:
                return None
            
            # 2. Max possible sum from remaining rem_k numbers
            # The largest rem_k numbers available from start onwards
            # are smooth_list[n_smooth-rem_k : n_smooth]
            # But we must ensure they are >= start.
            # If n_smooth - rem_k < start, then we can't pick rem_k numbers?
            # No, we pick from start to n_smooth-1.
            # The largest rem_k numbers in smooth_list[start:] are smooth_list[n_smooth-rem_k:]
            # provided n_smooth-rem_k >= start.
            
            max_start_idx = n_smooth - rem_k
            if max_start_idx < start:
                return None
            
            # Max sum is sum(smooth_list[max_start_idx : n_smooth])
            # This is suffix_sum[max_start_idx]
            if suffix_sum[max_start_idx] < rem_sum:
                return None
                
            # Min sum is sum(smooth_list[start : start+rem_k])
            # This is prefix_sum[start+rem_k] - prefix_sum[start]
            if prefix_sum[start + rem_k] - prefix_sum[start] > rem_sum:
                return None

            # Iterate candidates for the next element
            # We can restrict the upper bound of the current element.
            # smooth_list[i] must be <= rem_sum - (min sum of rem_k-1 numbers from i+1)
            # Min sum of rem_k-1 numbers from i+1 is prefix_sum[i+1+rem_k-1] - prefix_sum[i+1]
            
            end = n_smooth
            # We can break the loop if smooth_list[i] is too large.
            # If smooth_list[i] > rem_sum, break.
            
            for i in range(start, end):
                val = smooth_list[i]
                if val > rem_sum:
                    break
                
                # Check compatibility with current_path using bitmask
                # current_mask has bits set for indices in current_path
                # We need to check if i is incompatible with any in current_path
                # This is equivalent to checking if (incompat_mask[i] & current_mask) != 0
                if incompat_mask[i] & current_mask:
                    continue
                
                # Pruning: Can we form rem_sum - val with rem_k-1 numbers from i+1?
                if rem_k - 1 > 0:
                    # If i+1 + (rem_k-1) > n_smooth, not enough numbers
                    if i + 1 + (rem_k - 1) > n_smooth:
                        break # Since i is increasing, this will only get worse
                    
                    # Min sum of next rem_k-1 numbers
                    min_next_sum = prefix_sum[i + 1 + (rem_k - 1)] - prefix_sum[i + 1]
                    if min_next_sum > rem_sum - val:
                        break
                        
                    # Max sum of next rem_k-1 numbers
                    # Largest rem_k-1 numbers from i+1 are smooth_list[n_smooth-(rem_k-1):]
                    # provided n_smooth-(rem_k-1) >= i+1
                    max_next_start = n_smooth - (rem_k - 1)
                    if max_next_start < i + 1:
                        # Not enough numbers?
                        # Actually, if i+1 + rem_k-1 > n_smooth, we broke above.
                        # So there are at least rem_k-1 numbers.
                        # The largest rem_k-1 numbers are the last rem_k-1 in the list.
                        # But they must be > i.
                        # If n_smooth - (rem_k - 1) < i + 1, then some of the largest numbers are <= i.
                        # We need the largest rem_k-1 numbers from smooth_list[i+1:].
                        # These are smooth_list[n_smooth-(rem_k-1):n_smooth]
                        # IF n_smooth-(rem_k-1) >= i+1.
                        # If n_smooth-(rem_k-1) < i+1, then the set of available numbers is smaller.
                        # But we already checked i+1+rem_k-1 <= n_smooth, so there are enough numbers.
                        # The largest rem_k-1 numbers from i+1 to n_smooth-1 are indeed the last rem_k-1 numbers?
                        # Not necessarily if i is close to the end.
                        # Example: n_smooth=10, rem_k-1=2, i=8.
                        # Available: 9. Only 1 number. But we need 2.
                        # Wait, if i=8, rem_k-1=2, then i+1+2 = 11 > 10. Break.
                        # So if we reach here, i+1+rem_k-1 <= n_smooth.
                        # This implies n_smooth - (rem_k-1) >= i+1.
                        # So the largest rem_k-1 numbers are smooth_list[n_smooth-(rem_k-1):n_smooth].
                        # And their indices are >= n_smooth-(rem_k-1) >= i+1.
                        # So they are all > i.
                        max_next_sum = suffix_sum[n_smooth - (rem_k - 1)]
                        if max_next_sum < rem_sum - val:
                            # Even the largest numbers are too small.
                            # As i increases, rem_sum - val decreases.
                            # So this condition becomes easier to satisfy.
                            # We can't break, just continue?
                            # Actually, if max_next_sum < rem_sum - val, it's impossible for this i.
                            # For larger i, rem_sum - val is smaller, so it might be possible.
                            # So we continue.
                            pass

                current_path.append(i)
                res = backtrack(i+1, rem_sum - val, rem_k - 1, current_path, current_mask | (1 << i))
                if res:
                    return res
                current_path.pop()
                
            return None
        
        return backtrack(0, n, k, [], 0)

    answers = []
    for n in range(1, 129):
        if n in min_k_for_sum:
            answers.append(min_k_for_sum[n][1])
        else:
            found = None
            # Try k=5, 6, ...
            # Max k is 30, but likely much smaller.
            for k in range(5, 31):
                if time.time() - start_time > DEADLINE:
                    break
                r = find_antichain_sum(n, k)
                if r:
                    found = r
                    break
            if found:
                answers.append(found)
            else:
                # Fallback: should not happen if all targets are representable
                # But to ensure validity, we might need a more robust fallback or just fail.
                # The problem states "All targets require valid representations".
                # If we fail, we return an invalid answer or empty?
                # Let's try to find ANY antichain sum, even if not minimal, using a different strategy if needed.
                # But for now, if k up to 30 fails, it's a problem.
                # Given the density of smooth numbers, k should be small.
                answers.append([]) # Placeholder, hopefully not used
                
    return answers
