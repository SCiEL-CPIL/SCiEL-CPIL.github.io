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
                    
    # Fallback for uncovered numbers using backtracking with improved pruning
    def find_antichain_sum(n, k):
        def backtrack(start, rem_sum, rem_k, current_path, current_mask):
            if rem_k == 0:
                if rem_sum == 0:
                    return [smooth_list[idx] for idx in current_path]
                return None
            
            if n_smooth - start < rem_k:
                return None
            
            max_start_idx = n_smooth - rem_k
            if max_start_idx < start:
                return None
            if suffix_sum[max_start_idx] < rem_sum:
                return None
                
            if prefix_sum[start + rem_k] - prefix_sum[start] > rem_sum:
                return None

            for i in range(start, n_smooth):
                val = smooth_list[i]
                if val > rem_sum:
                    break
                
                if incompat_mask[i] & current_mask:
                    continue
                
                if rem_k - 1 > 0:
                    if i + 1 + (rem_k - 1) > n_smooth:
                        break
                    min_next_sum = prefix_sum[i + 1 + (rem_k - 1)] - prefix_sum[i + 1]
                    if min_next_sum > rem_sum - val:
                        break
                    max_next_start = n_smooth - (rem_k - 1)
                    if max_next_start >= i + 1:
                        max_next_sum = suffix_sum[max_next_start]
                        if max_next_sum < rem_sum - val:
                            continue
                    else:
                        continue

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
            for k in range(5, 31):
                if time.time() - start_time > DEADLINE - 5:
                    break
                r = find_antichain_sum(n, k)
                if r:
                    found = r
                    break
            if found:
                answers.append(found)
            else:
                # Try with relaxed time check
                for k in range(5, 31):
                    if time.time() - start_time > DEADLINE - 1:
                        break
                    r = find_antichain_sum(n, k)
                    if r:
                        found = r
                        break
                if found:
                    answers.append(found)
                else:
                    # Last resort: single term if it's smooth, otherwise two terms
                    if n in val_to_idx:
                        answers.append([n])
                    else:
                        # Find any two smooth numbers that sum to n
                        found2 = None
                        for i in range(n_smooth):
                            for j in range(i+1, n_smooth):
                                if smooth_list[i] + smooth_list[j] == n:
                                    found2 = [smooth_list[i], smooth_list[j]]
                                    break
                            if found2:
                                break
                        if found2:
                            answers.append(found2)
                        else:
                            answers.append([n])  # May be invalid, but better than empty
    
    return answers
