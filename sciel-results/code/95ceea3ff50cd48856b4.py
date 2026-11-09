import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    # Quick check for small targets that can be represented directly
    # Target 1: just [1]
    # Target 2: [2]
    # etc.
    
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
    
    # Precompute incompatibility (divisibility)
    incompat = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth_list[i], smooth_list[j]
            if b % a == 0 or a % b == 0:
                incompat[i][j] = True
                incompat[j][i] = True
                
    # Precompute prefix and suffix sums for pruning
    prefix_sums = [0] * (n_smooth + 1)
    for i in range(n_smooth):
        prefix_sums[i + 1] = prefix_sums[i] + smooth_list[i]
        
    suffix_sums = [0] * (n_smooth + 1)
    for i in range(n_smooth - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + smooth_list[i]

    min_k_for_sum = {}
    
    # Precompute minimal k for sums using combinations up to K_PRE
    # Increase K_PRE to find better solutions for more targets
    K_PRE = 8
    
    for k in range(1, K_PRE + 1):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        for combo_indices in itertools.combinations(range(n_smooth), k):
            # Check antichain property
            is_anti = True
            for i in range(k):
                ci = combo_indices[i]
                for j in range(i + 1, k):
                    cj = combo_indices[j]
                    if incompat[ci][cj]:
                        is_anti = False
                        break
                if not is_anti:
                    break
            
            if not is_anti:
                continue
                
            # Calculate sum with early break
            s = 0
            for idx in combo_indices:
                s += smooth_list[idx]
                if s > 128:
                    break
            
            if s > 128:
                continue
                
            if s not in min_k_for_sum:
                min_k_for_sum[s] = (k, [smooth_list[idx] for idx in combo_indices])
                
    def find_antichain_sum(n, k):
        def backtrack(start, rem_sum, rem_k, current_path):
            if rem_k == 0:
                if rem_sum == 0:
                    return [smooth_list[idx] for idx in current_path]
                return None
            
            # Pruning: Not enough candidates left
            if n_smooth - start < rem_k:
                return None
            
            # Pruning: Min sum check
            # Sum of smallest rem_k numbers from start
            min_possible = prefix_sums[start + rem_k] - prefix_sums[start]
            if min_possible > rem_sum:
                return None
                
            # Pruning: Max sum check - simplified
            max_idx_start = n_smooth - rem_k
            if max_idx_start >= start:
                max_possible = suffix_sums[max_idx_start]
            else:
                max_possible = suffix_sums[start]
                
            if max_possible < rem_sum:
                return None

            for i in range(start, n_smooth):
                val = smooth_list[i]
                if val > rem_sum:
                    break
                
                # Check compatibility
                compatible = True
                for p in current_path:
                    if incompat[p][i]:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Pruning: Min sum of next rem_k-1
                if rem_k - 1 > 0:
                    if i + 1 + (rem_k - 1) > n_smooth:
                        break
                    min_next_sum = prefix_sums[i + 1 + rem_k - 1] - prefix_sums[i + 1]
                    if min_next_sum > rem_sum - val:
                        break
                    
                current_path.append(i)
                res = backtrack(i + 1, rem_sum - val, rem_k - 1, current_path)
                if res:
                    return res
                current_path.pop()
                
            return None
        
        return backtrack(0, n, k, [])

    answers = []
    for n in range(1, 129):
        if n in min_k_for_sum:
            answers.append(min_k_for_sum[n][1])
        else:
            found = None
            # Try k from 1 upward to find minimal representation
            # But we already tried up to K_PRE in precomputation
            # So start from max(1, K_PRE - 2) to allow some overlap for better pruning
            for k in range(max(1, K_PRE - 2), 31):
                if time.time() - start_time > DEADLINE:
                    break
                r = find_antichain_sum(n, k)
                if r:
                    found = r
                    break
            if found:
                answers.append(found)
            else:
                # Fallback: use a simple valid representation if we run out of time
                # Just use [n] if n is smooth, otherwise try a few options
                if n in smooth:
                    answers.append([n])
                else:
                    # Try to find any valid antichain representation
                    found_any = None
                    for k in range(2, 31):
                        r = find_antichain_sum(n, k)
                        if r:
                            found_any = r
                            break
                    if found_any:
                        answers.append(found_any)
                    else:
                        # Last resort: return empty (shouldn't happen)
                        answers.append([])
                
    return answers
