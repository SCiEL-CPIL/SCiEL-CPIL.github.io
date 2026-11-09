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
    
    # Precompute incompatibility (divisibility)
    # incompat[i][j] is True if smooth_list[i] and smooth_list[j] cannot be together
    # i.e., one divides the other.
    incompat = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_list[i], smooth_list[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    incompat[i][j] = True
                    incompat[j][i] = True
    
    min_k_for_sum = {}
    
    # k=1
    for i in range(n_smooth):
        s = smooth_list[i]
        if s <= 128 and s not in min_k_for_sum:
            min_k_for_sum[s] = (1, [s])
            
    # k=2
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if incompat[i][j]:
                continue
            s = smooth_list[i] + smooth_list[j]
            if s > 128:
                break # Since smooth_list is sorted, further j will be larger
            if s not in min_k_for_sum:
                min_k_for_sum[s] = (2, [smooth_list[i], smooth_list[j]])
                
    # k=3
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if incompat[i][j]:
                continue
            s2 = smooth_list[i] + smooth_list[j]
            if s2 > 128:
                break
            for k in range(j+1, n_smooth):
                if incompat[i][k] or incompat[j][k]:
                    continue
                s3 = s2 + smooth_list[k]
                if s3 > 128:
                    break
                if s3 not in min_k_for_sum:
                    min_k_for_sum[s3] = (3, [smooth_list[i], smooth_list[j], smooth_list[k]])
                    
    # k=4
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if incompat[i][j]:
                continue
            s2 = smooth_list[i] + smooth_list[j]
            if s2 > 128:
                break
            for k in range(j+1, n_smooth):
                if incompat[i][k] or incompat[j][k]:
                    continue
                s3 = s2 + smooth_list[k]
                if s3 > 128:
                    break
                for l in range(k+1, n_smooth):
                    if incompat[i][l] or incompat[j][l] or incompat[k][l]:
                        continue
                    s4 = s3 + smooth_list[l]
                    if s4 > 128:
                        break
                    if s4 not in min_k_for_sum:
                        min_k_for_sum[s4] = (4, [smooth_list[i], smooth_list[j], smooth_list[k], smooth_list[l]])
                    
    # Fallback for uncovered numbers using backtracking
    # We search for minimal k starting from 5
    # To make it fast, we use a recursive helper that checks antichain constraint
    def find_antichain_sum(n, k):
        # Candidates are indices into smooth_list
        # We want to pick k distinct indices i1 < i2 < ... < ik
        # such that sum(smooth_list[ij]) == n and no pair is incompatible.
        
        # Optimization: Precompute compatible sets?
        # Or just check on the fly.
        
        # Heuristic: Sort candidates? smooth_list is already sorted.
        # We can prune if min_possible_sum > n or max_possible_sum < n?
        # Min sum of k numbers from remaining candidates.
        
        def backtrack(start, rem_sum, rem_k, current_path):
            if rem_k == 0:
                if rem_sum == 0:
                    return [smooth_list[idx] for idx in current_path]
                return None
            
            # Pruning:
            # 1. Not enough candidates left
            if len(smooth_list) - start < rem_k:
                return None
            
            # 2. Even taking the largest rem_k numbers, sum < rem_sum?
            #    Actually, we want sum == rem_sum.
            #    Max possible sum from remaining k numbers:
            #    Take the largest rem_k numbers from smooth_list[start:].
            #    If max_sum < rem_sum, return None.
            #    Min possible sum:
            #    Take the smallest rem_k numbers.
            #    If min_sum > rem_sum, return None.
            
            # Calculate min_sum: sum of smooth_list[start : start+rem_k]
            # Calculate max_sum: sum of smooth_list[-rem_k:] (but must be >= start)
            # Note: smooth_list is global. We are picking from start to end.
            
            # Efficient check:
            # If rem_sum > sum(smooth_list[-rem_k:]) and start <= n_smooth - rem_k:
            #    Actually, the available numbers are smooth_list[start:].
            #    The largest rem_k numbers in smooth_list[start:] are smooth_list[n_smooth-rem_k:]
            #    provided n_smooth-rem_k >= start.
            
            if rem_k > 0:
                # Min sum
                min_idx = start
                max_idx = n_smooth - rem_k
                if max_idx < min_idx:
                    return None
                
                # If the smallest possible sum (first rem_k from start) is > rem_sum
                # We can break early in the loop below, but this is a global check.
                
                # Let's just rely on the loop variable bounds.
                pass
            
            # Iterate candidates for the next element
            # We can restrict the upper bound of the current element.
            # If we pick smooth_list[i], the remaining sum is rem_sum - smooth_list[i].
            # The remaining rem_k-1 numbers must be >= smooth_list[i] (to keep distinct and sorted? No, just distinct indices > i).
            # Actually, indices must be increasing.
            
            # Upper bound for i:
            # smooth_list[i] must be <= rem_sum - (min sum of rem_k-1 numbers from i+1)
            # Min sum of rem_k-1 numbers from i+1 is sum(smooth_list[i+1 : i+1+rem_k-1])
            
            # Simpler upper bound:
            # smooth_list[i] <= rem_sum - (rem_k-1) * 1 (since min smooth number is 1? No, min is 1, but must be > smooth_list[i]? No, distinct indices, but values can be small).
            # Actually, the smallest available number after i is smooth_list[i+1].
            # So smooth_list[i] + sum(smallest rem_k-1 numbers > i) <= rem_sum.
            
            end = n_smooth
            # We can break the loop if smooth_list[i] is too large.
            # If smooth_list[i] > rem_sum, break.
            
            for i in range(start, end):
                val = smooth_list[i]
                if val > rem_sum:
                    break
                
                # Check compatibility with current_path
                compatible = True
                for p in current_path:
                    if incompat[p][i]:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Pruning: Can we form rem_sum - val with rem_k-1 numbers from i+1?
                # Min sum of rem_k-1 numbers from i+1:
                if rem_k - 1 > 0:
                    # If i+1 + (rem_k-1) > n_smooth, not enough numbers
                    if i + 1 + (rem_k - 1) > n_smooth:
                        break # Since i is increasing, this will only get worse
                    min_next_sum = sum(smooth_list[i+1 : i+1+rem_k-1])
                    if min_next_sum > rem_sum - val:
                        break # val is too small? No, if min_next_sum is too big, then even the smallest remaining numbers are too big?
                        # Wait. If min_next_sum > rem_sum - val, then we need a smaller val?
                        # No, val is fixed. The remaining numbers must sum to rem_sum - val.
                        # The minimum sum of the next rem_k-1 numbers is min_next_sum.
                        # If min_next_sum > rem_sum - val, then it's impossible to pick rem_k-1 numbers that sum to rem_sum - val?
                        # Yes, because any other set of rem_k-1 numbers will have a sum >= min_next_sum.
                        # So this branch fails.
                        # But does increasing i help?
                        # If we increase i, val increases, so rem_sum - val decreases.
                        # min_next_sum also increases (since we start from a larger index).
                        # It's not monotonic in a simple way.
                        # However, if val is too small, rem_sum - val is large.
                        # If val is too large, rem_sum - val is small.
                        # The condition min_next_sum > rem_sum - val means the "floor" of the next sums is higher than the target.
                        # This happens if the remaining numbers are all too large.
                        # As i increases, the remaining numbers get larger (min_next_sum increases) and the target (rem_sum - val) decreases.
                        # So the gap widens. Thus, if this condition holds, it will hold for all larger i.
                        # So we can break.
                        break
                
                # Max sum check?
                # Max sum of rem_k-1 numbers from i+1 is sum(smooth_list[n_smooth-rem_k+1:])
                # If max_next_sum < rem_sum - val, then even the largest numbers are too small.
                # As i increases, val increases, rem_sum - val decreases.
                # max_next_sum stays constant (or decreases if we run out of numbers? No, we just pick the largest available).
                # Actually, the available set shrinks. The largest available numbers are always the global largest, as long as they are > i.
                # So max_next_sum is roughly constant.
                # If max_next_sum < rem_sum - val, it's impossible.
                # As i increases, rem_sum - val decreases, so this condition becomes easier to satisfy (less likely to fail).
                # So we can't break based on this, but we can skip this i.
                
                current_path.append(i)
                res = backtrack(i+1, rem_sum - val, rem_k - 1, current_path)
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
