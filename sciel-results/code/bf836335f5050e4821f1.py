import time

_START = time.time()
DEADLINE = 85

def construct():
    t0 = time.time()
    
    # Generate all 2^a * 3^b * 5^c smooth numbers <= 128
    smooth_set = set()
    for a in range(0, 8):
        p2 = 2 ** a
        if p2 > 128: break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > 128: break
            for c in range(0, 5):
                p5 = 5 ** c
                v = p2 * p3 * p5
                if v > 128: break
                smooth_set.add(v)
    smooth = sorted(smooth_set)
    n_smooth = len(smooth)
    
    # Index mapping
    val_to_idx = {v: i for i, v in enumerate(smooth)}
    
    # Precompute divisibility: divs[i] = set of j where smooth[j] divides smooth[i]
    # For antichain check: when adding smooth[i] to a set, no existing element should divide it
    # i.e., for all j in current set, smooth[i] % smooth[j] != 0
    # Equivalently: j not in "divisors of i"
    
    # Precompute for each i, the set of j < i such that smooth[j] divides smooth[i]
    # This helps in checking: if we build in increasing index order, we just need
    # that none of the previously chosen indices divide the new one.
    
    # Actually, let's precompute a compatibility matrix:
    # compat[i][j] = True if smooth[i] and smooth[j] are compatible (neither divides the other)
    # But that's O(n^2) which is fine for n~50
    
    # Better: for each i, precompute which smaller indices divide i
    divides_smaller = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(i):
            if vi % smooth[j] == 0:
                divides_smaller[i].add(j)
    
    # min_terms[n] = minimum number of terms for target n
    min_terms = [100] * 129
    reps = [None] * 129
    
    # Helper to record
    def record(target, terms):
        if terms and target > 0 and target <= 128:
            if len(terms) < min_terms[target]:
                min_terms[target] = len(terms)
                reps[target] = list(terms)
    
    # Phase 1: Single terms
    for v in smooth:
        if v <= 128:
            record(v, [v])
    
    # Phase 2: Two-term antichains
    # For i < j, check that smooth[i] does not divide smooth[j]
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(i + 1, n_smooth):
            vj = smooth[j]
            s = vi + vj
            if s > 128:
                break
            # Check: vi does not divide vj
            if vj % vi != 0:
                record(s, [vi, vj])
    
    # Phase 3: Three-term antichains
    # For i < j < k, check pairwise: no one divides another
    # Since i<j<k, we need: smooth[j]%smooth[i]!=0, smooth[k]%smooth[i]!=0, smooth[k]%smooth[j]!=0
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(i + 1, n_smooth):
            vj = smooth[j]
            if vj % vi == 0:
                continue
            for k in range(j + 1, n_smooth):
                vk = smooth[k]
                s = vi + vj + vk
                if s > 128:
                    break
                if vk % vi == 0 or vk % vj == 0:
                    continue
                record(s, [vi, vj, vk])
    
    # Phase 4: Four-term antichains
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(i + 1, n_smooth):
            vj = smooth[j]
            if vj % vi == 0:
                continue
            for k in range(j + 1, n_smooth):
                vk = smooth[k]
                if vk % vi == 0 or vk % vj == 0:
                    continue
                for l in range(k + 1, n_smooth):
                    vl = smooth[l]
                    s = vi + vj + vk + vl
                    if s > 128:
                        break
                    if vl % vi == 0 or vl % vj == 0 or vl % vk == 0:
                        continue
                    record(s, [vi, vj, vk, vl])
    
    # Phase 5: For any targets still not covered, do a targeted DFS
    missing = [n for n in range(1, 129) if reps[n] is None]
    
    if missing:
        # DFS for each missing target, trying to find a representation with <= current best
        # Use a general DFS with pruning
        def dfs_target(target, start_idx, current_sum, current_list, max_terms):
            if current_sum > target:
                return False
            if len(current_list) >= max_terms:
                return False
            if current_sum == target:
                if current_list:
                    record(target, current_list)
                    return True
                return False
            
            if time.time() - t0 > DEADLINE - 2:
                return False
            
            for i in range(start_idx, n_smooth):
                new_val = smooth[i]
                new_sum = current_sum + new_val
                if new_sum > target:
                    break
                # Check divisibility
                valid = True
                for val in current_list:
                    if new_val % val == 0:
                        valid = False
                        break
                if not valid:
                    continue
                current_list.append(new_val)
                if dfs_target(target, i + 1, new_sum, current_list, max_terms):
                    current_list.pop()
                    return True
                current_list.pop()
            return False
        
        for target in missing:
            # Try increasing term counts
            found = False
            for max_t in range(5, 31):
                if time.time() - t0 > DEADLINE - 2:
                    break
                if dfs_target(target, 0, 0, [], max_t):
                    found = True
                    break
            if not found and reps[target] is None:
                # Last resort: if target is smooth, use it
                if target in smooth_set:
                    record(target, [target])
    
    # Build answers
    answers = []
    for n in range(1, 129):
        if reps[n] is not None:
            answers.append(reps[n])
        else:
            # Should not happen, but fallback
            if n in smooth_set:
                answers.append([n])
            else:
                # Emergency: find any valid representation via slow DFS
                def slow_find(target):
                    def dfs2(start_idx, cur_sum, cur_list):
                        if cur_sum > target:
                            return None
                        if len(cur_list) >= 30:
                            return None
                        if cur_sum == target:
                            return list(cur_list) if cur_list else None
                        if time.time() - t0 > DEADLINE - 1:
                            return None
                        for i in range(start_idx, n_smooth):
                            nv = smooth[i]
                            ns = cur_sum + nv
                            if ns > target:
                                break
                            ok = True
                            for v in cur_list:
                                if nv % v == 0:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            cur_list.append(nv)
                            res = dfs2(i + 1, ns, cur_list)
                            cur_list.pop()
                            if res:
                                return res
                        return None
                    return dfs2(0, 0, [])
                res = slow_find(n)
                if res:
                    answers.append(res)
                else:
                    # Absolute last resort - should never reach here
                    answers.append([1])
    
    return answers
