import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Base values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Generate all words as index arrays
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)

    # Precompute ball arrays as numpy
    ball_arrays = [np.array(bi, dtype=np.int32) for bi in ball_indices]

    # Precompute ball_mask: ball_mask[i, j] is True if j is in ball of i
    ball_mask = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mask[i, ball_arrays[i]] = True

    # Precompute coverers: coverers[point] = list of codewords that cover point
    coverers = [[] for _ in range(total)]
    for cw_idx in range(total):
        for pt in ball_indices[cw_idx]:
            coverers[pt].append(cw_idx)

    def greedy_cover_bitset(order=None):
        """Greedy set cover using bitsets for speed."""
        if order is None:
            order = list(range(total))
        
        # Use bitsets for coverage tracking
        ball_bitsets = [0] * total
        for i in range(total):
            mask = 0
            for j in ball_indices[i]:
                mask |= (1 << j)
            ball_bitsets[i] = mask
        
        all_mask = (1 << total) - 1
        current_cover_mask = 0
        code = []
        is_chosen = np.zeros(total, dtype=bool)
        
        while current_cover_mask != all_mask:
            if time.time() > deadline - 50:
                break
                
            best_c = -1
            best_new = -1
            
            for c in order:
                if is_chosen[c]:
                    continue
                new_bits = ball_bitsets[c] & ~current_cover_mask
                if new_bits:
                    count = new_bits.bit_count()
                    if count > best_new:
                        best_new = count
                        best_c = c
                        if best_new == ball_size:
                            break
            
            if best_c == -1:
                # Fallback: pick any uncovered
                for c in range(total):
                    if not is_chosen[c] and (ball_bitsets[c] & ~current_cover_mask):
                        best_c = c
                        break
                if best_c == -1:
                    break
            
            code.append(best_c)
            is_chosen[best_c] = True
            current_cover_mask |= ball_bitsets[best_c]
        
        return code

    def delete_redundant(code_set, cnt):
        """Remove redundant codewords where all points in ball have count >= 2."""
        changed = False
        for c in list(code_set):
            if np.all(cnt[ball_arrays[c]] >= 2):
                code_set.remove(c)
                cnt[ball_arrays[c]] -= 1
                changed = True
        return changed

    # === Initial solution: multiple greedy strategies ===
    best_code = None
    best_size = total + 1
    best_cnt = None

    strategies = ["natural", "random1", "random2", "random3", "random4", "random5"]

    for s in strategies:
        if time.time() > deadline - 45:
            break
        if s == "natural":
            order = list(range(total))
        else:
            order = list(range(total))
            random.shuffle(order)
        
        code = greedy_cover_bitset(order)
        code_set = set(code)
        
        # Build counts
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_arrays[c]] += 1
        
        # Delete redundant
        while delete_redundant(code_set, cnt):
            pass
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            best_cnt = cnt.copy()

    if best_code is None:
        best_code = list(range(0, total, max(1, total // 186)))
        best_size = len(best_code)
        best_cnt = np.zeros(total, dtype=np.int32)
        for c in best_code:
            best_cnt[ball_arrays[c]] += 1

    # === Local Search with Efficient Swap Checks ===
    
    def compute_unique_points_fast(code_set, cnt):
        """Fast version: only compute for codewords that might have unique points."""
        unique = {}
        for c in code_set:
            ball = ball_arrays[c]
            if np.any(cnt[ball] == 1):
                mask = (cnt[ball] == 1)
                unique[c] = ball[mask]
            else:
                unique[c] = np.array([], dtype=np.int32)
        return unique

    def is_valid_swap(c_remove, c_add, unique_points):
        """Check if swapping c_remove for c_add maintains coverage."""
        u = unique_points[c_remove]
        if len(u) == 0:
            return True
        return np.all(ball_mask[c_add, u])

    def apply_swap(c_remove, c_add, code_set, cnt):
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt[ball_arrays[c_remove]] -= 1
        cnt[ball_arrays[c_add]] += 1

    def is_valid_2for1(c1, c2, c_add, unique_points):
        """Check if removing c1, c2 and adding c_add maintains coverage."""
        u1 = unique_points[c1]
        u2 = unique_points[c2]
        if len(u1) == 0 and len(u2) == 0:
            return True
        if len(u1) > 0:
            if not np.all(ball_mask[c_add, u1]):
                return False
        if len(u2) > 0:
            if not np.all(ball_mask[c_add, u2]):
                return False
        return True

    def apply_2for1(c1, c2, c_add, code_set, cnt):
        code_set.remove(c1)
        code_set.remove(c2)
        code_set.add(c_add)
        cnt[ball_arrays[c1]] -= 1
        cnt[ball_arrays[c2]] -= 1
        cnt[ball_arrays[c_add]] += 1

    def is_valid_3for2(c1, c2, c3, c_add1, c_add2, unique_points):
        """Check if removing c1,c2,c3 and adding c_add1, c_add2 maintains coverage."""
        u1 = unique_points[c1]
        u2 = unique_points[c2]
        u3 = unique_points[c3]
        
        if len(u1) > 0:
            if not np.all(ball_mask[c_add1, u1] | ball_mask[c_add2, u1]):
                return False
        if len(u2) > 0:
            if not np.all(ball_mask[c_add1, u2] | ball_mask[c_add2, u2]):
                return False
        if len(u3) > 0:
            if not np.all(ball_mask[c_add1, u3] | ball_mask[c_add2, u3]):
                return False
        return True

    def apply_3for2(c1, c2, c3, c_add1, c_add2, code_set, cnt):
        code_set.remove(c1)
        code_set.remove(c2)
        code_set.remove(c3)
        code_set.add(c_add1)
        code_set.add(c_add2)
        cnt[ball_arrays[c1]] -= 1
        cnt[ball_arrays[c2]] -= 1
        cnt[ball_arrays[c3]] -= 1
        cnt[ball_arrays[c_add1]] += 1
        cnt[ball_arrays[c_add2]] += 1

    def find_best_candidate_for_unique(unique_pts, exclude_set):
        """Find candidate that covers all unique_pts, preferring low redundancy."""
        if len(unique_pts) == 0:
            # Any non-excluded word works
            candidates = np.ones(total, dtype=bool)
            for c in exclude_set:
                candidates[c] = False
            valid = np.where(candidates)[0]
            if len(valid) > 0:
                return random.choice(valid)
            return -1
        
        candidates = np.ones(total, dtype=bool)
        for pt in unique_pts:
            candidates &= ball_mask[pt]
        
        for c in exclude_set:
            candidates[c] = False
        
        valid_candidates = np.where(candidates)[0]
        if len(valid_candidates) == 0:
            return -1
        
        # Prefer candidates with lower average coverage count (less redundancy)
        if len(valid_candidates) <= 20:
            best_w = -1
            best_cost = float('inf')
            for w in valid_candidates:
                cost = np.mean(cnt[ball_arrays[w]])
                if cost < best_cost:
                    best_cost = cost
                    best_w = w
            return best_w
        else:
            # Sample to find good candidate
            sample = random.sample(list(valid_candidates), min(20, len(valid_candidates)))
            best_w = -1
            best_cost = float('inf')
            for w in sample:
                cost = np.mean(cnt[ball_arrays[w]])
                if cost < best_cost:
                    best_cost = cost
                    best_w = w
            return best_w

    current_code = set(best_code)
    cnt = best_cnt.copy()
    best_size = len(current_code)
    best_code_list = list(current_code)
    
    no_improve_streak = 0
    max_streak = 800
    
    while time.time() < deadline - 15.0 and no_improve_streak < max_streak:
        improved = False
        
        # 1. Compute unique points
        unique_points = compute_unique_points_fast(current_code, cnt)
        
        # 2. Try Deletions
        deletion_found = False
        for c in list(current_code):
            if len(unique_points[c]) == 0:
                current_code.remove(c)
                cnt[ball_arrays[c]] -= 1
                improved = True
                deletion_found = True
                break
        
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            if deletion_found:
                continue

        # 3. Try 1-for-1 swaps with better heuristic
        code_list = list(current_code)
        random.shuffle(code_list)
        
        swap_found = False
        for c in code_list:
            if time.time() > deadline - 15.0:
                break
            if c not in current_code:
                continue
            
            u = unique_points[c]
            if len(u) == 0:
                continue
            
            # Find candidate that covers all unique points
            w = find_best_candidate_for_unique(u, current_code)
            
            if w >= 0:
                apply_swap(c, w, current_code, cnt)
                improved = True
                swap_found = True
                break
        
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # 4. Try 2-for-1 swaps (remove 2, add 1) - reduces size by 1
        if len(current_code) >= 2 and time.time() < deadline - 18.0:
            found_2for1 = False
            code_list2 = list(current_code)
            random.shuffle(code_list2)
            
            for i in range(min(40, len(code_list2))):
                if time.time() > deadline - 18.0:
                    break
                c1 = code_list2[i]
                if c1 not in current_code:
                    continue
                
                u1 = unique_points[c1]
                
                for j in range(min(20, len(code_list2))):
                    if j == i:
                        continue
                    c2 = code_list2[j]
                    if c2 not in current_code or c2 == c1:
                        continue
                    
                    u2 = unique_points[c2]
                    
                    # Combine unique points from both
                    if len(u1) == 0 and len(u2) == 0:
                        # Both can be deleted
                        current_code.remove(c1)
                        current_code.remove(c2)
                        cnt[ball_arrays[c1]] -= 1
                        cnt[ball_arrays[c2]] -= 1
                        improved = True
                        found_2for1 = True
                        break
                    
                    all_unique = np.concatenate([u1, u2])
                    all_unique = np.unique(all_unique)
                    
                    # Find w such that all all_unique ⊆ ball(w)
                    w = find_best_candidate_for_unique(all_unique, current_code)
                    
                    if w >= 0:
                        apply_2for1(c1, c2, w, current_code, cnt)
                        improved = True
                        found_2for1 = True
                        break
                
                if found_2for1:
                    break
            
            if improved:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)
                no_improve_streak = 0
                continue

        # 5. Try 3-for-2 swaps (remove 3, add 2) - reduces size by 1
        if len(current_code) >= 3 and time.time() < deadline - 20.0:
            found_3for2 = False
            code_list3 = list(current_code)
            random.shuffle(code_list3)
            
            for i in range(min(25, len(code_list3))):
                if time.time() > deadline - 20.0:
                    break
                c1 = code_list3[i]
                if c1 not in current_code:
                    continue
                
                u1 = unique_points[c1]
                
                for j in range(min(15, len(code_list3))):
                    if j == i:
                        continue
                    c2 = code_list3[j]
                    if c2 not in current_code or c2 == c1:
                        continue
                    
                    u2 = unique_points[c2]
                    
                    for k in range(min(8, len(code_list3))):
                        if k == i or k == j:
                            continue
                        c3 = code_list3[k]
                        if c3 not in current_code or c3 == c1 or c3 == c2:
                            continue
                        
                        u3 = unique_points[c3]
                        
                        # Find w1, w2 such that unique points are covered
                        all_unique = np.concatenate([u1, u2, u3])
                        all_unique = np.unique(all_unique)
                        
                        if len(all_unique) == 0:
                            # All can be deleted
                            current_code.remove(c1)
                            current_code.remove(c2)
                            current_code.remove(c3)
                            cnt[ball_arrays[c1]] -= 1
                            cnt[ball_arrays[c2]] -= 1
                            cnt[ball_arrays[c3]] -= 1
                            improved = True
                            found_3for2 = True
                            break
                        
                        # Try to find two words that together cover all unique points
                        # Greedy: find w1 that covers most unique points
                        best_w1 = -1
                        best_w1_cov = -1
                        
                        # Find candidates that cover at least some unique points
                        candidates1 = np.ones(total, dtype=bool)
                        for pt in all_unique:
                            candidates1 &= ball_mask[pt]
                        for c3b in current_code:
                            candidates1[c3b] = False
                        
                        valid_candidates1 = np.where(candidates1)[0]
                        if len(valid_candidates1) > 0:
                            # Check top candidates for coverage
                            check_list = valid_candidates1[:30] if len(valid_candidates1) > 30 else valid_candidates1
                            for w1 in check_list:
                                cov = np.sum(ball_mask[w1, all_unique])
                                if cov > best_w1_cov:
                                    best_w1_cov = cov
                                    best_w1 = w1
                        
                        if best_w1 < 0:
                            continue
                        
                        w1 = best_w1
                        covered_by_w1 = ball_mask[w1]
                        remaining_unique_after_w1 = all_unique[~covered_by_w1[all_unique]]
                        
                        if len(remaining_unique_after_w1) == 0:
                            # w1 covers everything, need another distinct word
                            w2 = find_best_candidate_for_unique(np.array([], dtype=np.int32), current_code | {w1})
                            if w2 >= 0:
                                apply_3for2(c1, c2, c3, w1, w2, current_code, cnt)
                                improved = True
                                found_3for2 = True
                                break
                            continue
                        
                        # Find w2 that covers remaining
                        w2 = find_best_candidate_for_unique(remaining_unique_after_w1, current_code | {w1})
                        
                        if w2 >= 0:
                            apply_3for2(c1, c2, c3, w1, w2, current_code, cnt)
                            improved = True
                            found_3for2 = True
                            break
                        
                    if found_3for2:
                        break
                
                if found_3for2:
                    break
            
            if improved:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)
                no_improve_streak = 0
                continue

        # 6. Try 2-for-2 swaps (remove 2, add 2) - maintains size but may enable future reductions
        if len(current_code) >= 2 and time.time() < deadline - 22.0:
            found_2for2 = False
            code_list_22 = list(current_code)
            random.shuffle(code_list_22)
            
            for i in range(min(15, len(code_list_22))):
                if time.time() > deadline - 22.0:
                    break
                c1 = code_list_22[i]
                if c1 not in current_code:
                    continue
                
                u1 = unique_points[c1]
                
                for j in range(i + 1, min(i + 12, len(code_list_22))):
                    if time.time() > deadline - 22.0:
                        break
                    c2 = code_list_22[j]
                    if c2 not in current_code:
                        continue
                    
                    u2 = unique_points[c2]
                    
                    if len(u1) == 0 and len(u2) == 0:
                        continue  # Both deletable, handled above
                    
                    all_unique = np.concatenate([u1, u2]) if len(u1) > 0 and len(u2) > 0 else (u1 if len(u1) > 0 else u2)
                    if len(all_unique) == 0:
                        continue
                    all_unique = np.unique(all_unique)
                    
                    # Find w1 that covers most unique points
                    best_w1 = -1
                    best_w1_cov = -1
                    candidates_w1 = np.ones(total, dtype=bool)
                    for pt in all_unique:
                        candidates_w1 &= ball_mask[pt]
                    for c3 in current_code:
                        candidates_w1[c3] = False
                    
                    valid_w1 = np.where(candidates_w1)[0]
                    if len(valid_w1) > 0:
                        check_list = valid_w1[:20] if len(valid_w1) > 20 else valid_w1
                        for w1 in check_list:
                            cov = np.sum(ball_mask[w1, all_unique])
                            if cov > best_w1_cov:
                                best_w1_cov = cov
                                best_w1 = w1
                    
                    if best_w1 < 0:
                        continue
                    
                    # Find w2 that covers remaining unique points
                    remaining = all_unique[~ball_mask[best_w1, all_unique]]
                    if len(remaining) == 0:
                        # w1 covers everything, need another distinct word
                        w2 = find_best_candidate_for_unique(np.array([], dtype=np.int32), current_code | {best_w1})
                        if w2 >= 0:
                            current_code.remove(c1)
                            current_code.remove(c2)
                            current_code.add(best_w1)
                            current_code.add(w2)
                            cnt[ball_arrays[c1]] -= 1
                            cnt[ball_arrays[c2]] -= 1
                            cnt[ball_arrays[best_w1]] += 1
                            cnt[ball_arrays[w2]] += 1
                            improved = True
                            found_2for2 = True
                            break
                    else:
                        w2 = find_best_candidate_for_unique(remaining, current_code | {best_w1})
                        if w2 >= 0:
                            current_code.remove(c1)
                            current_code.remove(c2)
                            current_code.add(best_w1)
                            current_code.add(w2)
                            cnt[ball_arrays[c1]] -= 1
                            cnt[ball_arrays[c2]] -= 1
                            cnt[ball_arrays[best_w1]] += 1
                            cnt[ball_arrays[w2]] += 1
                            improved = True
                            found_2for2 = True
                            break
                
                if found_2for2:
                    break
            
            if improved:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)
                no_improve_streak = 0
                continue

        if not improved:
            no_improve_streak += 1

    # Final: one more delete pass
    while delete_redundant(current_code, cnt):
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code_list = list(current_code)

    # Final polish: try a few more rounds of deletion + 1-for-1 swaps
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        # Delete pass
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for c in list(current_code):
                if np.all(cnt[ball_arrays[c]] >= 2):
                    current_code.remove(c)
                    cnt[ball_arrays[c]] -= 1
                    changed = True
            if changed:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)
        
        # Quick 1-for-1 pass
        unique_points = compute_unique_points_fast(current_code, cnt)
        code_list_final = list(current_code)
        random.shuffle(code_list_final)
        for c in code_list_final[:60]:
            if time.time() > deadline - 5.0:
                break
            if c not in current_code:
                continue
            u = unique_points[c]
            if len(u) == 0:
                continue
            w = find_best_candidate_for_unique(u, current_code)
            if w >= 0:
                apply_swap(c, w, current_code, cnt)
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)

    # Convert back to list of lists
    result = []
    for idx in best_code_list:
        w = []
        temp_idx = idx
        for i in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)

    return result
