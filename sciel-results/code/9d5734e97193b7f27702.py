import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

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

    # Precompute neighbor offsets: for each coordinate, the delta to add/subtract
    # neighbor_offsets[i] = list of (delta, coord) pairs for all neighbors
    # We'll precompute a flat array of all neighbors for each word
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

    # Precompute reverse adjacency: for each point p, which codewords cover it
    # This is the transpose of ball_mask
    # coverers[p] = list of codewords whose ball contains p
    coverers = [np.where(ball_mask[:, p])[0] for p in range(total)]

    def greedy_cover(order=None):
        """Greedy set cover: repeatedly pick the codeword covering most uncovered points."""
        if order is None:
            order = list(range(total))
        covered = np.zeros(total, dtype=bool)
        code = []
        remaining = np.ones(total, dtype=bool)
        remaining_count = total
        
        while remaining_count > 0:
            best_c = -1
            best_new = -1
            for c in order:
                new_cov = np.count_nonzero(remaining & ball_mask[c])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            code.append(best_c)
            remaining[ball_mask[best_c]] = False
            remaining_count -= best_new
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

    strategies = ["natural", "random1", "random2", "random3", "random4", "random5", "random6", "random7"]

    for s in strategies:
        if time.time() > deadline - 40:
            break
        if s == "natural":
            order = list(range(total))
        else:
            order = list(range(total))
            random.shuffle(order)
        
        code = greedy_cover(order)
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
    
    def compute_unique_points(code_set, cnt):
        """
        Returns a dict mapping codeword index to array of unique points (cnt==1 in its ball).
        """
        unique = {}
        for c in code_set:
            ball = ball_arrays[c]
            mask = (cnt[ball] == 1)
            unique[c] = ball[mask]
        return unique

    def compute_unique_points_fast(code_set, cnt):
        """
        Fast version: only compute for codewords that might have unique points.
        A codeword c has unique points only if cnt[ball_arrays[c]] has any 1s.
        """
        unique = {}
        for c in code_set:
            ball = ball_arrays[c]
            mask = (cnt[ball] == 1)
            if np.any(mask):
                unique[c] = ball[mask]
            else:
                unique[c] = np.array([], dtype=np.int32)
        return unique

    def is_valid_swap(c_remove, c_add, unique_points):
        """
        Check if swapping c_remove for c_add maintains coverage.
        Valid if unique_points[c_remove] is subset of ball[c_add].
        """
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
        """
        Check if removing c1, c2 and adding c_add maintains coverage.
        """
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
        """
        Check if removing c1,c2,c3 and adding c_add1, c_add2 maintains coverage.
        """
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

    current_code = set(best_code)
    cnt = best_cnt.copy()
    best_size = len(current_code)
    best_code_list = list(current_code)
    
    no_improve_streak = 0
    max_streak = 2000
    
    while time.time() < deadline - 10.0 and no_improve_streak < max_streak:
        improved = False
        
        # 1. Compute unique points
        unique_points = compute_unique_points_fast(current_code, cnt)
        
        # 2. Try Deletions
        for c in list(current_code):
            if len(unique_points[c]) == 0:
                current_code.remove(c)
                cnt[ball_arrays[c]] -= 1
                improved = True
                break
                
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # Sort codewords by number of unique points (descending) for better search
        code_sorted = sorted(current_code, key=lambda c: len(unique_points[c]), reverse=True)
        
        # 3. Try 2-for-1 Swaps with informed candidates
        found_2for1 = False
        
        # Build informed candidate pool using coverers precomputation
        all_unique = set()
        for c in current_code:
            u = unique_points[c]
            for p in u:
                all_unique.add(int(p))
        
        candidate_pool = set()
        if all_unique:
            # For each unique point, use precomputed coverers to find candidates
            for p in list(all_unique)[:80]:  # Limit to 80 unique points
                p_idx = int(p)
                # coverers[p_idx] gives all codewords whose ball contains p_idx
                for c in coverers[p_idx]:
                    c_int = int(c)
                    if c_int not in current_code:
                        candidate_pool.add(c_int)
        
        # Add some random candidates
        for _ in range(50):
            c = random.randint(0, total - 1)
            if c not in current_code:
                candidate_pool.add(c)
        
        candidate_list = list(candidate_pool)
        random.shuffle(candidate_list)
        
        # Try pairs from the most "painful" codewords first
        subset_size = min(100, len(code_sorted))
        for i in range(subset_size):
            c1 = code_sorted[i]
            u1 = unique_points[c1]
            if len(u1) == 0:
                continue
            
            for j in range(i + 1, min(i + 12, len(code_sorted))):
                c2 = code_sorted[j]
                u2 = unique_points[c2]
                
                if len(u1) == 0 and len(u2) == 0:
                    continue
                
                for c_add in candidate_list:
                    if c_add in current_code:
                        continue
                    if is_valid_2for1(c1, c2, c_add, unique_points):
                        apply_2for1(c1, c2, c_add, current_code, cnt)
                        improved = True
                        found_2for1 = True
                        break
                if found_2for1:
                    break
            if found_2for1:
                break
                
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # 4. Try 1-for-1 Swaps with informed candidates
        swap_attempts = 0
        max_swap_attempts = 50
        while swap_attempts < max_swap_attempts and time.time() < deadline - 10.0:
            swap_attempts += 1
            if len(current_code) == 0:
                break
            
            # Pick a codeword with many unique points
            candidates_for_removal = [c for c in code_sorted if len(unique_points[c]) > 0]
            if not candidates_for_removal:
                break
            c_remove = random.choice(candidates_for_removal[:30])
            u_rem = unique_points[c_remove]
            
            if len(u_rem) == 0:
                current_code.remove(c_remove)
                cnt[ball_arrays[c_remove]] -= 1
                improved = True
                break
            
            # Build informed candidates using coverers
            local_candidates = set()
            for p in u_rem[:30]:
                p_idx = int(p)
                for c in coverers[p_idx]:
                    c_int = int(c)
                    if c_int not in current_code and c_int != c_remove:
                        local_candidates.add(c_int)
            
            # Add random candidates
            for _ in range(20):
                c = random.randint(0, total - 1)
                if c not in current_code and c != c_remove:
                    local_candidates.add(c)
            
            local_list = list(local_candidates)
            random.shuffle(local_list)
            
            for c_add in local_list:
                if c_add in current_code or c_add == c_remove:
                    continue
                if is_valid_swap(c_remove, c_add, unique_points):
                    apply_swap(c_remove, c_add, current_code, cnt)
                    improved = True
                    break
            if improved:
                break
                
        if improved:
            no_improve_streak = 0
            continue

        # 5. Try 3-for-2 Swaps with informed candidates
        if len(current_code) >= 3:
            found_3for2 = False
            attempts = 0
            max_attempts = 30
            while attempts < max_attempts and time.time() < deadline - 10.0:
                attempts += 1
                # Pick 3 codewords with the most unique points
                top_painful = [c for c in code_sorted if len(unique_points[c]) > 0][:20]
                if len(top_painful) < 3:
                    break
                c1, c2, c3 = random.sample(top_painful, 3)
                
                # Build informed candidates using coverers
                local_candidates = set()
                for c in [c1, c2, c3]:
                    u = unique_points[c]
                    for p in u[:15]:
                        p_idx = int(p)
                        for cc in coverers[p_idx]:
                            cc_int = int(cc)
                            if cc_int not in current_code:
                                local_candidates.add(cc_int)
                
                # Add random candidates
                for _ in range(25):
                    c = random.randint(0, total - 1)
                    if c not in current_code:
                        local_candidates.add(c)
                
                local_list = list(local_candidates)
                random.shuffle(local_list)
                
                # Try to find two valid additions
                for i in range(len(local_list)):
                    c_add1 = local_list[i]
                    if c_add1 in current_code:
                        continue
                    for j in range(i + 1, min(i + 30, len(local_list))):
                        c_add2 = local_list[j]
                        if c_add2 in current_code or c_add2 == c_add1:
                            continue
                        
                        if is_valid_3for2(c1, c2, c3, c_add1, c_add2, unique_points):
                            apply_3for2(c1, c2, c3, c_add1, c_add2, current_code, cnt)
                            improved = True
                            found_3for2 = True
                            break
                    if found_3for2:
                        break
                        
            if improved:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)
                no_improve_streak = 0
                continue

        if not improved:
            no_improve_streak += 1

    # Final: multiple delete passes with re-computation
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        changed = delete_redundant(current_code, cnt)
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code_list = list(current_code)
        if not changed:
            break
    
    # Final polish: try one more round of 2-for-1 with full candidate pool
    if time.time() < deadline - 5:
        unique_points = compute_unique_points_fast(current_code, cnt)
        all_unique = set()
        for c in current_code:
            u = unique_points[c]
            for p in u:
                all_unique.add(int(p))
        
        if all_unique:
            candidate_pool = set()
            for p in list(all_unique)[:100]:
                p_idx = int(p)
                for c in coverers[p_idx]:
                    c_int = int(c)
                    if c_int not in current_code:
                        candidate_pool.add(c_int)
            
            candidate_list = list(candidate_pool)
            random.shuffle(candidate_list)
            
            code_sorted = sorted(current_code, key=lambda c: len(unique_points[c]), reverse=True)
            for i in range(min(80, len(code_sorted))):
                c1 = code_sorted[i]
                u1 = unique_points[c1]
                if len(u1) == 0:
                    continue
                for j in range(i + 1, min(i + 8, len(code_sorted))):
                    c2 = code_sorted[j]
                    u2 = unique_points[c2]
                    if len(u1) == 0 and len(u2) == 0:
                        continue
                    for c_add in candidate_list[:80]:
                        if c_add in current_code:
                            continue
                        if is_valid_2for1(c1, c2, c_add, unique_points):
                            apply_2for1(c1, c2, c_add, current_code, cnt)
                            if len(current_code) < best_size:
                                best_size = len(current_code)
                                best_code_list = list(current_code)
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

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
