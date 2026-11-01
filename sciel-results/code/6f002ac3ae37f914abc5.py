"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

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

    # Precompute ball_indices: for each word, list of indices within Hamming distance <= radius
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

    def greedy_cover(order=None):
        """Greedy set cover: repeatedly pick the codeword covering most uncovered points."""
        if order is None:
            order = list(range(total))
        remaining = np.ones(total, dtype=bool)
        code = []
        
        while np.any(remaining):
            best_c = -1
            best_new = -1
            for c in order:
                new_cov = np.count_nonzero(remaining & ball_mask[c])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            if best_c == -1:
                break
            code.append(best_c)
            remaining[ball_mask[best_c]] = False
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
        # Fallback: use a simple construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # === Local Search with Efficient Swap Checks ===
    
    def compute_unique_points_fast(code_set, cnt):
        """
        Fast version: only compute for codewords that might have unique points.
        """
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
        """
        Check if swapping c_remove for c_add maintains coverage.
        Valid if unique_points[c_remove] is subset of ball[c_add].
        """
        u = unique_points.get(c_remove, np.array([], dtype=np.int32))
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
        u1 = unique_points.get(c1, np.array([], dtype=np.int32))
        u2 = unique_points.get(c2, np.array([], dtype=np.int32))
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
        u1 = unique_points.get(c1, np.array([], dtype=np.int32))
        u2 = unique_points.get(c2, np.array([], dtype=np.int32))
        u3 = unique_points.get(c3, np.array([], dtype=np.int32))
        
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
    max_streak = 500
    
    # Precompute candidate pool for swaps: all words not in current code
    # We'll maintain this dynamically
    
    while time.time() < deadline - 10.0 and no_improve_streak < max_streak:
        improved = False
        
        # 1. Compute unique points
        unique_points = compute_unique_points_fast(current_code, cnt)
        
        # 2. Try Deletions
        for c in list(current_code):
            if len(unique_points.get(c, np.array([], dtype=np.int32))) == 0:
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

        # 3. Try 1-for-1 swaps
        code_list = list(current_code)
        random.shuffle(code_list)
        
        for c in code_list:
            if time.time() > deadline - 10.0:
                break
            if c not in current_code:
                continue
            
            u = unique_points.get(c, np.array([], dtype=np.int32))
            if len(u) == 0:
                continue
            
            # Find candidates that cover all unique points of c
            candidates = []
            for cand in range(total):
                if cand in current_code:
                    continue
                if np.all(ball_mask[cand, u]):
                    candidates.append(cand)
            
            if candidates:
                c_add = random.choice(candidates)
                apply_swap(c, c_add, current_code, cnt)
                improved = True
                break
        
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # 4. Try 2-for-1 swaps
        code_list = list(current_code)
        random.shuffle(code_list)
        
        found_2for1 = False
        for i in range(len(code_list)):
            if time.time() > deadline - 10.0:
                break
            c1 = code_list[i]
            if c1 not in current_code:
                continue
            
            u1 = unique_points.get(c1, np.array([], dtype=np.int32))
            if len(u1) == 0:
                continue
            
            for j in range(i + 1, len(code_list)):
                c2 = code_list[j]
                if c2 not in current_code:
                    continue
                
                u2 = unique_points.get(c2, np.array([], dtype=np.int32))
                if len(u2) == 0:
                    continue
                
                # Find candidates that cover all unique points of c1 and c2
                candidates = []
                for cand in range(total):
                    if cand in current_code:
                        continue
                    if np.all(ball_mask[cand, u1]) and np.all(ball_mask[cand, u2]):
                        candidates.append(cand)
                
                if candidates:
                    c_add = random.choice(candidates)
                    apply_2for1(c1, c2, c_add, current_code, cnt)
                    found_2for1 = True
                    break
            
            if found_2for1:
                break
        
        if found_2for1:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # 5. Try 3-for-2 swaps
        code_list = list(current_code)
        random.shuffle(code_list)
        
        found_3for2 = False
        for i in range(len(code_list)):
            if time.time() > deadline - 10.0:
                break
            c1 = code_list[i]
            if c1 not in current_code:
                continue
            
            u1 = unique_points.get(c1, np.array([], dtype=np.int32))
            if len(u1) == 0:
                continue
            
            for j in range(i + 1, len(code_list)):
                c2 = code_list[j]
                if c2 not in current_code:
                    continue
                
                u2 = unique_points.get(c2, np.array([], dtype=np.int32))
                if len(u2) == 0:
                    continue
                
                for k in range(j + 1, len(code_list)):
                    c3 = code_list[k]
                    if c3 not in current_code:
                        continue
                    
                    u3 = unique_points.get(c3, np.array([], dtype=np.int32))
                    if len(u3) == 0:
                        continue
                    
                    # Find pairs of candidates that together cover all unique points
                    candidates = []
                    for cand1 in range(total):
                        if cand1 in current_code:
                            continue
                        if not np.all(ball_mask[cand1, u1] | ball_mask[cand1, u2] | ball_mask[cand1, u3]):
                            continue
                        for cand2 in range(total):
                            if cand2 in current_code or cand2 == cand1:
                                continue
                            if np.all(ball_mask[cand1, u1] | ball_mask[cand2, u1]) and \
                               np.all(ball_mask[cand1, u2] | ball_mask[cand2, u2]) and \
                               np.all(ball_mask[cand1, u3] | ball_mask[cand2, u3]):
                                candidates.append((cand1, cand2))
                                break
                        if candidates:
                            break
                    
                    if candidates:
                        c_add1, c_add2 = random.choice(candidates)
                        apply_3for2(c1, c2, c3, c_add1, c_add2, current_code, cnt)
                        found_3for2 = True
                        break
                
                if found_3for2:
                    break
            
            if found_3for2:
                break
        
        if found_3for2:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # No improvement found
        no_improve_streak += 1

    # Convert indices back to words
    return [idx_to_word(idx) for idx in best_code_list]

def idx_to_word(idx):
    word = []
    for i in range(n):
        word.append(idx % q)
        idx //= q
    word.reverse()
    return word
# EVOLVE-BLOCK-END
