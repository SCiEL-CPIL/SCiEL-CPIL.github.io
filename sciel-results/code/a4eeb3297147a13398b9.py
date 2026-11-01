import time
import numpy as np
import random
from itertools import combinations

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # Generate all words as index arrays
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

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

    ball_arrays = [np.array(bi, dtype=np.int32) for bi in ball_indices]

    # Build ball_mask: ball_mask[i, j] is True if j is in ball of i
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

    def find_2for1(code_set, cnt):
        """Find a valid 2-for-1 swap: remove c1, c2 and add c_add."""
        code_list = list(code_set)
        if len(code_list) < 2:
            return None
        
        # Compute unique points for each codeword
        unique_points = {}
        for c in code_list:
            ball_c = ball_arrays[c]
            unique_points[c] = ball_c[cnt[ball_c] == 1]
        
        # Try all pairs (c1, c2) and find a c_add that covers their unique points
        for c1, c2 in combinations(code_list, 2):
            u1 = unique_points[c1]
            u2 = unique_points[c2]
            
            # Union of unique points that must be covered by c_add
            if len(u1) == 0 and len(u2) == 0:
                # Both are redundant, just delete them (0-for-2)
                return (c1, c2, None)
            
            # Find candidates that cover both u1 and u2
            candidates = []
            for c_add in range(total):
                if c_add in code_set:
                    continue
                if len(u1) > 0 and not np.all(ball_mask[c_add, u1]):
                    continue
                if len(u2) > 0 and not np.all(ball_mask[c_add, u2]):
                    continue
                candidates.append(c_add)
            
            if candidates:
                # Pick the one with maximum coverage of remaining points
                best_c_add = candidates[0]
                best_cov = -1
                for c_add in candidates:
                    cov = np.count_nonzero(ball_mask[c_add])
                    if cov > best_cov:
                        best_cov = cov
                        best_c_add = c_add
                return (c1, c2, best_c_add)
        
        return None

    def apply_2for1(code_set, cnt, swap):
        """Apply a 2-for-1 swap."""
        c1, c2, c_add = swap
        code_set.remove(c1)
        code_set.remove(c2)
        cnt[ball_arrays[c1]] -= 1
        cnt[ball_arrays[c2]] -= 1
        if c_add is not None:
            code_set.add(c_add)
            cnt[ball_arrays[c_add]] += 1

    # === Initial solution: multiple greedy strategies ===
    best_code = None
    best_size = total + 1
    best_cnt = None

    random.seed(42)
    strategies = ["natural", "random1", "random2", "random3"]

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

    # === Local Search with 2-for-1 moves ===
    current_code = set(best_code)
    cnt = best_cnt.copy()
    best_size = len(current_code)
    best_code_list = list(current_code)
    
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline - 10.0 and steps < max_steps:
        steps += 1
        
        improved = False
        
        # 1. Try deletions
        code_list = list(current_code)
        random.shuffle(code_list)
        for c in code_list:
            if np.all(cnt[ball_arrays[c]] >= 2):
                current_code.remove(c)
                cnt[ball_arrays[c]] -= 1
                improved = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)
                break
        
        if improved:
            continue
        
        # 2. Try 2-for-1 swaps
        if time.time() < deadline - 15.0:
            swap = find_2for1(current_code, cnt)
            if swap is not None:
                apply_2for1(current_code, cnt, swap)
                improved = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)
        
        # 3. Try 1-for-1 swaps (neutral moves to explore state space)
        if not improved and len(current_code) > 0:
            c_remove = random.choice(list(current_code))
            c_add = random.randint(0, total - 1)
            
            if c_add in current_code or c_add == c_remove:
                continue
            
            # Check validity
            u = ball_arrays[c_remove][(cnt[ball_arrays[c_remove]] == 1)]
            valid = True
            if len(u) > 0:
                valid = np.all(ball_mask[c_add, u])
            
            if valid:
                current_code.remove(c_remove)
                current_code.add(c_add)
                cnt[ball_arrays[c_remove]] -= 1
                cnt[ball_arrays[c_add]] += 1
        
        # 4. Occasionally try adding a codeword and then removing two (3-for-2)
        if not improved and random.random() < 0.1 and time.time() < deadline - 20.0:
            # Add a random codeword
            c_add = random.randint(0, total - 1)
            if c_add not in current_code:
                current_code.add(c_add)
                cnt[ball_arrays[c_add]] += 1
                
                # Now try to find a 2-for-1
                swap = find_2for1(current_code, cnt)
                if swap is not None:
                    apply_2for1(current_code, cnt, swap)
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code_list = list(current_code)
                else:
                    # Revert
                    current_code.remove(c_add)
                    cnt[ball_arrays[c_add]] -= 1

    # Final: one more delete pass
    while delete_redundant(current_code, cnt):
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
