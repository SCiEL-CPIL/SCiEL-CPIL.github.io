import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit

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
    # This is a boolean matrix of size total x total
    ball_mask = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mask[i, ball_arrays[i]] = True

    def greedy_cover(order=None):
        """Greedy set cover: repeatedly pick the codeword covering most uncovered points."""
        if order is None:
            order = list(range(total))
        covered = np.zeros(total, dtype=bool)
        code = []
        remaining = np.ones(total, dtype=bool)
        
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

    def local_search(code_set, cnt, time_limit=1.0):
        """
        Run deterministic local search on the given state.
        Returns the improved code_set and cnt.
        """
        start_time = time.time()
        no_improve_streak = 0
        max_streak = 500 # Reduced streak to allow more restarts in ILS
        
        while time.time() - start_time < time_limit and no_improve_streak < max_streak:
            improved = False
            
            # 1. Compute unique points
            unique_points = compute_unique_points_fast(code_set, cnt)
            
            # 2. Try Deletions
            for c in list(code_set):
                if len(unique_points[c]) == 0:
                    code_set.remove(c)
                    cnt[ball_arrays[c]] -= 1
                    improved = True
                    break
                    
            if improved:
                no_improve_streak = 0
                continue

            # 3. Try 2-for-1 Swaps
            code_list = list(code_set)
            random.shuffle(code_list)
            found_2for1 = False
            # Limit attempts to keep it fast
            for i in range(min(40, len(code_list))):
                c1 = code_list[i]
                u1 = unique_points[c1]
                if len(u1) == 0:
                    continue
                
                for j in range(i + 1, min(i + 5, len(code_list))):
                    c2 = code_list[j]
                    u2 = unique_points[c2]
                    
                    if len(u1) == 0 and len(u2) == 0:
                        continue
                    
                    for _ in range(10):
                        c_add = random.randint(0, total - 1)
                        if c_add in code_set:
                            continue
                        if is_valid_2for1(c1, c2, c_add, unique_points):
                            apply_2for1(c1, c2, c_add, code_set, cnt)
                            improved = True
                            found_2for1 = True
                            break
                    if found_2for1:
                        break
                if found_2for1:
                    break
                    
            if improved:
                no_improve_streak = 0
                continue

            # 4. Try 1-for-1 Swaps
            swap_attempts = 0
            max_swap_attempts = 20
            while swap_attempts < max_swap_attempts:
                swap_attempts += 1
                if len(code_set) == 0:
                    break
                
                c_remove = random.choice(list(code_set))
                u_rem = unique_points[c_remove]
                
                if len(u_rem) == 0:
                    code_set.remove(c_remove)
                    cnt[ball_arrays[c_remove]] -= 1
                    improved = True
                    break
                
                for _ in range(10):
                    c_add = random.randint(0, total - 1)
                    if c_add in code_set or c_add == c_remove:
                        continue
                    if is_valid_swap(c_remove, c_add, unique_points):
                        apply_swap(c_remove, c_add, code_set, cnt)
                        improved = True
                        break
                if improved:
                    break
                    
            if improved:
                no_improve_streak = 0
                continue

            # 5. Try 3-for-2 Swaps
            if len(code_set) >= 3:
                found_3for2 = False
                attempts = 0
                max_attempts = 10
                while attempts < max_attempts:
                    attempts += 1
                    c1, c2, c3 = random.sample(list(code_set), 3)
                    c_add1 = random.randint(0, total - 1)
                    c_add2 = random.randint(0, total - 1)
                    
                    if c_add1 in code_set or c_add2 in code_set or c_add1 == c_add2:
                        continue
                        
                    if is_valid_3for2(c1, c2, c3, c_add1, c_add2, unique_points):
                        apply_3for2(c1, c2, c3, c_add1, c_add2, code_set, cnt)
                        improved = True
                        found_3for2 = True
                        break
                        
                if improved:
                    no_improve_streak = 0
                    continue

            if not improved:
                no_improve_streak += 1

        return code_set, cnt

    # === Initial solution: multiple greedy strategies ===
    best_code = None
    best_size = total + 1
    best_cnt = None

    strategies = ["natural", "random1", "random2", "random3", "random4", "random5", "random6"]

    for s in strategies:
        if time.time() > deadline - 30:
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

    # === Iterated Local Search (ILS) / Simulated Annealing-like Perturbation ===
    
    current_code = set(best_code)
    cnt = best_cnt.copy()
    current_size = len(current_code)
    
    # Run initial local search
    current_code, cnt = local_search(current_code, cnt, time_limit=10.0)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)
        best_cnt = cnt.copy()

    # Perturbation loop
    # We try to escape local minima by perturbing the current best solution
    # and re-optimizing.
    while time.time() < deadline - 5.0:
        
        # Create a perturbed version of the current best code
        # Strategy: Remove K random codewords, then repair with greedy
        # K is chosen to be small enough to allow repair, large enough to change structure
        k_perturb = random.randint(3, 6)
        
        perturbed_code = set(best_code)
        perturbed_cnt = best_cnt.copy()
        
        # Remove k_perturb random codewords
        if len(perturbed_code) > k_perturb:
            to_remove = random.sample(list(perturbed_code), k_perturb)
            for c in to_remove:
                perturbed_code.remove(c)
                perturbed_cnt[ball_arrays[c]] -= 1
        
        # Repair: Add codewords to cover uncovered points
        # Find uncovered points
        uncovered = perturbed_cnt == 0
        if np.any(uncovered):
            # Greedy repair
            remaining = uncovered
            while np.any(remaining):
                best_c = -1
                best_new = -1
                # Try random candidates first, or just iterate
                # To be fast, we pick a random candidate that covers the most remaining
                # Since total is 2187, a full scan is ~2187 * 15 ops, which is fast
                for c in range(total):
                    if c in perturbed_code:
                        continue
                    new_cov = np.count_nonzero(remaining & ball_mask[c])
                    if new_cov > best_new:
                        best_new = new_cov
                        best_c = c
                        if best_new == ball_size:
                            break
                if best_c == -1:
                    # Should not happen if we have a valid cover initially
                    break
                perturbed_code.add(best_c)
                perturbed_cnt[ball_mask[best_c]] += 1
                remaining[ball_mask[best_c]] = False
        
        # Run local search on the perturbed state
        perturbed_code, perturbed_cnt = local_search(perturbed_code, perturbed_cnt, time_limit=5.0)
        
        if len(perturbed_code) < best_size:
            best_size = len(perturbed_code)
            best_code = list(perturbed_code)
            best_cnt = perturbed_cnt.copy()
            # If we found a new best, we can optionally restart perturbation from here
            # The loop naturally does this as it uses best_code

    # Final cleanup: delete redundant again on the very best
    final_code = set(best_code)
    final_cnt = best_cnt.copy()
    while delete_redundant(final_code, final_cnt):
        pass
    
    if len(final_code) < best_size:
        best_size = len(final_code)
        best_code = list(final_code)

    # Convert back to list of lists
    result = []
    for idx in best_code:
        w = []
        temp_idx = idx
        for i in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
    
    return result
