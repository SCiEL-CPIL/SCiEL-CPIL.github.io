import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    N = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls using numpy vectorization
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls_arr = np.zeros((N, ball_size), dtype=np.int32)
    
    for w in range(N):
        neighbors = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors.append(new_w)
        balls_arr[w] = neighbors
    
    # Precompute for each word, which codewords cover it (inverse ball)
    # This helps with efficient gain computation
    # Actually, let's precompute the ball membership as a boolean matrix
    # ball_membership[i, j] = True if word j is in ball(i)
    # This is 6561 x 6561 which is ~43M bools = ~43MB, acceptable
    ball_membership = np.zeros((N, N), dtype=np.bool_)
    for i in range(N):
        ball_membership[i, balls_arr[i]] = True
    
    def idx_to_list(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]
    
    def prune(selected, cover_count):
        """Remove codewords that are not essential."""
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(selected):
                c = selected[i]
                ball = balls_arr[c]
                if np.min(cover_count[ball]) > 1:
                    cover_count[ball] -= 1
                    selected.pop(i)
                    changed = True
                else:
                    i += 1
        return selected
    
    def greedy_construct(cover_count, uncovered):
        """Greedy set cover with vectorized gain computation."""
        selected = []
        while np.any(uncovered):
            # Compute gain: for each candidate, how many uncovered words it covers
            # gain[i] = sum of uncovered over ball(i)
            gains = np.sum(uncovered[balls_arr], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            ball = balls_arr[best_idx]
            cover_count[ball] += 1
            uncovered[ball] = False
        return selected
    
    def local_search_swap(selected, cover_count, deadline):
        """Try random swaps to enable further pruning."""
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            if not selected:
                break
            for _ in range(80):
                if time.time() > deadline - 5.0:
                    break
                i = np.random.randint(len(selected))
                u = selected[i]
                v = np.random.randint(N)
                if v == u:
                    continue
                ball_u = balls_arr[u]
                ball_v = balls_arr[v]
                # Check safety
                # Words in ball_u \ ball_v must have cover_count >= 2
                # Create a boolean mask
                in_u = np.zeros(N, dtype=np.bool_)
                in_u[ball_u] = True
                in_v = np.zeros(N, dtype=np.bool_)
                in_v[ball_v] = True
                risk_mask = in_u & ~in_v
                if np.any(risk_mask):
                    if np.min(cover_count[risk_mask]) < 2:
                        continue
                # Safe swap
                cover_count[ball_u] -= 1
                cover_count[ball_v] += 1
                selected[i] = v
                new_len_before = len(selected)
                selected = prune(selected, cover_count)
                if len(selected) < new_len_before:
                    improved = True
                    break
                else:
                    # Revert
                    cover_count[ball_v] -= 1
                    cover_count[ball_u] += 1
                    selected[i] = u
        return selected
    
    def local_search_2for1(selected, cover_count, deadline):
        """Try to replace two codewords with one."""
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            K = len(selected)
            if K < 2:
                break
            for _ in range(30):
                if time.time() > deadline - 5.0:
                    break
                i1 = np.random.randint(K)
                i2 = np.random.randint(K)
                if i1 == i2:
                    continue
                if i1 > i2:
                    i1, i2 = i2, i1
                c1 = selected[i1]
                c2 = selected[i2]
                ball1 = balls_arr[c1]
                ball2 = balls_arr[c2]
                
                # Find words uniquely covered by c1 or c2
                unique1 = ball1[cover_count[ball1] == 1]
                unique2 = ball2[cover_count[ball2] == 1]
                all_unique = set(unique1) | set(unique2)
                
                if not all_unique:
                    # Can remove one
                    min1 = np.min(cover_count[ball1])
                    min2 = np.min(cover_count[ball2])
                    if min1 >= min2:
                        cover_count[ball1] -= 1
                        selected.pop(i1)
                    else:
                        cover_count[ball2] -= 1
                        selected.pop(i2)
                    selected = prune(selected, cover_count)
                    improved = True
                    break
                
                # Try to find v covering all_unique
                found = False
                for _ in range(15):
                    v = np.random.randint(N)
                    ball_v = balls_arr[v]
                    ball_v_set = set(ball_v)
                    if all(x in ball_v_set for x in all_unique):
                        # Check safety
                        in1 = np.zeros(N, dtype=np.bool_)
                        in1[ball1] = True
                        in2 = np.zeros(N, dtype=np.bool_)
                        in2[ball2] = True
                        in_v = np.zeros(N, dtype=np.bool_)
                        in_v[ball_v] = True
                        combined = in1 | in2
                        risk_mask = combined & ~in_v
                        if np.any(risk_mask):
                            cc_check = cover_count.copy()
                            cc_check[ball1] -= 1
                            cc_check[ball2] -= 1
                            if np.min(cc_check[risk_mask]) < 2:
                                continue
                        # Perform 2-for-1
                        cover_count[ball1] -= 1
                        cover_count[ball2] -= 1
                        cover_count[ball_v] += 1
                        selected.pop(max(i1, i2))
                        selected.pop(min(i1, i2))
                        selected.append(v)
                        selected = prune(selected, cover_count)
                        improved = True
                        found = True
                        break
                if found:
                    break
        return selected
    
    def local_search_3for2(selected, cover_count, deadline):
        """Try to replace three codewords with two."""
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            K = len(selected)
            if K < 3:
                break
            for _ in range(15):
                if time.time() > deadline - 5.0:
                    break
                i1 = np.random.randint(K)
                i2 = np.random.randint(K)
                i3 = np.random.randint(K)
                # Ensure distinct
                if i1 == i2 or i1 == i3 or i2 == i3:
                    continue
                c1, c2, c3 = selected[i1], selected[i2], selected[i3]
                ball1, ball2, ball3 = balls_arr[c1], balls_arr[c2], balls_arr[c3]
                
                # Find words uniquely covered by any of the three
                unique = set()
                for b in [ball1, ball2, ball3]:
                    unique |= set(b[cover_count[b] == 1])
                
                if not unique:
                    # Can remove one
                    mins = [np.min(cover_count[b]) for b in [ball1, ball2, ball3]]
                    idx_to_remove = [i1, i2, i3][np.argmin(mins)]
                    idx_in_list = selected.index(idx_to_remove)
                    b_to_remove = balls_arr[idx_to_remove]
                    cover_count[b_to_remove] -= 1
                    selected.pop(idx_in_list)
                    selected = prune(selected, cover_count)
                    improved = True
                    break
                
                # Try to find two words v, w covering all unique
                # This is harder; try random pairs
                found = False
                for _ in range(10):
                    v = np.random.randint(N)
                    w = np.random.randint(N)
                    if v == w:
                        continue
                    ball_v = set(balls_arr[v])
                    ball_w = set(balls_arr[w])
                    if all(x in ball_v or x in ball_w for x in unique):
                        # Check safety
                        in1 = np.zeros(N, dtype=np.bool_)
                        in1[ball1] = True
                        in2 = np.zeros(N, dtype=np.bool_)
                        in2[ball2] = True
                        in3 = np.zeros(N, dtype=np.bool_)
                        in3[ball3] = True
                        in_v = np.zeros(N, dtype=np.bool_)
                        in_v[balls_arr[v]] = True
                        in_w = np.zeros(N, dtype=np.bool_)
                        in_w[balls_arr[w]] = True
                        combined = in1 | in2 | in3
                        new_cover = in_v | in_w
                        risk_mask = combined & ~new_cover
                        if np.any(risk_mask):
                            cc_check = cover_count.copy()
                            cc_check[ball1] -= 1
                            cc_check[ball2] -= 1
                            cc_check[ball3] -= 1
                            if np.min(cc_check[risk_mask]) < 2:
                                continue
                        # Perform 3-for-2
                        cover_count[ball1] -= 1
                        cover_count[ball2] -= 1
                        cover_count[ball3] -= 1
                        cover_count[balls_arr[v]] += 1
                        cover_count[balls_arr[w]] += 1
                        # Remove the three, add two
                        for idx in sorted([i1, i2, i3], reverse=True):
                            selected.pop(idx)
                        selected.append(v)
                        selected.append(w)
                        selected = prune(selected, cover_count)
                        improved = True
                        found = True
                        break
                if found:
                    break
        return selected
    
    best_size = N
    best_set = None
    
    # Main search loop
    while time.time() < deadline - 10.0:
        # 1. Greedy Construction
        cover_count = np.zeros(N, dtype=np.int32)
        uncovered = np.ones(N, dtype=bool)
        selected = greedy_construct(cover_count, uncovered)
        
        if np.any(uncovered):
            continue
        
        # 2. Initial Pruning
        selected = prune(selected, cover_count)
        
        # 3. Local Search: Swaps
        selected = local_search_swap(selected, cover_count, deadline)
        
        # 4. Local Search: 2-for-1
        selected = local_search_2for1(selected, cover_count, deadline)
        
        # 5. Local Search: 3-for-2
        selected = local_search_3for2(selected, cover_count, deadline)
        
        # 6. More swaps after 2-for-1 and 3-for-2
        selected = local_search_swap(selected, cover_count, deadline)
        
        # 7. Final Pruning
        selected = prune(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # If we're getting close to the deadline, do a final polish
        if time.time() > deadline - 20.0 and best_set is not None:
            cover_count_best = np.zeros(N, dtype=np.int32)
            for c in best_set:
                cover_count_best[balls_arr[c]] += 1
            best_set = prune(best_set, cover_count_best)
            best_set = local_search_swap(best_set, cover_count_best, deadline)
            best_set = local_search_2for1(best_set, cover_count_best, deadline)
            best_set = local_search_3for2(best_set, cover_count_best, deadline)
            best_set = prune(best_set, cover_count_best)
            if len(best_set) < best_size:
                best_size = len(best_set)
                best_set = best_set.copy()
            break
    
    if best_set is None:
        return [idx_to_list(w) for w in range(N)]
        
    return [idx_to_list(w) for w in best_set]
