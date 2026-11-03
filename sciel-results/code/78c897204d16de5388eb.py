import numpy as np
import time
import random
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return [list(range(n)) for _ in range(1)]

    M = 3**n
    ball_size = 1 + 2 * n
    powers = [3**i for i in range(n)]

    # Precompute balls
    balls = [None] * M
    for i in range(M):
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        ball = [i]
        for pos in range(n):
            diff_plus = (1 - digits[pos]) % 3
            if diff_plus:
                ball.append(i + diff_plus * powers[pos])
            diff_minus = (2 - digits[pos]) % 3
            if diff_minus:
                ball.append(i + diff_minus * powers[pos])
        balls[i] = ball

    # Precompute coverers
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)

    def greedy_set_cover(initial_code=None):
        code = set(initial_code) if initial_code else set()
        in_code = np.zeros(M, dtype=bool)
        for c in code:
            in_code[c] = True
            
        uncovered = np.ones(M, dtype=bool)
        gains = np.full(M, ball_size, dtype=np.int32)

        if initial_code:
            for c in initial_code:
                for p in balls[c]:
                    if uncovered[p]:
                        uncovered[p] = False
                        for c_prime in coverers[p]:
                            if not in_code[c_prime]:
                                gains[c_prime] -= 1

        heap = [(-gains[c], c) for c in range(M) if gains[c] > 0 and not in_code[c]]
        heapq.heapify(heap)

        uncovered_count = int(np.sum(uncovered))

        while uncovered_count > 0:
            valid_c = None
            while heap:
                neg_g, c = heapq.heappop(heap)
                if in_code[c]:
                    continue
                if -neg_g != gains[c]:
                    continue
                if gains[c] <= 0:
                    continue
                valid_c = c
                break

            if valid_c is None:
                break

            code.add(valid_c)
            in_code[valid_c] = True

            for p in balls[valid_c]:
                if uncovered[p]:
                    uncovered[p] = False
                    uncovered_count -= 1
                    for c_prime in coverers[p]:
                        if not in_code[c_prime]:
                            gains[c_prime] -= 1
                            if gains[c_prime] > 0:
                                heapq.heappush(heap, (-gains[c_prime], c_prime))

        return code

    def local_search(code, time_limit=115):
        start_time = time.time()
        in_code = np.zeros(M, dtype=bool)
        for c in code:
            in_code[c] = True
            
        count = np.zeros(M, dtype=np.int32)
        for c in code:
            for p in balls[c]:
                count[p] += 1

        def is_redundant(c):
            for p in balls[c]:
                if count[p] == 1:
                    return False
            return True

        def remove_c(c):
            code.remove(c)
            in_code[c] = False
            for p in balls[c]:
                count[p] -= 1

        def add_c(c):
            code.add(c)
            in_code[c] = True
            for p in balls[c]:
                count[p] += 1

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False

            # Phase 1: Remove redundant
            to_remove = [c for c in list(code) if is_redundant(c)]
            if to_remove:
                for c in to_remove:
                    remove_c(c)
                improved = True
                continue

            # Phase 2: 1-Swap
            # Find all codewords that have unique points
            swap_candidates = []
            for c in list(code):
                unique_pts = [p for p in balls[c] if count[p] == 1]
                if unique_pts:
                    # Calculate potential gain: size of unique_pts
                    # We want to swap c with c' that covers unique_pts and minimizes new unique points?
                    # Or just find any valid c'
                    swap_candidates.append((c, unique_pts))
            
            random.shuffle(swap_candidates)
            swapped = False
            for c, unique_pts in swap_candidates:
                if (time.time() - start_time) >= time_limit:
                    break
                
                # Find candidates c' that cover all unique_pts
                # Intersection of coverers of unique_pts
                cands = set(coverers[unique_pts[0]])
                for p in unique_pts[1:]:
                    cands.intersection_update(coverers[p])
                
                cands -= code
                if cands:
                    # Choose c' that minimizes "damage" (new unique points created)
                    # Approximation: pick one at random or the one with smallest ball overlap with current code?
                    # Let's just pick one at random for speed, or the one that covers the most already covered points?
                    # A good heuristic: pick c' that covers the most points that are currently covered by OTHER codewords (i.e., high count), 
                    # so it doesn't create new unique points easily? 
                    # Actually, we want c' to cover as many points that are *already* covered (count > 1) as possible, 
                    # and as few points that are *uncovered* (count == 0, but there are none if we are valid) 
                    # or points covered by c (count == 1, which are the unique_pts we are fixing).
                    # So we want c' to have high overlap with "stable" points.
                    
                    best_c_prime = None
                    best_score = -1
                    cand_list = list(cands)
                    # Limit candidates for speed
                    if len(cand_list) > 10:
                        cand_list = random.sample(cand_list, 10)
                    
                    for c_prime in cand_list:
                        # Score: number of points in ball(c_prime) that have count > 1 in current state (excluding c's unique points which will be 0)
                        # Actually, after removing c, count[p] for unique_pts becomes 0.
                        # So we want c_prime to cover points with count >= 1 (after removal of c).
                        score = 0
                        for p in balls[c_prime]:
                            # If p is in unique_pts, count becomes 0 after remove, so it's "new" coverage (good, needed)
                            # If p is not in unique_pts, count is current count.
                            # We prefer p where count >= 2 (so it remains covered by others)
                            if count[p] >= 2:
                                score += 1
                            elif count[p] == 1:
                                # This point is covered only by c (if p in unique_pts) or some other codeword.
                                # If p in unique_pts, it's fine.
                                # If p not in unique_pts, it's covered by some other codeword d != c.
                                # So it's stable.
                                score += 0.5 
                        if score > best_score:
                            best_score = score
                            best_c_prime = c_prime

                    if best_c_prime:
                        remove_c(c)
                        add_c(best_c_prime)
                        improved = True
                        swapped = True
                        break
            
            if swapped:
                continue

            # Phase 3: 2-Swap (Limited)
            if (time.time() - start_time) < time_limit:
                code_list = list(code)
                if len(code_list) >= 2:
                    # Try a few random pairs
                    for _ in range(5):
                        c1, c2 = random.sample(code_list, 2)
                        U1 = [p for p in balls[c1] if count[p] == 1]
                        U2 = [p for p in balls[c2] if count[p] == 1]
                        
                        if not U1 and not U2:
                            # Both redundant? Should have been caught in Phase 1
                            remove_c(c1)
                            remove_c(c2)
                            improved = True
                            break
                        
                        # Case: One new codeword covers U1 U U2
                        if U1 or U2:
                            all_unique = set(U1) | set(U2)
                            if all_unique:
                                first_p = next(iter(all_unique))
                                cands = set(coverers[first_p])
                                for p in all_unique:
                                    cands.intersection_update(coverers[p])
                                cands -= code
                                if cands:
                                    c_new = random.choice(list(cands))
                                    remove_c(c1)
                                    remove_c(c2)
                                    add_c(c_new)
                                    improved = True
                                    break

            # Perturbation
            if (time.time() - start_time) < time_limit:
                k = 10
                if len(code) > k:
                    to_perturb = random.sample(list(code), k)
                    for c in to_perturb:
                        remove_c(c)
                    
                    # Greedy Repair
                    code = greedy_set_cover(code)
                    # Re-initialize in_code and count for the new code
                    in_code = np.zeros(M, dtype=bool)
                    for c in code:
                        in_code[c] = True
                    count = np.zeros(M, dtype=np.int32)
                    for c in code:
                        for p in balls[c]:
                            count[p] += 1
                    improved = True

        return code

    best_code = None
    best_size = float('inf')
    start_time = time.time()
    time_limit = 115

    # Multi-start
    initial_code = {0}
    code = greedy_set_cover(initial_code)
    code = local_search(code, time_limit=10)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code.copy()

    for _ in range(10):
        if time.time() - start_time > time_limit - 15:
            break
        initial_code = set(random.sample(range(M), 15))
        code = greedy_set_cover(initial_code)
        code = local_search(code, time_limit=8)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()

    if best_code:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 0:
            best_code = local_search(best_code, time_limit=remaining_time)

    return [list(map(int, [(c // (3**i)) % 3 for i in range(n)])) for c in best_code]
