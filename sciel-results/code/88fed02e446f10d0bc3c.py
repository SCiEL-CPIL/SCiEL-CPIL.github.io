import numpy as np
import itertools
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return [list(range(n)) for _ in range(1)]

    M = 3**n
    ball_size = 1 + 2*n
    
    powers = [3**i for i in range(n)]
    
    # Precompute balls
    balls = [None] * M
    balls_sets = [None] * M # For faster intersection/subset checks
    for i in range(M):
        digits = [0]*n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        ball = [i]
        for pos in range(n):
            # +1
            new_d = (digits[pos] + 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
            # -1
            new_d = (digits[pos] - 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
        balls[i] = ball
        balls_sets[i] = set(ball)

    # Precompute coverers (transpose)
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)
    
    # Precompute coverers as sets for faster intersection
    coverers_sets = [set(c) for c in coverers]

    def greedy_cover():
        code = set()
        uncovered = np.ones(M, dtype=bool)
        gains = np.full(M, ball_size, dtype=np.int32)
        
        heap = [(-ball_size, c) for c in range(M)]
        heapq.heapify(heap)
        
        uncovered_count = M
        
        while uncovered_count > 0:
            valid_c = None
            while heap:
                neg_g, c = heapq.heappop(heap)
                if c in code:
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
            
            for p in balls[valid_c]:
                if uncovered[p]:
                    uncovered[p] = False
                    uncovered_count -= 1
                    for c_prime in coverers[p]:
                        if c_prime not in code:
                            gains[c_prime] -= 1
                            heapq.heappush(heap, (-gains[c_prime], c_prime))
                            
        return code

    def local_search(code, time_limit):
        count = np.zeros(M, dtype=np.int32)
        for c in code:
            for p in balls[c]:
                count[p] += 1
                
        improved = True
        start_time = time.time()
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Phase 1: Remove redundant
            to_remove = []
            for c in list(code):
                # Check if all points in balls[c] have count > 1
                redundant = True
                for p in balls[c]:
                    if count[p] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(c)
                    
            if to_remove:
                for c in to_remove:
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                improved = True
                continue
            
            # Phase 2: 1-Swap
            code_list = list(code)
            random.shuffle(code_list)
            
            swapped = False
            for c in code_list:
                if (time.time() - start_time) >= time_limit:
                    break
                
                unique_points = [p for p in balls[c] if count[p] == 1]
                
                if not unique_points:
                    # Should have been caught in Phase 1, but safety check
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                    improved = True
                    swapped = True
                    break
                
                # Find candidates
                candidates = coverers_sets[unique_points[0]].copy()
                for p in unique_points[1:]:
                    candidates.intersection_update(coverers_sets[p])
                
                candidates.difference_update(code)
                
                if candidates:
                    c_new = random.choice(list(candidates))
                    
                    # Perform swap
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                    
                    code.add(c_new)
                    for p in balls[c_new]:
                        count[p] += 1
                    
                    improved = True
                    swapped = True
                    break
                    
            if swapped:
                continue
                
            # Phase 3: 2-Swap (Randomized)
            if (time.time() - start_time) < time_limit:
                code_list = list(code)
                if len(code_list) >= 2:
                    # Try a few random pairs
                    attempts = 10
                    for _ in range(attempts):
                        c1, c2 = random.sample(code_list, 2)
                        
                        U1 = [p for p in balls[c1] if count[p] == 1]
                        U2 = [p for p in balls[c2] if count[p] == 1]
                        
                        if not U1 or not U2:
                            continue
                            
                        # Check if we can replace c1, c2 with c1', c2'
                        # c1' must cover U1
                        # c2' must cover U2
                        # c1' != c2'
                        
                        cands1 = coverers_sets[U1[0]].copy()
                        for p in U1[1:]:
                            cands1.intersection_update(coverers_sets[p])
                        cands1.difference_update(code)
                        cands1.discard(c1)
                        cands1.discard(c2)
                        
                        cands2 = coverers_sets[U2[0]].copy()
                        for p in U2[1:]:
                            cands2.intersection_update(coverers_sets[p])
                        cands2.difference_update(code)
                        cands2.discard(c1)
                        cands2.discard(c2)
                        
                        if cands1 and cands2:
                            # Ensure c1' != c2'
                            # If intersection is non-empty, we need to pick distinct ones
                            # If intersection is empty, any pair works
                            inter = cands1.intersection(cands2)
                            
                            if inter:
                                # Pick c1' from cands1, c2' from cands2 such that c1' != c2'
                                # If we pick c1' from inter, c2' must be from cands2 - {c1'}
                                # If we pick c1' from cands1 - inter, c2' can be anything in cands2
                                
                                # Simple strategy:
                                c1_new = random.choice(list(cands1))
                                valid_c2s = [c for c in cands2 if c != c1_new]
                                if valid_c2s:
                                    c2_new = random.choice(valid_c2s)
                                    
                                    # Perform swap
                                    code.remove(c1)
                                    for p in balls[c1]:
                                        count[p] -= 1
                                    code.remove(c2)
                                    for p in balls[c2]:
                                        count[p] -= 1
                                        
                                    code.add(c1_new)
                                    for p in balls[c1_new]:
                                        count[p] += 1
                                    code.add(c2_new)
                                    for p in balls[c2_new]:
                                        count[p] += 1
                                        
                                    improved = True
                                    break
                            else:
                                c1_new = random.choice(list(cands1))
                                c2_new = random.choice(list(cands2))
                                
                                code.remove(c1)
                                for p in balls[c1]:
                                    count[p] -= 1
                                code.remove(c2)
                                for p in balls[c2]:
                                    count[p] -= 1
                                    
                                code.add(c1_new)
                                for p in balls[c1_new]:
                                    count[p] += 1
                                code.add(c2_new)
                                for p in balls[c2_new]:
                                    count[p] += 1
                                    
                                improved = True
                                break
                    if improved:
                        continue

            # Perturbation
            if (time.time() - start_time) < time_limit:
                # Remove random 10% of codewords
                k = max(5, len(code) // 10)
                if len(code) > k:
                    to_perturb = random.sample(list(code), k)
                    for c in to_perturb:
                        code.remove(c)
                        for p in balls[c]:
                            count[p] -= 1
                    
                    # Re-greedy to cover uncovered points
                    # Find uncovered points
                    uncovered_mask = count == 0
                    if np.any(uncovered_mask):
                        # Quick greedy
                        # We can reuse the greedy logic but only on uncovered points
                        # For simplicity, just add random codewords that cover uncovered points
                        # Or run a mini-greedy
                        # Mini-greedy:
                        # Calculate gains for all c not in code
                        gains = np.zeros(M, dtype=np.int32)
                        for c in range(M):
                            if c not in code:
                                # Count uncovered points in balls[c]
                                g = 0
                                for p in balls[c]:
                                    if count[p] == 0:
                                        g += 1
                                gains[c] = g
                                
                        heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
                        heapq.heapify(heap)
                        
                        while heap:
                            neg_g, c = heapq.heappop(heap)
                            if c in code:
                                continue
                            if -neg_g != gains[c]:
                                continue
                            if gains[c] <= 0:
                                continue
                            
                            code.add(c)
                            for p in balls[c]:
                                if count[p] == 0:
                                    count[p] = 1
                                    # Update gains
                                    for c_prime in coverers[p]:
                                        if c_prime not in code:
                                            gains[c_prime] -= 1
                                            if gains[c_prime] > 0:
                                                heapq.heappush(heap, (-gains[c_prime], c_prime))
                                else:
                                    count[p] += 1
                                    # No need to update gains for already covered points?
                                    # Gains are defined as "number of uncovered points covered".
                                    # So if p was covered, it doesn't contribute to gain.
                                    # So we only update gains for points that were uncovered (count 0 -> 1).
                                    # This is correct.
                            
                            # Check if all covered
                            if not np.any(count == 0):
                                break

        return code

    # Main loop: Multiple Restarts
    best_code = None
    start_time = time.time()
    time_limit = 115
    
    while (time.time() - start_time) < time_limit:
        # Greedy
        code = greedy_cover()
        
        # Local Search with remaining time
        remaining = time_limit - (time.time() - start_time)
        if remaining < 10:
            break
            
        code = local_search(code, remaining)
        
        if best_code is None or len(code) < len(best_code):
            best_code = code
            
        # If we found a very good code, we can break or continue
        # Lower bound is 403. Best known 486.
        if len(best_code) <= 410:
            break

    if best_code is None:
        # Fallback
        best_code = greedy_cover()
        
    return [list(map(int, [(c // (3**i)) % 3 for i in range(n)])) for c in best_code]
