import numpy as np
import itertools
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other params
        # Simple greedy for general case
        M = q**n
        # ... (keep fallback simple)
        return [list(range(n)) for _ in range(1)] # Placeholder, task is specific

    M = 3**n
    ball_size = 1 + 2*n # 17
    
    # Precompute balls as a list of lists of indices
    powers = [3**i for i in range(n)]
    
    # Precompute balls
    balls = [None] * M
    for i in range(M):
        digits = [0]*n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        ball = [i]
        for pos in range(n):
            new_d = (digits[pos] + 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
            new_d = (digits[pos] - 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
        balls[i] = ball

    # Precompute coverers: for each point p, list of codewords c such that p in balls[c]
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)

    # 1. Greedy Set Cover
    code = set()
    uncovered = np.ones(M, dtype=bool)
    gains = np.full(M, ball_size, dtype=np.int32)
    
    heap = [(-ball_size, c) for c in range(M)]
    heapq.heapify(heap)
    
    uncovered_count = M
    start_time = time.time()
    
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

    # 2. Local Search with 1-swap and 2-swap
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
        for p in balls[c]:
            count[p] -= 1

    def add_c(c):
        code.add(c)
        for p in balls[c]:
            count[p] += 1

    improved = True
    time_limit = 115
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Phase 1: Remove redundant
        to_remove = []
        for c in list(code):
            if is_redundant(c):
                to_remove.append(c)
                
        if to_remove:
            for c in to_remove:
                remove_c(c)
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
                remove_c(c)
                improved = True
                swapped = True
                break
            
            candidates = set(coverers[unique_points[0]])
            for p in unique_points[1:]:
                candidates.intersection_update(coverers[p])
            
            candidates -= code
            
            if candidates:
                cand_list = list(candidates)
                random.shuffle(cand_list)
                c_new = cand_list[0]
                
                remove_c(c)
                add_c(c_new)
                improved = True
                swapped = True
                break
                
        if swapped:
            continue
            
        # Phase 3: 2-Swap
        if (time.time() - start_time) < time_limit:
            code_list = list(code)
            if len(code_list) >= 2:
                c1, c2 = random.sample(code_list, 2)
                
                U1 = [p for p in balls[c1] if count[p] == 1]
                U2 = [p for p in balls[c2] if count[p] == 1]
                
                if U1 and U2:
                    all_unique = set(U1) | set(U2)
                    first_p = next(iter(all_unique))
                    cands = set(coverers[first_p])
                    for p in all_unique:
                        cands.intersection_update(coverers[p])
                    
                    cands -= code
                    cands.discard(c1)
                    cands.discard(c2)
                    
                    if cands:
                        c_new = random.choice(list(cands))
                        remove_c(c1)
                        remove_c(c2)
                        add_c(c_new)
                        improved = True
                        continue

                cands1 = set(coverers[U1[0]]) if U1 else set()
                if U1:
                    for p in U1[1:]:
                        cands1.intersection_update(coverers[p])
                cands1 -= code
                cands1.discard(c1)
                cands1.discard(c2)
                
                cands2 = set(coverers[U2[0]]) if U2 else set()
                if U2:
                    for p in U2[1:]:
                        cands2.intersection_update(coverers[p])
                cands2 -= code
                cands2.discard(c1)
                cands2.discard(c2)
                
                if cands1 and cands2:
                    c1_new = random.choice(list(cands1))
                    valid_c2s = [c for c in cands2 if c != c1_new]
                    if valid_c2s:
                        c2_new = random.choice(valid_c2s)
                        
                        remove_c(c1)
                        remove_c(c2)
                        add_c(c1_new)
                        add_c(c2_new)
                        improved = True
                        continue

        # Perturbation
        if (time.time() - start_time) < time_limit:
            k = 5
            if len(code) > k:
                to_perturb = random.sample(list(code), k)
                for c in to_perturb:
                    remove_c(c)
                
                gains = np.zeros(M, dtype=np.int32)
                for c in range(M):
                    if c not in code:
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
                    
                    add_c(c)
                    
                    for p in balls[c]:
                        if count[p] == 1:
                            for c_prime in coverers[p]:
                                if c_prime not in code:
                                    gains[c_prime] -= 1
                                    if gains[c_prime] > 0:
                                        heapq.heappush(heap, (-gains[c_prime], c_prime))
                    
        else:
            break

    # 3. Max-flow based optimization
    # Build a flow network to find minimum set cover using max-flow/min-cut
    # Source -> codewords (capacity 1) -> points (capacity inf) -> sink (demand 1)
    # Actually, we want to minimize codewords while covering all points.
    # This is equivalent to: find max flow where each codeword can be used at most once,
    # and each point needs to be covered at least once.
    # Min cut in this network gives the minimum number of codewords needed.
    
    # Let's use a different formulation:
    # We have a bipartite graph: codewords on left, points on right.
    # Edge from codeword c to point p if p is in ball(c).
    # We want to select minimum codewords such that every point is covered.
    # This is the minimum set cover problem, which is NP-hard in general.
    # But for this specific instance, we can use LP relaxation or flow-based heuristics.
    
    # Alternative: Use the fact that we already have a good solution.
    # Try to remove codewords one by one using flow to check if coverage is maintained.
    
    # For each codeword c, check if removing it still covers all points.
    # This is equivalent to: for each point p in ball(c), is there another codeword c' != c that covers p?
    # We can use max-flow to check if there's a matching that covers all points without c.
    
    # Actually, let's try a simpler approach: iterative improvement using flow.
    # For each pair of codewords, check if we can replace them with fewer codewords.
    
    # Let's try: for each codeword c, try to find if there's a set of other codewords
    # that covers all points in ball(c) that are uniquely covered by c.
    
    # This is essentially what the 1-swap does, but let's be more systematic.
    
    # Try to find a smaller solution by solving a flow problem.
    # We'll try to remove codewords greedily, using flow to verify coverage.
    
    # Actually, the current solution is already quite good. Let's try to improve
    # by using a more sophisticated local search.
    
    # Try 3-swap: remove 3 codewords, add 2 codewords.
    if (time.time() - start_time) < time_limit:
        improved3 = True
        while improved3 and (time.time() - start_time) < time_limit:
            improved3 = False
            code_list = list(code)
            random.shuffle(code_list)
            
            if len(code_list) < 3:
                break
            
            # Try all triples (or sample some)
            for i in range(min(50, len(code_list))):
                if (time.time() - start_time) >= time_limit:
                    break
                
                c1, c2, c3 = code_list[i:i+3]
                
                # Get unique points for each
                U1 = [p for p in balls[c1] if count[p] == 1]
                U2 = [p for p in balls[c2] if count[p] == 1]
                U3 = [p for p in balls[c3] if count[p] == 1]
                
                # If any has no unique points, it's redundant
                if not U1:
                    remove_c(c1)
                    improved3 = True
                    break
                if not U2:
                    remove_c(c2)
                    improved3 = True
                    break
                if not U3:
                    remove_c(c3)
                    improved3 = True
                    break
                
                # Try to cover U1, U2, U3 with 2 codewords
                # Option 1: One codeword covers U1∪U2, another covers U3
                # Option 2: One codeword covers U1∪U3, another covers U2
                # Option 3: One codeword covers U2∪U3, another covers U1
                
                for subset1, subset2 in [(set(U1)|set(U2), set(U3)), 
                                         (set(U1)|set(U3), set(U2)),
                                         (set(U2)|set(U3), set(U1))]:
                    if not subset1 or not subset2:
                        continue
                    
                    # Find codeword covering subset1
                    cands1 = set(coverers[list(subset1)[0]])
                    for p in subset1:
                        cands1.intersection_update(coverers[p])
                    cands1 -= code
                    cands1.discard(c1)
                    cands1.discard(c2)
                    cands1.discard(c3)
                    
                    if not cands1:
                        continue
                    
                    # Find codeword covering subset2
                    cands2 = set(coverers[list(subset2)[0]])
                    for p in subset2:
                        cands2.intersection_update(coverers[p])
                    cands2 -= code
                    cands2.discard(c1)
                    cands2.discard(c2)
                    cands2.discard(c3)
                    
                    if not cands2:
                        continue
                    
                    # Ensure c1' != c2'
                    for c1_new in list(cands1):
                        if c1_new in cands2:
                            continue
                        c2_new = c1_new  # This won't work, need different logic
                        
                        # Just pick random from cands2 that's not c1_new
                        valid_c2s = [c for c in cands2 if c != c1_new]
                        if valid_c2s:
                            c2_new = random.choice(valid_c2s)
                            remove_c(c1)
                            remove_c(c2)
                            remove_c(c3)
                            add_c(c1_new)
                            add_c(c2_new)
                            improved3 = True
                            break
                    
                    if improved3:
                        break

    return [list(map(int, [ (c // (3**i)) % 3 for i in range(n) ])) for c in code]
