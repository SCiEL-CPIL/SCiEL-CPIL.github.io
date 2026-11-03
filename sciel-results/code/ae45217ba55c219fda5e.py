import numpy as np
import itertools
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    M = 3**8
    
    # Precompute balls
    # balls[i] is a tuple of ints
    balls = [None] * M
    for i in range(M):
        digits = [0]*n
        x = i
        for j in range(n):
            digits[j] = x % 3
            x //= 3
        
        ball = [i]
        for pos in range(n):
            # +1
            d = digits[:]
            d[pos] = (d[pos] + 1) % 3
            val = 0
            for k in range(n-1, -1, -1):
                val = val * 3 + d[k]
            ball.append(val)
            
            # -1
            d = digits[:]
            d[pos] = (d[pos] - 1) % 3
            val = 0
            for k in range(n-1, -1, -1):
                val = val * 3 + d[k]
            ball.append(val)
        
        balls[i] = tuple(ball)
        
    # Precompute coverers: for each word p, which codewords cover it?
    # coverers[p] is a list of codewords c such that p in balls[c]
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)
            
    # Greedy Set Cover with Heap
    code = set()
    uncovered = [True] * M
    gains = [17] * M # Initial gain is size of ball
    heap = []
    for c in range(M):
        heapq.heappush(heap, (-17, c))
        
    uncovered_count = M
    
    while uncovered_count > 0:
        # Pop valid max gain
        valid_c = None
        while heap:
            neg_g, c = heapq.heappop(heap)
            if c in code:
                continue
            # Check if gain is still valid (lazy deletion)
            # We don't store current gain in heap, we check against gains[c]
            if -neg_g != gains[c]:
                continue
            valid_c = c
            break
            
        if valid_c is None:
            # Should not happen if logic is correct
            break
            
        # Add valid_c to code
        code.add(valid_c)
        
        # Update gains for neighbors
        for p in balls[valid_c]:
            if uncovered[p]:
                uncovered[p] = False
                uncovered_count -= 1
                # Update gains for all codewords that cover p
                for c_prime in coverers[p]:
                    if c_prime not in code:
                        gains[c_prime] -= 1
                        heapq.heappush(heap, (-gains[c_prime], c_prime))
                        
    # Local Search
    start_time = time.time()
    time_limit = 110
    
    # Prepare count array for local search
    in_code = np.zeros(M, dtype=bool)
    for c in code:
        in_code[c] = True
        
    count = np.zeros(M, dtype=np.int32)
    for c in code:
        for p in balls[c]:
            count[p] += 1
            
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Phase 1: Remove redundant
        to_remove = []
        for c in list(code):
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
                in_code[c] = False
                for p in balls[c]:
                    count[p] -= 1
            improved = True
            continue
        
        # Phase 2: Swap
        swapped = False
        code_list = list(code)
        random.shuffle(code_list)
        
        for c in code_list:
            if (time.time() - start_time) >= time_limit:
                break
            
            unique_points = []
            for p in balls[c]:
                if count[p] == 1:
                    unique_points.append(p)
            
            if not unique_points:
                code.remove(c)
                in_code[c] = False
                for p in balls[c]:
                    count[p] -= 1
                improved = True
                swapped = True
                break
                
            # Find candidates
            candidates = set(balls[unique_points[0]])
            for p in unique_points[1:]:
                candidates.intersection_update(balls[p])
            
            candidates -= code
            
            if candidates:
                best_c_new = None
                best_score = -1
                # Limit candidates
                cand_list = list(candidates)
                random.shuffle(cand_list)
                for c_new in cand_list[:20]:
                    score = 0
                    for p in balls[c_new]:
                        if count[p] > 1:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_c_new = c_new
                
                if best_c_new is not None:
                    code.remove(c)
                    in_code[c] = False
                    for p in balls[c]:
                        count[p] -= 1
                        
                    code.add(best_c_new)
                    in_code[best_c_new] = True
                    for p in balls[best_c_new]:
                        count[p] += 1
                        
                    improved = True
                    swapped = True
                    break
                
        if swapped:
            continue
        else:
            break
            
    return [list(map(int, [c % 3 for _ in range(n)] if False else [ (c // (3**i)) % 3 for i in range(n) ])) for c in code]
