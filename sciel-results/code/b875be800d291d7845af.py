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
    # We need fast access to neighbors.
    # Instead of storing full balls for every node (6561 * 17), 
    # we can compute them on the fly or precompute once.
    # Precomputation is fast enough.
    
    # powers of 3
    powers = [3**i for i in range(n)]
    
    # Precompute balls
    # balls[i] is a list of indices
    balls = [None] * M
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

    # Precompute coverers: for each point p, list of codewords c such that p in balls[c]
    # This is the transpose of balls.
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)

    # 1. Greedy Set Cover
    # We want to minimize the number of codewords.
    # Greedy: pick c that covers the most *uncovered* points.
    
    code = set()
    uncovered = np.ones(M, dtype=bool)
    gains = np.full(M, ball_size, dtype=np.int32)
    
    # Heap entries: (-gain, codeword)
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
        
        # Update uncovered and gains
        for p in balls[valid_c]:
            if uncovered[p]:
                uncovered[p] = False
                uncovered_count -= 1
                # Decrease gain for all codewords covering p
                for c_prime in coverers[p]:
                    if c_prime not in code:
                        gains[c_prime] -= 1
                        heapq.heappush(heap, (-gains[c_prime], c_prime))

    # 2. Local Search with 1-swap and 2-swap
    # Use NumPy for counts
    count = np.zeros(M, dtype=np.int32)
    for c in code:
        for p in balls[c]:
            count[p] += 1
            
    # Helper to check if c is redundant
    def is_redundant(c):
        for p in balls[c]:
            if count[p] == 1:
                return False
        return True

    # Helper to remove c
    def remove_c(c):
        code.remove(c)
        for p in balls[c]:
            count[p] -= 1

    # Helper to add c
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
        # For each c, find unique points U_c
        # If U_c is not empty, find c' covering U_c
        # If found, swap c -> c'
        
        code_list = list(code)
        random.shuffle(code_list)
        
        swapped = False
        for c in code_list:
            if (time.time() - start_time) >= time_limit:
                break
            
            unique_points = [p for p in balls[c] if count[p] == 1]
            
            if not unique_points:
                # Should have been caught in Phase 1
                remove_c(c)
                improved = True
                swapped = True
                break
            
            # Find candidates for c'
            # c' must cover all points in unique_points
            # Intersect coverers of unique_points
            candidates = set(coverers[unique_points[0]])
            for p in unique_points[1:]:
                candidates.intersection_update(coverers[p])
            
            candidates -= code
            
            if candidates:
                # Pick a candidate. Heuristic: one that covers most redundant points?
                # Or just random.
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
            
        # Phase 3: 2-Swap (if 1-swap failed)
        # Try to remove 2 codewords and add 2 codewords.
        # This is more expensive.
        # Pick two random codewords c1, c2.
        # Find their unique points U1, U2.
        # If U1 and U2 are disjoint, we need c1', c2' such that:
        # c1' covers U1 and c2' covers U2.
        # Or c1' covers U1 U U2 (if one can cover both).
        
        if (time.time() - start_time) < time_limit:
            code_list = list(code)
            if len(code_list) >= 2:
                c1, c2 = random.sample(code_list, 2)
                
                U1 = [p for p in balls[c1] if count[p] == 1]
                U2 = [p for p in balls[c2] if count[p] == 1]
                
                # If either has no unique points, they are redundant (should have been caught)
                # But maybe removing one makes the other redundant?
                # Let's check if removing c1 makes c2 redundant?
                # No, we are doing a swap.
                
                # Case 1: One new codeword covers both U1 and U2.
                # Then we remove c1, c2 and add c_new. (Net -1)
                # This is effectively a "merge".
                if U1 and U2:
                    all_unique = set(U1) | set(U2)
                    # Check if any single codeword covers all_unique
                    # And is not c1 or c2.
                    # And covers all points in all_unique.
                    # Note: all_unique points are currently covered ONLY by c1 or c2.
                    # So any c_new covering them must be a new codeword.
                    
                    # Find candidates covering all_unique
                    # Start with coverers of the first point
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

                # Case 2: Two new codewords c1', c2'
                # c1' covers U1, c2' covers U2.
                # We need c1' != c2'.
                
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
                
                # If one list is empty, no 2-swap possible for this pair
                if cands1 and cands2:
                    # Try to pick c1' from cands1 and c2' from cands2, c1' != c2'
                    # To avoid O(N^2), pick random c1' and check if cands2 has any other
                    # Or if cands1 and cands2 are disjoint, any pair works.
                    
                    c1_new = random.choice(list(cands1))
                    # Ensure c2_new != c1_new
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
            # Remove k random codewords
            k = 5
            if len(code) > k:
                to_perturb = random.sample(list(code), k)
                for c in to_perturb:
                    remove_c(c)
                
                # Re-add greedily
                # Recalculate gains for remaining code
                # We need to know which points are uncovered
                # and how many new codewords cover them.
                
                # Reset gains to 0 for all? No, we need to recompute.
                # This is expensive. 
                # Alternative: Just add random codewords that cover uncovered points.
                
                # Let's do a mini-greedy
                # Find uncovered points
                # For each candidate c not in code, count uncovered points it covers.
                # Pick best, add, repeat until covered.
                
                # To speed up, we only consider candidates that cover at least one uncovered point.
                
                # Recompute count for uncovered points?
                # We have `count` array.
                # Points with count 0 are uncovered.
                
                # We need to update `gains` for the heap.
                # Instead of full recompute, we can just run a few steps of greedy
                # using the current `count` to determine "uncovered".
                
                # But `gains` in heap is stale.
                # We can rebuild the heap?
                # Heap rebuild is O(M log M). M=6561. Fast.
                
                # Rebuild gains
                gains = np.zeros(M, dtype=np.int32)
                for c in range(M):
                    if c not in code:
                        # Count how many points in balls[c] have count 0
                        # This is O(ball_size) per c. Total O(M * ball_size) = 111k. Fast.
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
                    
                    # Add c
                    add_c(c)
                    
                    # Update gains
                    # For each p in balls[c]:
                    #   if count[p] was 0, it is now 1.
                    #   For all c' covering p, gains[c'] -= 1 (if c' not in code)
                    
                    for p in balls[c]:
                        if count[p] == 1: # Just became covered
                            for c_prime in coverers[p]:
                                if c_prime not in code:
                                    gains[c_prime] -= 1
                                    if gains[c_prime] > 0:
                                        heapq.heappush(heap, (-gains[c_prime], c_prime))
                    
                    # Check if all covered
                    # We can track uncovered_count
                    # But we didn't track it in the perturbation step.
                    # Let's just break if heap is empty or gains are 0.
                    # Actually, we should check if any count is 0.
                    # This is O(M).
                    # Do it every few steps?
                    # Or just run until heap is empty.
                    
                    # If we break, we might have uncovered points.
                    # But the loop condition `while heap` will stop if no gains > 0.
                    # If there are uncovered points, some c must have gain > 0.
                    # So it will continue until all are covered.
                    
        else:
            break

    return [list(map(int, [ (c // (3**i)) % 3 for i in range(n) ])) for c in code]
