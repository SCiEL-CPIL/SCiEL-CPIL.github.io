import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # 1. Precomputation
    all_words = [list(w) for w in product(range(q), repeat=n)]
    
    # Map word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Compute balls (neighbors) for each word
    balls = []
    for w in all_words:
        covered = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            orig_val = w_list[i]
            for val in range(q):
                if val != orig_val:
                    w_list[i] = val
                    covered.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig_val
        balls.append(covered)
    
    # Convert balls to numpy arrays for faster min/sum operations
    balls_np = [np.array(b, dtype=np.int32) for b in balls]
    
    # Precompute coverers: for each point, which words cover it
    coverers = [[] for _ in range(N)]
    for i, ball in enumerate(balls):
        for p in ball:
            coverers[p].append(i)
            
    # Convert coverers to numpy arrays for intersection logic if needed, 
    # but sets/lists are probably fine for small intersections.
    
    # 2. Greedy Initialization
    # Use bitsets for fast "new coverage" calculation
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in balls[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask
        
    all_mask = (1 << N) - 1
    current_cover_mask = 0
    codewords = []
    is_chosen = np.zeros(N, dtype=bool)
    
    # Standard Greedy
    while current_cover_mask != all_mask:
        if time.time() > deadline - 100:
            break
        best_i = -1
        best_count = 0
        for i in range(N):
            if is_chosen[i]:
                continue
            new_bits = ball_bitsets[i] & ~current_cover_mask
            if new_bits:
                count = new_bits.bit_count()
                if count > best_count:
                    best_count = count
                    best_i = i
        if best_i == -1:
            break
        codewords.append(best_i)
        is_chosen[best_i] = True
        current_cover_mask |= ball_bitsets[best_i]
        
    # Ensure validity (should be covered by greedy, but just in case)
    if current_cover_mask != all_mask:
        while current_cover_mask != all_mask:
            i = random.randint(0, N-1)
            if not is_chosen[i]:
                codewords.append(i)
                is_chosen[i] = True
            current_cover_mask |= ball_bitsets[i]
            if time.time() > deadline - 95:
                break

    # 3. Local Search Setup
    cover_count = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        cover_count[balls_np[cw]] += 1
        
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    random.seed(42)
    
    # Helper: Remove redundant codewords
    def remove_redundant(codewords, cover_count):
        removed = 0
        i = 0
        while i < len(codewords):
            cw = codewords[i]
            b = balls_np[cw]
            # Check if all points in ball are covered by >1 codeword
            if np.min(cover_count[b]) > 1:
                cover_count[b] -= 1
                codewords.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    # Initial cleanup
    remove_redundant(codewords, cover_count)
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)

    no_improvement = 0
    max_no_improvement = 20 # Restart perturbation more often
    
    while time.time() < deadline - 5:
        improved = False
        
        # 1. Try Removals
        removed = remove_redundant(codewords, cover_count)
        if removed > 0:
            improved = True
            no_improvement = 0
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue # Restart loop after modification
        
        # 2. Try Swaps
        swap_performed = False
        for i, cw in enumerate(codewords):
            if time.time() > deadline - 10:
                break
                
            b = balls_np[cw]
            # Find critical points: points in ball(cw) with count == 1
            critical_points = []
            for pt in b:
                if cover_count[pt] == 1:
                    critical_points.append(pt)
            
            if not critical_points:
                # Should have been removed
                continue
                
            # Find candidates that cover ALL critical points
            # Start with coverers of the first critical point
            candidate_set = set(coverers[critical_points[0]])
            for p in critical_points[1:]:
                candidate_set.intersection_update(coverers[p])
            
            # Remove current codeword and other existing codewords from candidates
            candidate_set.discard(cw)
            for other_cw in codewords:
                if other_cw != cw:
                    candidate_set.discard(other_cw)
            
            if candidate_set:
                # Evaluate candidates
                # Heuristic: Minimize sum of cover_count in the candidate's ball
                # This tends to reduce redundancy
                best_new_cw = -1
                min_cost = float('inf')
                
                # Limit candidates to speed up
                candidates_list = list(candidate_set)
                if len(candidates_list) > 50:
                    random.shuffle(candidates_list)
                    candidates_list = candidates_list[:50]
                
                for new_cw in candidates_list:
                    b_new = balls_np[new_cw]
                    cost = np.sum(cover_count[b_new])
                    if cost < min_cost:
                        min_cost = cost
                        best_new_cw = new_cw
                
                if best_new_cw != -1:
                    # Perform swap
                    cover_count[b] -= 1
                    b_new = balls_np[best_new_cw]
                    cover_count[b_new] += 1
                    codewords[i] = best_new_cw
                    swap_performed = True
                    improved = True
                    no_improvement = 0
                    # Try to remove redundancies created by the swap
                    remove_redundant(codewords, cover_count)
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                    break # Restart loop after swap
        
        if not improved:
            no_improvement += 1
            
            if no_improvement >= max_no_improvement:
                # Perturbation: Add random codewords to break local structure
                num_perturb = random.randint(2, 4)
                for _ in range(num_perturb):
                    rand_cw = random.randint(0, N - 1)
                    if rand_cw not in codewords:
                        codewords.append(rand_cw)
                        cover_count[balls_np[rand_cw]] += 1
                no_improvement = 0
                # Immediately try to clean up
                remove_redundant(codewords, cover_count)
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)

    # Final Polish: Run removals one last time on the best solution
    final_codewords = list(best_codewords)
    final_cover_count = np.zeros(N, dtype=np.int32)
    for cw in final_codewords:
        final_cover_count[balls_np[cw]] += 1
        
    remove_redundant(final_codewords, final_cover_count)
    
    # Verify final validity
    if np.min(final_cover_count) > 0:
        return [list(all_words[cw]) for cw in final_codewords]
    else:
        # Fallback
        return [list(all_words[cw]) for cw in codewords]
