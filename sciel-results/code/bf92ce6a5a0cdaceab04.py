import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # Generate all words as index arrays
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # Precompute ball_indices and ball_sets
    # ball_arrays[i] is numpy array of points covered by codeword i
    # ball_sets[i] is set of points covered by codeword i
    ball_arrays = []
    ball_sets = []
    for i in range(total):
        w = words[i]
        ball = [i]
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                # Calculate index
                # i is in base q.
                # digit at 'coord' is (i // q**coord) % q
                # new index = i + (v - orig) * q**coord
                # We need base powers
                pass
        
        # Faster index calculation using precomputed bases
        # base[k] = q**k
        base = [1] * n
        for i in range(n - 2, -1, -1):
            base[i] = base[i + 1] * q
            
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball.append(new_idx)
        
        ball_arrays.append(np.array(ball, dtype=np.int32))
        ball_sets.append(set(ball))

    # Greedy Cover
    def greedy_cover():
        covered = np.zeros(total, dtype=bool)
        code = set()
        remaining = np.ones(total, dtype=bool)
        
        # Precompute "potential" for each word? 
        # Just iterate.
        # To speed up, we can maintain a list of uncovered points.
        # But simple loop is fine for 2187 items.
        
        # Randomize order for diversity
        order = list(range(total))
        random.shuffle(order)
        
        while np.any(remaining):
            best_c = -1
            best_new = -1
            # Sample a subset if time is tight? No, 2187 is small.
            for c in order:
                ball = ball_arrays[c]
                new_cov = np.count_nonzero(remaining[ball])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            if best_c == -1:
                break
            code.add(best_c)
            remaining[ball_arrays[best_c]] = False
            
        return code

    # Initialize
    best_code = greedy_cover()
    
    # Build counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in best_code:
        cnt[ball_arrays[c]] += 1

    best_size = len(best_code)
    best_code_snapshot = set(best_code)

    # Local Search
    current_code = set(best_code)
    
    no_improve_streak = 0
    max_streak = 2000 # Allow many iterations
    
    while time.time() < deadline - 2.0 and no_improve_streak < max_streak:
        improved = False
        
        # 1. Delete Phase
        # Recalculate unique counts and find redundant
        redundant_found = False
        # We need to check all codewords
        # To be fair, check in random order
        code_list = list(current_code)
        random.shuffle(code_list)
        
        for c in code_list:
            ball = ball_arrays[c]
            if np.all(cnt[ball] >= 2):
                current_code.remove(c)
                cnt[ball] -= 1
                improved = True
                redundant_found = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_snapshot = set(current_code)
                break # Restart delete phase
        
        if redundant_found:
            no_improve_streak = 0
            continue
            
        # 2. Swap Phase
        # Find codeword with min unique_count > 0
        min_c = -1
        min_uc = 100
        for c in current_code:
            ball = ball_arrays[c]
            uc = np.sum(cnt[ball] == 1)
            if uc > 0 and uc < min_uc:
                min_uc = uc
                min_c = c
                
        swap_performed = False
        
        if min_c != -1:
            # Targeted Swap
            U = [p for p in ball_arrays[min_c] if cnt[p] == 1]
            # Find candidates
            cand_set = set(ball_sets[U[0]])
            for u in U[1:]:
                cand_set &= set(ball_sets[u])
            cand_set -= current_code
            
            if cand_set:
                c_new = random.choice(list(cand_set))
                current_code.remove(min_c)
                current_code.add(c_new)
                cnt[ball_arrays[min_c]] -= 1
                cnt[ball_arrays[c_new]] += 1
                swap_performed = True
                improved = True # Structure changed, might enable deletion
        else:
            # No codeword has unique points? Then all are redundant?
            # This implies cnt[p] >= 2 for all p in all balls.
            # Then we should have deleted them.
            # So this case shouldn't happen if delete phase works.
            pass

        if not swap_performed:
            # Random Valid Swap
            for _ in range(50):
                c_old = random.choice(list(current_code))
                c_new = random.randint(0, total - 1)
                if c_new in current_code:
                    continue
                
                # Check validity
                # Points in ball(c_old) not in ball(c_new) must have cnt >= 2
                diff = ball_sets[c_old] - ball_sets[c_new]
                if diff:
                    valid = True
                    for p in diff:
                        if cnt[p] < 2:
                            valid = False
                            break
                    if not valid:
                        continue
                
                # Perform swap
                current_code.remove(c_old)
                current_code.add(c_new)
                cnt[ball_arrays[c_old]] -= 1
                cnt[ball_arrays[c_new]] += 1
                swap_performed = True
                improved = True
                break
        
        if not improved:
            no_improve_streak += 1
        else:
            no_improve_streak = 0

    # Final cleanup
    while True:
        deleted = False
        for c in list(current_code):
            ball = ball_arrays[c]
            if np.all(cnt[ball] >= 2):
                current_code.remove(c)
                cnt[ball] -= 1
                deleted = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_snapshot = set(current_code)
        if not deleted:
            break

    # Use the best found
    final_code = best_code_snapshot
    
    # Convert back to list of lists
    result = []
    for idx in final_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
