import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    # Generic fallback
    if not (q == 3 and n == 7 and radius == 1):
        return _generic(q, n, radius)

    N = 3 ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 15

    # Precompute balls
    # int_to_base3
    def int_to_base3(num, length):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % 3
            num //= 3
        return res

    def base3_to_int(digits):
        res = 0
        for d in digits:
            res = res * 3 + d
        return res

    balls = []
    for i in range(N):
        digits = int_to_base3(i, n)
        neighbors = {i}
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    neighbors.add(base3_to_int(new_digits))
        balls.append(list(neighbors))
    
    # Convert balls to sets for faster intersection/union checks if needed, 
    # but for cover_count logic, lists are better for numpy updates.
    # We'll use lists for numpy, and sets for logic checks if necessary.
    
    # cover_map[w] = list of codewords covering w
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    # Greedy Initialization
    def greedy_cover(seed):
        rng = random.Random(seed)
        covered = np.zeros(N, dtype=bool)
        cnt = np.array([ball_size] * N, dtype=np.int32)
        code = []
        uncovered_count = N
        while uncovered_count > 0:
            max_val = int(cnt.max())
            cands = np.flatnonzero(cnt == max_val)
            c = int(rng.choice(cands))
            code.append(c)
            ball_c = balls[c]
            newly = [w for w in ball_c if not covered[w]]
            if newly:
                covered[newly] = True
                uncovered_count -= len(newly)
                for w in newly:
                    cnt[cover_map[w]] -= 1
        return code

    # Local Search: 1-for-0 and 2-for-1
    def local_search(code, seed, time_limit):
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list
        
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
            
        code_set = set(code_list)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try 1-for-0 (Removal)
            # Find c such that all w in balls[c] have cover_count[w] > 1
            removal_found = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            for i in indices:
                c = code_list[i]
                # Check if removable
                removable = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        removable = False
                        break
                if removable:
                    # Remove it
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code_list.pop(i)
                    code_set.discard(c)
                    improved = True
                    removal_found = True
                    break
            
            if removal_found:
                continue
                
            # 2. Try 2-for-1 (Replacement)
            # We need to find c1, c2 and c_new such that:
            # Let U be the set of vertices covered ONLY by c1 or c2.
            # i.e., w in U iff w in balls[c1] U balls[c2] AND (cover_count[w] == 1)
            # Note: If w is in both balls, and cover_count[w] > 1, it's fine.
            # If w is in balls[c1] only, and cover_count[w] == 1, it's in U.
            # We need c_new to cover U.
            
            # To speed up, we can iterate over pairs.
            # But pairs are O(L^2). L is ~200. 40000 pairs.
            # For each pair, calculating U and checking candidates is expensive.
            
            # Optimization: 
            # Only consider pairs where c1 and c2 "overlap" significantly?
            # Or just random sampling of pairs.
            
            pair_found = False
            num_tries = 100
            for _ in range(num_tries):
                if time.time() > time_limit:
                    break
                if len(code_list) < 2:
                    break
                i, j = rng.sample(range(len(code_list)), 2)
                c1, c2 = code_list[i], code_list[j]
                
                # Calculate U: vertices in balls[c1] U balls[c2] with cover_count == 1
                # We must be careful: cover_count includes c1 and c2.
                # So for w in balls[c1], if cover_count[w] == 1, it means ONLY c1 covers it.
                # For w in balls[c2], if cover_count[w] == 1, it means ONLY c2 covers it.
                # If w is in both, and cover_count[w] == 2, it's fine (covered by both).
                # If w is in both, and cover_count[w] == 1, that's impossible if c1!=c2? 
                # No, if c1 and c2 are distinct, and w is in both, cover_count[w] >= 2.
                # So U = {w in balls[c1] | cover_count[w]==1} U {w in balls[c2] | cover_count[w]==1}
                
                U = set()
                for w in balls[c1]:
                    if cover_count[w] == 1:
                        U.add(w)
                for w in balls[c2]:
                    if cover_count[w] == 1:
                        U.add(w)
                        
                if not U:
                    # This is a 2-for-0 case, should have been caught by 1-for-0?
                    # No, 1-for-0 checks if a SINGLE codeword is redundant.
                    # If c1 and c2 are mutually redundant (each covers the other's unique parts? No, they don't have unique parts),
                    # then removing c1 is valid? 
                    # If U is empty, it means no vertex is uniquely covered by c1 or c2.
                    # So removing c1 is valid (all its vertices are covered by others).
                    # Why didn't 1-for-0 catch it? Maybe c1 was not checked, or order.
                    # In any case, we can remove c1 and c2.
                    # But let's stick to 2-for-1 logic. If U is empty, we can remove both.
                    # Let's just remove c1 for now, loop will handle c2 later.
                    # Actually, let's just mark improved and remove one.
                    for w in balls[c1]:
                        cover_count[w] -= 1
                    code_list.pop(i)
                    code_set.discard(c1)
                    improved = True
                    pair_found = True
                    break

                # We need a c_new that covers U.
                # c_new must be NOT in code_set.
                # Iterate through all possible c_new?
                # N=2187.
                # Check if U is subset of balls[c_new].
                
                # To make this faster, we can pre-compute for each w, the list of c covering it.
                # Intersection of cover_map[w] for all w in U gives candidates.
                
                if len(U) > 0:
                    # Start with all N, intersect with cover_map[w]
                    # This is O(|U| * avg_degree).
                    # |U| is small.
                    candidates = set(range(N))
                    # Actually, we can just iterate through cover_map of the first element in U
                    first_w = next(iter(U))
                    potential_cands = cover_map[first_w]
                    
                    valid_cand = None
                    for c_new in potential_cands:
                        if c_new in code_set:
                            continue
                        # Check if c_new covers all of U
                        ok = True
                        ball_new = balls[c_new]
                        # Check membership
                        # Since ball_new is a list, membership is O(15).
                        # U is a set.
                        for w in U:
                            if w not in ball_new:
                                ok = False
                                break
                        if ok:
                            valid_cand = c_new
                            break
                    
                    if valid_cand is not None:
                        # Perform swap: remove c1, c2, add valid_cand
                        # Update cover_count
                        for w in balls[c1]:
                            cover_count[w] -= 1
                        for w in balls[c2]:
                            cover_count[w] -= 1
                        for w in balls[valid_cand]:
                            cover_count[w] += 1
                        
                        code_list[i] = valid_cand
                        code_list.pop(j) # j < i? No, we need to handle indices carefully.
                        # Better: remove from list by value or index.
                        # Since we modified list, let's rebuild or be careful.
                        # Let's remove by index from end to start if i < j?
                        # Actually, we can just pop the values and re-add.
                        # Let's revert to a safer list manipulation.
                        
                        # Rollback cover_count and list changes to do it cleanly
                        for w in balls[valid_cand]:
                            cover_count[w] -= 1
                        for w in balls[c2]:
                            cover_count[w] += 1
                        for w in balls[c1]:
                            cover_count[w] += 1
                        
                        # Clean removal
                        code_list.remove(c1)
                        code_list.remove(c2)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        
                        # Add new
                        code_list.append(valid_cand)
                        code_set.add(valid_cand)
                        for w in balls[valid_cand]:
                            cover_count[w] += 1
                        
                        improved = True
                        pair_found = True
                        break

            if pair_found:
                continue

            # 3. Random Kick if stuck
            if time.time() < time_limit:
                # Remove a random codeword and add a random one to escape local optimum
                i = rng.randint(0, len(code_list) - 1)
                old_c = code_list[i]
                for w in balls[old_c]:
                    cover_count[w] -= 1
                code_set.discard(old_c)
                
                new_c = rng.randint(0, N - 1)
                while new_c in code_set:
                    new_c = rng.randint(0, N - 1)
                
                code_list[i] = new_c
                code_set.add(new_c)
                for w in balls[new_c]:
                    cover_count[w] += 1
                improved = True

        return code_list

    # Driver
    best_code = None
    best_len = float('inf')
    
    # Greedy restarts
    for seed in range(10):
        if time.time() > deadline - 30.0:
            break
        code = greedy_cover(seed)
        # Refine
        ls_code = local_search(code, seed, deadline - 20.0)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code

    # Final polish
    if best_code is not None and time.time() < deadline - 2.0:
        ls_code = local_search(best_code, 999, deadline - 1.0)
        if len(ls_code) < best_len:
            best_code = ls_code

    if best_code is None:
        best_code = greedy_cover(0)

    return [int_to_base3(c, n) for c in best_code]

def _generic(q, n, radius):
    N = q ** n

    def int_to_base(num, length, base):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % base
            num //= base
        return res

    balls = []
    for i in range(N):
        digits = int_to_base(i, n, q)
        neighbors = set([i])
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    val = 0
                    for d in new_digits:
                        val = val * q + d
                    neighbors.add(val)
        balls.append(list(neighbors))

    covered = set()
    code = []
    remaining = set(range(N))
    while remaining:
        best_c = None
        best_count = -1
        for c in remaining:
            count = len(balls[c] & remaining)
            if count > best_count:
                best_count = count
                best_c = c
        code.append(best_c)
        covered.update(balls[best_c])
        remaining -= set(balls[best_c])

    # light refinement
    cover_count = [0] * N
    for c in code:
        for w in balls[c]:
            cover_count[w] += 1
    changed = True
    while changed:
        changed = False
        for i in range(len(code)):
            c = code[i]
            if all(cover_count[w] > 1 for w in balls[c]):
                for w in balls[c]:
                    cover_count[w] -= 1
                code.pop(i)
                changed = True
                break

    return [int_to_base(c, n, q) for c in code]
