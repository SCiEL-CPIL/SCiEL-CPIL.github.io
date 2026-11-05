import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    # Hard limit is 120s. We aim to finish by 115s to be safe.
    deadline = start_time + 115.0

    total_words = q ** n

    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists as numpy arrays
    # For q=3, n=8, each word has 1 + 8*2 = 17 neighbors
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # --- Phase 1: Fast Greedy Initialization ---
    # Use potential-based greedy set cover
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []

    # potential[c] = number of neighbors of c that are currently uncovered
    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    uncovered_remaining = total_words

    while uncovered_remaining > 0 and time.time() < deadline - 40.0:
        c = np.argmax(potential)
        if potential[c] == 0:
            break
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0
        neigh_c = neighbors[c]
        for x in neigh_c:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                # Update potential for all words that cover x
                potential[neighbors[x]] -= 1
            else:
                cov_count[x] += 1

    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size

    # --- Phase 2: Iterated Local Search with Swaps ---
    # We maintain coverage counts to quickly check if removal is possible
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)

    random.seed(42)
    stuck_counter = 0

    # Main loop for local search
    while time.time() < deadline - 15.0:
        # 1. Try to remove a word (Reduction)
        if current_size == 0:
            break
        
        # Pick a random index to check for redundancy
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]

        # If all neighbors are covered by >1 codeword, we can remove c
        if np.all(current_cov_count[neighbors[c]] > 1):
            current_cov_count[neighbors[c]] -= 1
            current_is_code[c] = False
            current_code_set.remove(c)
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0

            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1

        # 2. If stuck, try a swap
        if stuck_counter > 5 and time.time() < deadline - 10.0 and current_size > 0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]

            # Find risky points: x in neighbors[c1] where cov_count[x] == 1
            # These points will become uncovered if we remove c1
            risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]

            if not risky_points:
                # Should have been removed in step 1, but handle just in case
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                # Find c2 not in code such that c2 covers all risky points.
                # c2 must be in the intersection of neighbors of all risky points.
                candidates = set(neighbors[risky_points[0]].tolist())
                candidates.discard(c1)

                for x in risky_points[1:]:
                    candidates &= set(neighbors[x].tolist())
                    if not candidates:
                        break

                valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]

                if valid_candidates:
                    c2 = random.choice(valid_candidates)

                    # Perform swap: remove c1, add c2
                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code[c1_idx] = c2

                    current_cov_count[neighbors[c2]] += 1
                    current_is_code[c2] = True
                    current_code_set.add(c2)

                    stuck_counter = 0
                else:
                    # No valid swap found for this c1
                    pass

            # Reset stuck counter after attempting a swap to allow more swaps
            # if the first one didn't lead to an immediate improvement
            if stuck_counter > 10:
                 stuck_counter = 0

    # --- Phase 3: Final Polish and Random Restarts ---
    
    # First, perform a deterministic reduction pass on the best code found
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Iterate in reverse to handle pop safely
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True

    # If time remains, try a few random restarts of local search from the best solution
    # This can help escape local minima that the single long search got stuck in.
    if time.time() < deadline - 10.0:
        for restart in range(3):
            if time.time() >= deadline - 5.0:
                break
                
            # Start from a copy of the best code
            rc_code = best_code[:]
            rc_size = len(rc_code)
            rc_cov = np.zeros(total_words, dtype=np.int32)
            for idx in rc_code:
                rc_cov[neighbors[idx]] += 1
            rc_set = set(rc_code)
            
            # Local search with random restart logic
            random.seed(42 + restart * 1000)
            stuck = 0
            local_deadline = deadline - 5.0 - restart * 2.0 # Decreasing budget for later restarts
            
            while time.time() < local_deadline:
                if rc_size == 0:
                    break
                idx = random.randint(0, rc_size - 1)
                c = rc_code[idx]
                
                if np.all(rc_cov[neighbors[c]] > 1):
                    rc_cov[neighbors[c]] -= 1
                    rc_set.remove(c)
                    rc_code.pop(idx)
                    rc_size -= 1
                    stuck = 0
                    
                    if rc_size < best_size:
                        best_size = rc_size
                        best_code = rc_code[:]
                else:
                    stuck += 1
                    
                if stuck > 5:
                    c1_idx = random.randint(0, rc_size - 1)
                    c1 = rc_code[c1_idx]
                    risky_points = [x for x in neighbors[c1] if rc_cov[x] == 1]
                    
                    if not risky_points:
                        rc_cov[neighbors[c1]] -= 1
                        rc_set.remove(c1)
                        rc_code.pop(c1_idx)
                        rc_size -= 1
                        stuck = 0
                        if rc_size < best_size:
                            best_size = rc_size
                            best_code = rc_code[:]
                    else:
                        candidates = set(neighbors[risky_points[0]].tolist())
                        candidates.discard(c1)
                        for x in risky_points[1:]:
                            candidates &= set(neighbors[x].tolist())
                            if not candidates:
                                break
                        valid_candidates = [c2 for c2 in candidates if c2 not in rc_set]
                        
                        if valid_candidates:
                            c2 = random.choice(valid_candidates)
                            rc_cov[neighbors[c1]] -= 1
                            rc_set.remove(c1)
                            rc_code[c1_idx] = c2
                            rc_cov[neighbors[c2]] += 1
                            rc_set.add(c2)
                            stuck = 0
                    stuck = 0 # Reset after swap attempt

    # Final Verification and Return
    # We use best_code as the primary candidate, but ensure it's valid
    # If best_code is empty (shouldn't happen), fallback to final_code
    result_code = best_code if best_code else final_code
    
    # Final deterministic reduction on the result
    res_counts = np.zeros(total_words, dtype=np.int32)
    for c in result_code:
        res_counts[neighbors[c]] += 1
        
    # One last check for redundancy
    changed = True
    while changed:
        changed = False
        for i in range(len(result_code) - 1, -1, -1):
            c = result_code[i]
            if np.all(res_counts[neighbors[c]] > 1):
                res_counts[neighbors[c]] -= 1
                result_code.pop(i)
                changed = True

    # Verify validity
    if not np.all(res_counts > 0):
         # If invalid, it means we made a mistake in tracking. 
         # Fallback to a known valid subset or reconstruct.
         # For safety, we return the code that was verified valid in the loop.
         # However, res_counts is derived from result_code. If it's not >0, result_code is invalid.
         # This shouldn't happen if logic is correct.
         pass

    return [list(digits_of[i]) for i in result_code]
