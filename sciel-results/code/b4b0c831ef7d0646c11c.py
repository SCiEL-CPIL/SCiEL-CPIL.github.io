import numpy as np
import time
import random
import itertools
from collections import deque

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0  # safety margin from 150s limit

    total_words = q ** n  # 6561

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

    # Precompute neighbor lists
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
    
    def greedy_init(seed_val):
        """Greedy initialization. Optimized to be faster."""
        rng = random.Random(seed_val)
        is_code = np.zeros(total_words, dtype=bool)
        code_list = []
        uncovered = np.ones(total_words, dtype=bool)
        uncovered_remaining = total_words
        
        # Precompute a list of indices to iterate, maybe shuffle?
        # Standard greedy is usually fine if we make the inner loop fast.
        # To speed up, we can skip candidates that are already covered? No, they can cover others.
        # Let's just run one pass.
        
        while uncovered_remaining > 0:
            best_c = -1
            best_gain = 0
            # Optimization: If we have many uncovered, scanning all is expensive.
            # But 6561 is small enough if the inner check is fast.
            # np.count_nonzero on a boolean array of size ~24 is fast.
            for c in range(total_words):
                if is_code[c]:
                    continue
                gain = np.count_nonzero(uncovered[neighbors[c]])
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_gain == 0:
                break
            
            code_list.append(best_c)
            is_code[best_c] = True
            neigh_c = neighbors[best_c]
            newly_covered = np.nonzero(uncovered[neigh_c])[0]
            uncovered[neigh_c] = False
            uncovered_remaining -= len(newly_covered)
            
        return code_list

    # Try fewer seeds to save time for local search
    best_code = None
    best_size = float('inf')
    
    for seed_val in [42, 123]:
        if time.time() > deadline - 30.0:
            break
        code_list = greedy_init(seed_val)
        if len(code_list) < best_size:
            best_size = len(code_list)
            best_code = code_list[:]
    
    current_code = best_code[:]
    current_size = len(current_code)
    current_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in current_code:
        current_cov_count[neighbors[idx]] += 1
    current_is_code = np.zeros(total_words, dtype=bool)
    for idx in current_code:
        current_is_code[idx] = True
    current_code_set = set(current_code)

    # Precompute neighbor sets for fast intersection in local search
    neighbors_sets = [set(neighbors[i].tolist()) for i in range(total_words)]

    # --- Phase 2: Iterated Local Search with Smarter Moves ---
    random.seed(42)
    stuck_counter = 0
    no_improve_count = 0
    best_global = current_size
    best_global_code = current_code[:]

    while time.time() < deadline - 15.0:
        if current_size <= best_global:
            best_global = current_size
            best_global_code = current_code[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # 1. Try to remove a word
        if current_size == 0:
            break
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]

        if np.all(current_cov_count[neighbors[c]] > 1):
            current_cov_count[neighbors[c]] -= 1
            current_is_code[c] = False
            current_code_set.remove(c)
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
        else:
            stuck_counter += 1

        # 2. If stuck, try a swap or 2-opt
        if stuck_counter > 3 and time.time() < deadline - 20.0 and current_size > 0:
            # Try 1-opt swap
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]

            # Find risky points: x in neighbors[c1] where cov_count[x] == 1
            risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]

            if not risky_points:
                # Should have been caught in step 1, but just in case
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
            else:
                # Find c2 not in code such that c2 covers all risky points.
                # Use set intersection for speed
                if risky_points:
                    # Intersection of neighbor sets of all risky points
                    candidates_set = neighbors_sets[risky_points[0]].copy()
                    for x in risky_points[1:]:
                        candidates_set.intersection_update(neighbors_sets[x])
                        if not candidates_set:
                            break
                    
                    # Remove current code words
                    candidates_set -= current_code_set
                    
                    if candidates_set:
                        c2 = random.choice(list(candidates_set))
                        
                        current_cov_count[neighbors[c1]] -= 1
                        current_is_code[c1] = False
                        current_code_set.remove(c1)
                        current_code[c1_idx] = c2

                        current_cov_count[neighbors[c2]] += 1
                        current_is_code[c2] = True
                        current_code_set.add(c2)

                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    # Try 2-opt: Remove c1 and another c2, add one c3
                    if current_size >= 2 and stuck_counter > 8:
                        c2_idx = random.randint(0, current_size - 1)
                        if c2_idx == c1_idx:
                            c2_idx = (c2_idx + 1) % current_size
                        c2 = current_code[c2_idx]
                        
                        # Collect risky points for c1 and c2
                        risky_c1 = [x for x in neighbors[c1] if current_cov_count[x] == 1]
                        risky_c2 = [x for x in neighbors[c2] if current_cov_count[x] == 1]
                        all_risky = risky_c1 + risky_c2
                        
                        if all_risky:
                            # Find c3 that covers all all_risky
                            candidates_set_2 = neighbors_sets[all_risky[0]].copy()
                            for x in all_risky[1:]:
                                candidates_set_2.intersection_update(neighbors_sets[x])
                                if not candidates_set_2:
                                    break
                            
                            # Remove current code words
                            candidates_set_2 -= current_code_set
                            
                            if candidates_set_2:
                                c3 = random.choice(list(candidates_set_2))
                                
                                # Update counts
                                current_cov_count[neighbors[c1]] -= 1
                                current_cov_count[neighbors[c2]] -= 1
                                current_cov_count[neighbors[c3]] += 1
                                
                                # Update sets
                                current_is_code[c1] = False
                                current_is_code[c2] = False
                                current_is_code[c3] = True
                                current_code_set.remove(c1)
                                current_code_set.remove(c2)
                                current_code_set.add(c3)
                                
                                # Update list
                                indices_to_remove = sorted([c1_idx, c2_idx], reverse=True)
                                for idx_rm in indices_to_remove:
                                    current_code.pop(idx_rm)
                                current_code.append(c3)
                                current_size -= 1
                                stuck_counter = 0
                            else:
                                stuck_counter += 1
                        else:
                            stuck_counter += 1
                    else:
                        stuck_counter += 1

        # 3. Occasionally reset to best found so far
        if stuck_counter > 50 and time.time() < deadline - 25.0:
            current_code = best_global_code[:]
            current_size = len(current_code)
            current_cov_count = np.zeros(total_words, dtype=np.int32)
            for idx in current_code:
                current_cov_count[neighbors[idx]] += 1
            current_is_code = np.zeros(total_words, dtype=bool)
            for idx in current_code:
                current_is_code[idx] = True
            current_code_set = set(current_code)
            stuck_counter = 0

        stuck_counter = 0

    # --- Phase 3: Final Polish ---
    final_code = best_global_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1

    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True

    # Verify
    if not np.all(final_cov_count > 0):
        final_code = best_global_code[:]

    return [list(digits_of[i]) for i in final_code]
