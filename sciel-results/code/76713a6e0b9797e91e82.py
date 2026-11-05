import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0
    
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
    
    # Precompute neighbors as a 2D array for vectorized operations
    neigh_size = len(neighbors[0])
    neighbors_2d = np.array([neighbors[i] for i in range(total_words)], dtype=np.int32)
    
    # --- Phase 1: Greedy Initialization ---
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    potential = np.full(total_words, neigh_size, dtype=np.int32)
    uncovered_remaining = total_words
    
    while uncovered_remaining > 0 and time.time() < deadline - 40.0:
        c = np.argmax(potential)
        if potential[c] == 0:
            break
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0
        neigh_c = neighbors[c]
        newly_covered = neigh_c[cov_count[neigh_c] == 0]
        if len(newly_covered) > 0:
            cov_count[newly_covered] += 1
            uncovered_remaining -= len(newly_covered)
            # Update potential for all neighbors of newly covered points
            for x in newly_covered:
                potential[neighbors[x]] -= 1
        else:
            cov_count[neigh_c] += 1
    
    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size
    
    # --- Phase 2: Iterated Local Search with improved swaps ---
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)
    
    stuck_counter = 0
    
    while time.time() < deadline - 8.0:
        # Try to remove a word - try multiple random picks
        found_removal = False
        for _ in range(10):
            idx = random.randint(0, current_size - 1)
            c = current_code[idx]
            
            if np.all(current_cov_count[neighbors[c]] > 1):
                current_cov_count[neighbors[c]] -= 1
                current_is_code[c] = False
                current_code_set.remove(c)
                current_code.pop(idx)
                current_size -= 1
                stuck_counter = 0
                found_removal = True
                
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
                break
        
        if found_removal:
            continue
            
        # If stuck, try a swap
        stuck_counter += 1
        if stuck_counter > 20 and time.time() < deadline - 12.0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            
            # Find risky points: x in neighbors[c1] where cov_count[x] == 1
            risky_mask = current_cov_count[neighbors[c1]] == 1
            risky_points = neighbors[c1][risky_mask]
            
            if len(risky_points) == 0:
                # Shouldn't happen if we checked removal, but handle it
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
                # Find c2 not in code such that c2 covers all risky points
                # c2 must be in intersection of neighbors[x] for all x in risky_points
                # Use set intersection
                candidates = set(neighbors[risky_points[0]].tolist())
                for x in risky_points[1:]:
                    candidates &= set(neighbors[x].tolist())
                    if not candidates:
                        break
                
                # Filter out candidates already in code
                valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]
                
                if valid_candidates:
                    c2 = random.choice(valid_candidates)
                    
                    # Perform swap
                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code[c1_idx] = c2
                    
                    current_cov_count[neighbors[c2]] += 1
                    current_is_code[c2] = True
                    current_code_set.add(c2)
                    
                    stuck_counter = 0
                else:
                    # No valid swap, reset to try removals again
                    stuck_counter = 0
    
    # Final Polish: exhaustive removal pass
    final_code = best_code[:]
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
                
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
