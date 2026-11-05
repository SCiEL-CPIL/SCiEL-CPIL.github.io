import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Specific to the task: q=3, n=8, radius=1
    # Total words = 3^8 = 6561
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
        
    # Precompute neighbor lists
    # neighbors[i] is a numpy array of indices j such that dist(i, j) <= radius
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
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
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
                potential[neighbors[x]] -= 1
            else:
                cov_count[x] += 1
                
    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size
    
    # --- Phase 2: Iterated Local Search ---
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)
    
    stuck_counter = 0
    two_one_stuck = 0
    
    # Helper to perform a 1-remove
    def try_remove(idx):
        nonlocal current_cov_count, current_is_code, current_code_set, current_code, current_size, best_size, best_code, stuck_counter
        c = current_code[idx]
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
            return True
        return False

    # Helper to perform a 1-swap
    def try_swap(idx):
        nonlocal current_cov_count, current_is_code, current_code_set, current_code, current_size, stuck_counter
        c1 = current_code[idx]
        risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]
        if not risky_points:
            return try_remove(idx)
        
        # Find c2 not in code such that c2 covers all risky points
        candidates = set(neighbors[risky_points[0]].tolist())
        candidates.discard(c1)
        
        for x in risky_points[1:]:
            candidates &= set(neighbors[x].tolist())
            if not candidates:
                return False
        
        valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]
        if not valid_candidates:
            return False
            
        c2 = random.choice(valid_candidates)
        
        # Perform swap
        current_cov_count[neighbors[c1]] -= 1
        current_is_code[c1] = False
        current_code_set.remove(c1)
        current_code[idx] = c2
        
        current_cov_count[neighbors[c2]] += 1
        current_is_code[c2] = True
        current_code_set.add(c2)
        
        stuck_counter = 0
        return True

    # Helper to perform a 2-1 move (remove c1, c4, add c2)
    def try_two_one(idx1, idx4):
        nonlocal current_cov_count, current_is_code, current_code_set, current_code, current_size, best_size, best_code, stuck_counter, two_one_stuck
        c1 = current_code[idx1]
        c4 = current_code[idx4]
        
        # Risky points if we remove c1 and c4
        # We need to find a c2 that covers all points that would become uncovered.
        # Points at risk are those in neighbors[c1] with count 1 (only covered by c1)
        # OR points in neighbors[c4] with count 1 (only covered by c4)
        # BUT if a point is covered by both c1 and c4, and count is 2, removing both makes it 0.
        # So risky points = {x in N(c1) U N(c4) | count[x] <= 1 (if x in N(c1) and x in N(c4) and count[x]==2, it becomes 0? No.
        
        # Let's simulate the removal counts.
        # temp_count = current_cov_count.copy() is too slow.
        # We can check:
        # A point x is at risk if:
        # 1. x in N(c1) and x in N(c4) and current_cov_count[x] == 2
        # 2. x in N(c1) and x not in N(c4) and current_cov_count[x] == 1
        # 3. x not in N(c1) and x in N(c4) and current_cov_count[x] == 1
        
        # Efficient check:
        # Let R1 = {x in N(c1) | count[x] == 1}
        # Let R4 = {x in N(c4) | count[x] == 1}
        # Let C14 = {x in N(c1) & N(c4) | count[x] == 2}
        # Risky = R1 U R4 U C14
        
        n_c1 = neighbors[c1]
        n_c4 = neighbors[c4]
        
        # Create a set of neighbors for fast lookup?
        # neighbors are small arrays (17 elements).
        
        risky = set()
        
        # Points covered only by c1 (among the two)
        for x in n_c1:
            if current_cov_count[x] == 1:
                risky.add(x)
            elif current_cov_count[x] == 2:
                # Check if x is also in n_c4
                if x in n_c4:
                    risky.add(x)
        
        # Points covered only by c4 (among the two)
        for x in n_c4:
            if current_cov_count[x] == 1:
                risky.add(x)
            # If count[x] == 2 and x in n_c1, it was already added above.
            # If count[x] > 2, removing 2 still leaves >= 1.
        
        if not risky:
            # Both are redundant
            # Remove c1
            current_cov_count[n_c1] -= 1
            current_is_code[c1] = False
            current_code_set.remove(c1)
            # Remove c4 (indices might shift if we pop, so handle carefully)
            # We are in a function, let's do it step by step
            # Pop c1
            current_code.pop(idx1)
            current_size -= 1
            # Adjust idx4 if necessary
            if idx4 > idx1:
                idx4 -= 1
            
            c4_val = current_code[idx4]
            n_c4_val = neighbors[c4_val]
            current_cov_count[n_c4_val] -= 1
            current_is_code[c4_val] = False
            current_code_set.remove(c4_val)
            current_code.pop(idx4)
            current_size -= 1
            
            stuck_counter = 0
            two_one_stuck = 0
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
            return True
        
        # Find c2 that covers all risky points
        # c2 must be in intersection of neighbors[x] for all x in risky
        candidates = set(neighbors[list(risky)[0]].tolist())
        candidates.discard(c1)
        candidates.discard(c4)
        
        for x in list(risky)[1:]:
            candidates &= set(neighbors[x].tolist())
            if not candidates:
                return False
        
        valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]
        if not valid_candidates:
            return False
            
        c2 = random.choice(valid_candidates)
        
        # Perform 2-1 move
        # Remove c1
        current_cov_count[n_c1] -= 1
        current_is_code[c1] = False
        current_code_set.remove(c1)
        current_code.pop(idx1)
        current_size -= 1
        
        # Adjust idx4
        if idx4 > idx1:
            idx4 -= 1
            
        # Remove c4
        c4_val = current_code[idx4]
        n_c4_val = neighbors[c4_val]
        current_cov_count[n_c4_val] -= 1
        current_is_code[c4_val] = False
        current_code_set.remove(c4_val)
        current_code.pop(idx4)
        current_size -= 1
        
        # Add c2
        current_cov_count[neighbors[c2]] += 1
        current_is_code[c2] = True
        current_code_set.add(c2)
        current_code.append(c2)
        current_size += 1
        
        stuck_counter = 0
        two_one_stuck = 0
        if current_size < best_size:
            best_size = current_size
            best_code = current_code[:]
        return True

    while time.time() < deadline - 10.0:
        # 1. Try to remove a word
        if current_size > 0:
            idx = random.randint(0, current_size - 1)
            if not try_remove(idx):
                stuck_counter += 1
                # 2. If stuck, try a swap
                if stuck_counter > 5:
                    if not try_swap(idx):
                        stuck_counter += 10
                        # 3. If still stuck, try a 2-1 move
                        if stuck_counter > 10 and current_size >= 2:
                            # Pick a random idx4 different from idx
                            idx4 = random.randint(0, current_size - 1)
                            if idx4 == idx:
                                idx4 = (idx4 + 1) % current_size
                            
                            if not try_two_one(idx, idx4):
                                two_one_stuck += 1
                                # If 2-1 fails repeatedly, reset counters to try different indices
                                if two_one_stuck > 20:
                                    stuck_counter = 0
                                    two_one_stuck = 0
                            else:
                                two_one_stuck = 0
                        else:
                            stuck_counter = 0
                    else:
                        stuck_counter = 0
                else:
                    stuck_counter = 0
            else:
                stuck_counter = 0
        else:
            break
            
    # --- Phase 3: Restart Polish ---
    # Use the best code found so far as a starting point for a more aggressive polish.
    # We perform a series of random restarts or aggressive local searches on the best code.
    
    # First, a deterministic removal pass
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True
                
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]
        
    # If we still have time, try a few random restarts from the best solution
    if time.time() < deadline - 15.0:
        # Try 3 random restarts
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
            # Limit restart time to 5 seconds each
            restart_deadline = time.time() + 5.0
            while time.time() < restart_deadline:
                if rc_size == 0:
                    break
                idx = random.randint(0, rc_size - 1)
                c = rc_code[idx]
                
                # Try remove
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
                    # Try swap
                    risky_points = [x for x in neighbors[c] if rc_cov[x] == 1]
                    if not risky_points:
                        # Should have been removed
                        pass
                    else:
                        candidates = set(neighbors[risky_points[0]].tolist())
                        candidates.discard(c)
                        for x in risky_points[1:]:
                            candidates &= set(neighbors[x].tolist())
                            if not candidates:
                                break
                        
                        valid_candidates = [c2 for c2 in candidates if c2 not in rc_set]
                        if valid_candidates:
                            c2 = random.choice(valid_candidates)
                            rc_cov[neighbors[c]] -= 1
                            rc_set.remove(c)
                            rc_code[idx] = c2
                            rc_cov[neighbors[c2]] += 1
                            rc_set.add(c2)
                            stuck = 0
                        else:
                            stuck = 0 # Reset to try again
                    stuck = 0

    # Final Polish on the absolute best
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1

    improved = True
    while improved and time.time() < deadline - 1.0:
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
