import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Fallback for other parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each index
    # neighbors[idx] is a list of 17 indices
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
            
        neighbors[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]
            
            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

    # Build reverse mapping: for each point, which balls contain it
    # rev_neighbors[p] is a list of ball indices b such that p is in neighbors[b]
    rev_neighbors = [[] for _ in range(total_points)]
    for b in range(total_points):
        for p in neighbors[b]:
            rev_neighbors[p].append(b)
    # Convert to numpy arrays for faster access
    rev_neighbors_arr = []
    for p in range(total_points):
        rev_neighbors_arr.append(np.array(rev_neighbors[p], dtype=np.int32))

    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def fast_greedy_init(deadline):
        """
        Vectorized greedy initialization.
        Maintains a 'potential' array where potential[b] is the number of uncovered points in ball b.
        """
        is_covered = np.zeros(total_points, dtype=bool)
        potential = np.full(total_points, ball_size, dtype=np.int32)
        selected = []
        uncovered_count = total_points
        
        while uncovered_count > 0 and time.time() < deadline - 40.0:
            # Find ball with max potential
            best_idx = np.argmax(potential)
            if potential[best_idx] == 0:
                break
                
            selected.append(best_idx)
            
            # Update coverage and potentials
            ball = neighbors[best_idx]
            for p in ball:
                if not is_covered[p]:
                    is_covered[p] = True
                    uncovered_count -= 1
                    # Decrement potential for all balls covering p
                    potential[rev_neighbors_arr[p]] -= 1
            
        return selected

    def prune(selected_indices):
        """Efficient pruning: remove redundant codewords"""
        if not selected_indices:
            return []
            
        selected_list = list(selected_indices)
        if len(selected_list) == 0:
            return []
        
        # Build coverage count array
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
        
        # Iteratively remove redundant codewords
        changed = True
        while changed:
            changed = False
            # We iterate backwards to safely pop
            for i in range(len(selected_list) - 1, -1, -1):
                idx = selected_list[i]
                ball = neighbors[idx]
                # If all points in this ball have count > 1, it's redundant
                if np.all(coverage_count[ball] > 1):
                    coverage_count[ball] -= 1
                    selected_list.pop(i)
                    changed = True
        
        return selected_list

    def local_search(selected_indices, deadline, seed=0):
        """
        Local search with swaps and removals.
        1. Try to remove a codeword if safe.
        2. If stuck, try to swap a codeword with another that covers its 'risky' points.
        3. If still stuck, perform a random kick (perturbation) and restart local search.
        """
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(seed)
        selected_set = set(selected_list)
        
        stuck_counter = 0
        max_stuck = 200 # Threshold for kicking
        
        while time.time() < deadline - 5.0:
            improvement_made = False
            current_len = len(selected_list)
            
            # 1. Try removals
            # We try a few random removals
            attempts = 10
            for _ in range(attempts):
                if current_len == 0:
                    break
                c = selected_list[rng.integers(current_len)]
                
                # Check if removing c leaves any point uncovered
                # risky points are those with count == 1
                ball = neighbors[c]
                risky = ball[coverage_count[ball] == 1]
                
                if len(risky) == 0:
                    # Safe to remove
                    coverage_count[ball] -= 1
                    idx_in_list = selected_list.index(c)
                    selected_list.pop(idx_in_list)
                    selected_set.remove(c)
                    current_len -= 1
                    improvement_made = True
                    stuck_counter = 0
                    if current_len < best_len:
                        best_len = current_len
                        best_code = selected_list[:]
                    break
                else:
                    # Try to find a swap: find c' not in selected_set that covers all risky points
                    # and ideally covers some other points to maintain coverage or allow future removals
                    # Simple heuristic: c' must cover all risky points.
                    # We check candidates from the neighborhood of the first risky point
                    candidate_indices = set(rev_neighbors_arr[risky[0]].tolist())
                    candidate_indices.discard(c)
                    
                    valid_candidate = -1
                    for c_prime in candidate_indices:
                        if c_prime in selected_set:
                            continue
                        # Check if c_prime covers all risky points
                        if np.all(np.isin(risky, neighbors[c_prime])):
                            valid_candidate = c_prime
                            break
                    
                    if valid_candidate != -1:
                        # Perform swap
                        coverage_count[neighbors[c]] -= 1
                        coverage_count[neighbors[valid_candidate]] += 1
                        
                        idx_in_list = selected_list.index(c)
                        selected_list[idx_in_list] = valid_candidate
                        selected_set.remove(c)
                        selected_set.add(valid_candidate)
                        
                        # A swap doesn't reduce size, but it might enable a removal later.
                        # We treat it as a "move" that resets the stuck counter but doesn't update best_len yet.
                        # However, often a swap allows an immediate removal.
                        # Let's try to remove immediately after swap? No, that's complex.
                        # Just reset stuck counter.
                        stuck_counter = 0
                        improvement_made = True
                        break
            
            if not improvement_made:
                stuck_counter += 1
                
                # 2. If stuck too long, perform a kick (perturbation)
                if stuck_counter > max_stuck:
                    # Remove a random codeword and add a random new one to escape local minima
                    if current_len > 0:
                        c_out = selected_list[rng.integers(current_len)]
                        coverage_count[neighbors[c_out]] -= 1
                        idx_in_list = selected_list.index(c_out)
                        selected_list.pop(idx_in_list)
                        selected_set.remove(c_out)
                        current_len -= 1
                        
                        # Add a random new codeword
                        c_in = rng.integers(total_points)
                        while c_in in selected_set:
                            c_in = rng.integers(total_points)
                        
                        coverage_count[neighbors[c_in]] += 1
                        selected_list.append(c_in)
                        selected_set.add(c_in)
                        current_len += 1
                        
                    stuck_counter = 0
                    max_stuck = int(max_stuck * 1.5) # Increase threshold slightly
                    if max_stuck > 1000: max_stuck = 1000

        return best_code

    start_time = time.time()
    # 150s limit, keep 5s buffer for safety/conversion
    deadline = start_time + 145.0
    
    # Phase 1: Initial Solution
    # Use fast greedy
    initial_code = fast_greedy_init(deadline)
    initial_code = prune(initial_code)
    
    best_code = initial_code
    best_len = len(initial_code)
    
    # Phase 2: Local Search with Restarts
    # We run local search multiple times. If it gets stuck, it kicks.
    # We also do a few full restarts of the local search phase from the best solution found.
    
    # Run 1: Standard local search
    if time.time() < deadline - 10.0:
        code1 = local_search(best_code, deadline, seed=42)
        if len(code1) < best_len:
            best_len = len(code1)
            best_code = code1
    
    # Run 2: Another local search, maybe with different seed
    if time.time() < deadline - 10.0:
        code2 = local_search(best_code, deadline, seed=123)
        if len(code2) < best_len:
            best_len = len(code2)
            best_code = code2
            
    # Run 3: If time permits, try a "greedy repair" from a slightly perturbed best_code
    # Or just run local search again.
    if time.time() < deadline - 5.0:
        code3 = local_search(best_code, deadline, seed=999)
        if len(code3) < best_len:
            best_len = len(code3)
            best_code = code3

    # Final Prune
    best_code = prune(best_code)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
