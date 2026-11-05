import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n  # 6561
    
    # Precompute digits for all words for final output
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbor lists as numpy arrays for fast indexing
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
    
    # Precompute neighbor sets for fast intersection (used in swap moves)
    neighbor_sets = [set(neigh.tolist()) for neigh in neighbors]
    
    def generate_greedy_code(seed=None):
        cov_count = np.zeros(total_words, dtype=np.int32)
        is_code = np.zeros(total_words, dtype=bool)
        code_list = []
        
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        # Potential: number of uncovered neighbors
        potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
        uncovered_remaining = total_words
        
        while uncovered_remaining > 0 and time.time() < deadline - 45.0:
            # Find word covering most uncovered points
            c = int(np.argmax(potential))
            if potential[c] == 0:
                break
                
            code_list.append(c)
            is_code[c] = True
            potential[c] = 0  # Prevent re-picking
            
            neigh_c = neighbors[c]
            for x in neigh_c:
                if cov_count[x] == 0:
                    cov_count[x] = 1
                    uncovered_remaining -= 1
                    # Update potentials of neighbors of x
                    potential[neighbors[x]] -= 1
                else:
                    cov_count[x] += 1
        return code_list, cov_count, is_code

    def perform_local_search(code, cov_count, is_code, max_time, seed=42):
        """Enhanced local search with deletion, single-swap, and two-swap moves."""
        end_time = time.time() + max_time
        rng = random.Random(seed)
        stuck_counter = 0
        best_code = code[:]
        best_size = len(code)
        best_cov_count = cov_count.copy()
        best_is_code = is_code.copy()
        
        current_code = code[:]
        current_size = len(current_code)
        
        while time.time() < end_time:
            if current_size == 0:
                break
            
            move_made = False
            
            # 1. Try Deletion
            idx = rng.randint(0, current_size - 1)
            c = current_code[idx]
            
            # Check if all neighbors are covered > 1 time
            if np.all(cov_count[neighbors[c]] > 1):
                cov_count[neighbors[c]] -= 1
                is_code[c] = False
                current_code.pop(idx)
                current_size -= 1
                stuck_counter = 0
                move_made = True
                
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
                    best_cov_count = cov_count.copy()
                    best_is_code = is_code.copy()
                continue

            # 2. Single Swap: remove c, add c2
            neigh_c = neighbors[c]
            risky_mask = cov_count[neigh_c] == 1
            risky_points = neigh_c[risky_mask]
            
            if len(risky_points) > 0:
                # Find candidates that cover all risky points
                candidates = neighbor_sets[risky_points[0].item()].copy()
                for r in risky_points[1:]:
                    candidates &= neighbor_sets[r.item()]
                    if not candidates:
                        break
                
                if candidates:
                    # Exclude current code words
                    code_set = set(current_code)
                    valid_candidates = [c2 for c2 in candidates if c2 not in code_set]
                    
                    if valid_candidates:
                        c2 = rng.choice(valid_candidates)
                        
                        # Perform swap
                        cov_count[neighbors[c]] -= 1
                        is_code[c] = False
                        cov_count[neighbors[c2]] += 1
                        is_code[c2] = True
                        current_code[idx] = c2
                        stuck_counter = 0
                        move_made = True
                        continue
            
            # 3. Two-Swap: remove 2 codewords, add 1
            if not move_made and current_size >= 2 and stuck_counter > 5:
                i1 = rng.randint(0, current_size - 1)
                i2 = rng.randint(0, current_size - 1)
                if i2 == i1:
                    i2 = (i2 + 1) % current_size
                
                c1 = current_code[i1]
                c2 = current_code[i2]
                
                # Find points that become uncovered if we remove both c1 and c2
                risky = set()
                for x in neighbors[c1]:
                    if cov_count[x] == 1:
                        risky.add(x)
                for x in neighbors[c2]:
                    if cov_count[x] == 1:
                        risky.add(x)
                
                if not risky:
                    # Both are removable, remove one
                    for x in neighbors[c1]:
                        cov_count[x] -= 1
                    is_code[c1] = False
                    if i1 < i2:
                        current_code.pop(i1)
                        current_code.pop(i2 - 1)
                    else:
                        current_code.pop(i2)
                        current_code.pop(i1)
                    current_size -= 2
                    stuck_counter = 0
                    move_made = True
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                        best_cov_count = cov_count.copy()
                        best_is_code = is_code.copy()
                    continue
                else:
                    # Find candidate c3 that covers all risky points
                    risky_list = list(risky)
                    candidates = neighbor_sets[risky_list[0]].copy()
                    for r in risky_list[1:]:
                        candidates &= neighbor_sets[r]
                        if not candidates:
                            break
                    
                    if candidates:
                        code_set = set(current_code)
                        valid_candidates = [c3 for c3 in candidates if c3 not in code_set]
                        
                        if valid_candidates:
                            c3 = rng.choice(valid_candidates)
                            
                            # Remove c1 and c2, add c3
                            cov_count[neighbors[c1]] -= 1
                            cov_count[neighbors[c2]] -= 1
                            is_code[c1] = False
                            is_code[c2] = False
                            cov_count[neighbors[c3]] += 1
                            is_code[c3] = True
                            
                            # Update current_code
                            if i1 < i2:
                                current_code.pop(i2)
                                current_code.pop(i1)
                            else:
                                current_code.pop(i1)
                                current_code.pop(i2)
                            current_code.append(c3)
                            current_size -= 1
                            stuck_counter = 0
                            move_made = True
                            
                            if current_size < best_size:
                                best_size = current_size
                                best_code = current_code[:]
                                best_cov_count = cov_count.copy()
                                best_is_code = is_code.copy()
                            continue
            
            if not move_made:
                stuck_counter += 1
                # If stuck for too long, try a random perturbation: add a random codeword
                if stuck_counter > 50 and current_size < 600:
                    random_c = rng.randint(0, total_words - 1)
                    if random_c not in set(current_code):
                        cov_count[neighbors[random_c]] += 1
                        is_code[random_c] = True
                        current_code.append(random_c)
                        current_size += 1
                        stuck_counter = 0
        
        return best_code, best_cov_count, best_is_code

    # Run multiple restarts with different seeds
    best_overall = None
    best_overall_size = float('inf')
    
    # Strategy: Try several restarts, each with greedy + local search
    # Allocate time: ~15s per restart for greedy, rest for local search
    num_restarts = 4
    seeds = [42, 123, 456, 789]
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        seed = seeds[restart % len(seeds)]
        
        # Generate greedy code
        code, cov_count, is_code = generate_greedy_code(seed)
        
        if not code:
            continue
        
        # Time remaining for local search
        time_left = deadline - time.time() - 5.0  # 5s safety margin
        if time_left < 5.0:
            break
        
        # Perform local search
        code, cov_count, is_code = perform_local_search(code, cov_count, is_code, time_left, seed=seed)
        
        # Verify and update best
        if len(code) < best_overall_size:
            # Verify coverage
            final_cov = np.zeros(total_words, dtype=np.int32)
            for c in code:
                final_cov[neighbors[c]] += 1
            
            if np.all(final_cov > 0):
                best_overall = code
                best_overall_size = len(code)
    
    # If no valid solution found, use last greedy
    if best_overall is None:
        code, _, _ = generate_greedy_code(42)
        best_overall = code
    
    return [list(digits_of[i]) for i in best_overall]
