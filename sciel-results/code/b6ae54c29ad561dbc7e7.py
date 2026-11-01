"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def construct(q, n, radius):
    # For the specific task: q=3, n=7, radius=1
    # We use a refined Simulated Annealing approach with Restarts.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precomputation for q=3, n=7, r=1
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # neighbors[i] is a frozenset of indices of words within distance 1 of all_words[i]
    # Using frozenset allows fast 'in' checks and prevents modification
    neighbors = []
    for w in all_words:
        w_list = list(w)
        idxs = [word_to_idx[w]]
        for i in range(7):
            orig = w_list[i]
            for v in range(3):
                if v != orig:
                    w_list[i] = v
                    idxs.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig
        neighbors.append(frozenset(idxs))
        
    random.seed(42)
    
    # 1. Greedy Initialization: Max Coverage Greedy
    counts = [0] * 2187
    covered = 0
    code = []
    uncovered_set = set(range(2187))
    
    while uncovered_set:
        best_word_idx = -1
        max_cov = 0
        
        for w_idx in range(2187):
            cov = 0
            for nb in neighbors[w_idx]:
                if nb in uncovered_set:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_word_idx = w_idx
        
        if best_word_idx == -1:
            break
            
        code.append(all_words[best_word_idx])
        for nb in neighbors[best_word_idx]:
            if counts[nb] == 0:
                covered += 1
                uncovered_set.remove(nb)
            counts[nb] += 1
            
    P = 5.0
    
    def get_cost(size, covered):
        return size + P * (2187 - covered)
        
    best_code = list(code)
    best_size = len(code)
    best_covered = covered
    best_cost = get_cost(best_size, best_covered)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Iterated Simulated Annealing
    # We run SA in phases. If we don't improve, we restart with perturbation.
    phase_time = 5.0  # seconds per phase
    max_phases = int((time_limit - 5.0) / phase_time) + 1 # Reserve 5s for greedy
    
    for phase in range(max_phases):
        if time.time() - start_time > time_limit - 2.0:
            break
            
        # Initialize current state from best found so far
        # Perturb the best solution to escape local minima
        current_code = list(best_code)
        current_counts = [0] * 2187
        for w in current_code:
            w_idx = word_to_idx[w]
            for nb in neighbors[w_idx]:
                current_counts[nb] += 1
        
        current_covered = 0
        for c in current_counts:
            if c > 0:
                current_covered += 1
        current_size = len(current_code)
        current_cost = get_cost(current_size, current_covered)
        
        # Apply random perturbation to current_code to start from a new local minimum
        # Remove a few random words and add random ones to break symmetry/stuck states
        if phase > 0:
            # Perturb by removing 2-5 words
            num_removals = random.randint(2, 5)
            for _ in range(num_removals):
                if current_code:
                    idx = random.randint(0, len(current_code) - 1)
                    w = current_code[idx]
                    w_idx = word_to_idx[w]
                    for nb in neighbors[w_idx]:
                        current_counts[nb] -= 1
                    current_code.pop(idx)
                    current_size -= 1
                    # Recalculate covered? No, we can just update current_covered later or keep it approximate
                    # It's safer to recalculate covered from counts for accuracy after perturbation
            # Recalculate covered
            current_covered = 0
            for c in current_counts:
                if c > 0:
                    current_covered += 1
            current_cost = get_cost(current_size, current_covered)
            
        T = 10.0
        T_min = 0.1
        cooling = 0.9995
        
        # Inner SA Loop for this phase
        phase_start = time.time()
        while time.time() - phase_start < phase_time:
            if time.time() - start_time > time_limit - 1.0:
                break
                
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to Remove
                if not current_code:
                    continue
                idx = random.randint(0, current_size - 1)
                w = current_code[idx]
                w_idx = word_to_idx[w]
                
                delta_covered = 0
                for nb in neighbors[w_idx]:
                    if current_counts[nb] == 1:
                        delta_covered -= 1
                
                if delta_covered == 0:
                    current_code.pop(idx)
                    for nb in neighbors[w_idx]:
                        current_counts[nb] -= 1
                    current_size -= 1
                    current_cost = get_cost(current_size, current_covered)
                    
                    if current_size < best_size and current_covered == 2187:
                        best_size = current_size
                        best_covered = current_covered
                        best_code = list(current_code)
                        best_cost = current_cost
                        
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                else:
                    new_size = current_size - 1
                    new_covered = current_covered + delta_covered
                    new_cost = get_cost(new_size, new_covered)
                    delta_cost = new_cost - current_cost
                    
                    if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                        current_code.pop(idx)
                        for nb in neighbors[w_idx]:
                            current_counts[nb] -= 1
                        current_size = new_size
                        current_covered = new_covered
                        current_cost = new_cost
                        
                        if current_covered == 2187 and current_size < best_size:
                            best_size = current_size
                            best_covered = current_covered
                            best_code = list(current_code)
                            best_cost = current_cost
                            
                        T *= cooling
                        if T < T_min: T = T_min
                        continue
                    else:
                        T *= cooling
                        if T < T_min: T = T_min
                        continue

            elif move_type < 0.9:
                # Replace
                if not current_code:
                    continue
                idx = random.randint(0, current_size - 1)
                old_w = current_code[idx]
                old_w_idx = word_to_idx[old_w]
                
                u_idx = random.randint(0, 2186)
                new_w_idx = random.choice(list(neighbors[u_idx]))
                new_w = all_words[new_w_idx]
                
                if new_w_idx == old_w_idx:
                    continue
                    
                old_nbs = neighbors[old_w_idx]
                new_nbs = neighbors[new_w_idx]
                
                delta_covered = 0
                
                # Check neighbors of old that are NOT in new
                # Since they are frozensets, 'in' is fast
                for nb in old_nbs:
                    if nb not in new_nbs:
                        if current_counts[nb] == 1:
                            delta_covered -= 1
                            
                # Check neighbors of new that are NOT in old
                for nb in new_nbs:
                    if nb not in old_nbs:
                        if current_counts[nb] == 0:
                            delta_covered += 1
                            
                new_size = current_size
                new_covered = current_covered + delta_covered
                new_cost = get_cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    current_code[idx] = new_w
                    for nb in old_nbs:
                        current_counts[nb] -= 1
                    for nb in new_nbs:
                        current_counts[nb] += 1
                    current_covered = new_covered
                    current_cost = new_cost
                    
                    if current_covered == 2187 and current_size < best_size:
                        best_size = current_size
                        best_covered = current_covered
                        best_code = list(current_code)
                        best_cost = current_cost
                        
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue

            else:
                # Add
                if current_covered == 2187:
                    continue
                
                u_idx = random.randint(0, 2186)
                if current_counts[u_idx] == 0:
                    new_w_idx = random.choice(list(neighbors[u_idx]))
                else:
                    new_w_idx = random.randint(0, 2186)
                    
                new_w = all_words[new_w_idx]
                
                if new_w in current_code:
                    continue
                
                new_nbs = neighbors[new_w_idx]
                delta_covered = 0
                for nb in new_nbs:
                    if current_counts[nb] == 0:
                        delta_covered += 1
                
                new_size = current_size + 1
                new_covered = current_covered + delta_covered
                new_cost = get_cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    current_code.append(new_w)
                    for nb in new_nbs:
                        current_counts[nb] += 1
                    current_size = new_size
                    current_covered = new_covered
                    current_cost = new_cost
                    
                    if current_covered == 2187 and current_size < best_size:
                        best_size = current_size
                        best_covered = current_covered
                        best_code = list(current_code)
                        best_cost = current_cost
                        
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue

    return [list(w) for w in best_code]
# EVOLVE-BLOCK-END
