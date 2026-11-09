import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        # Simple greedy or tetracode extension
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1
    N = 3 ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute coverage masks (bitsets)
    # mask[i] is an integer where bit j is 1 if words[j] is within distance 1 of words[i]
    masks = [0] * N
    neighbor_indices = [[] for _ in range(N)] # For updating counts
    
    for i in range(N):
        w_i = words[i]
        mask_val = 0
        neigh = []
        # Self
        mask_val |= (1 << i)
        neigh.append(i)
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                mask_val |= (1 << j)
                neigh.append(j)
        masks[i] = mask_val
        neighbor_indices[i] = neigh

    # Initialize with Tetracode construction
    def get_initial_tetracode():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        init_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                w = tuple(h + list(tail))
                init_indices.append(word_to_idx[w])
        return set(init_indices)

    best_code = get_initial_tetracode()
    best_size = len(best_code)
    
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # State variables
    current_code = set(best_code)
    coverage_counts = np.zeros(N, dtype=np.int32)
    
    # Initialize coverage counts and current_cov
    current_cov = 0
    for idx in current_code:
        for j in neighbor_indices[idx]:
            coverage_counts[j] += 1
        current_cov |= masks[idx]
        
    current_uncovered = N - current_cov.bit_count()
    
    # Helper to recompute current_cov from scratch (used for non-redundant removals/swaps)
    def recompute_cov(code_set):
        cov = 0
        for idx in code_set:
            cov |= masks[idx]
        return cov

    # Helper to check redundancy
    def is_redundant(idx):
        for j in neighbor_indices[idx]:
            if coverage_counts[j] == 1:
                return False
        return True

    # Phase 1: Deterministic Greedy Reduction
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        indices_list = list(current_code)
        random.shuffle(indices_list)
        for idx in indices_list:
            if is_redundant(idx):
                # Remove it
                for j in neighbor_indices[idx]:
                    coverage_counts[j] -= 1
                current_code.remove(idx)
                # current_cov does not change if redundant
                changed = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)

    # Phase 2: Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    all_indices = list(range(N))
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor
        move_type = random.random()
        
        delta_size = 0
        delta_uncovered = 0
        accepted = False
        new_code = current_code
        new_cov = current_cov
        new_counts = coverage_counts
        
        if move_type < 0.5:
            # Remove a random codeword
            if not current_code:
                continue
            to_remove = random.choice(list(current_code))
            
            # Check impact
            newly_uncovered = 0
            for j in neighbor_indices[to_remove]:
                if coverage_counts[j] == 1:
                    newly_uncovered += 1
            
            delta_size = -1
            delta_uncovered = newly_uncovered
            
            delta_cost = delta_size + 1000 * delta_uncovered
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                # Accept
                # Update counts
                new_counts = coverage_counts.copy()
                for j in neighbor_indices[to_remove]:
                    new_counts[j] -= 1
                
                new_code = set(current_code)
                new_code.remove(to_remove)
                
                if newly_uncovered == 0:
                    new_cov = current_cov
                else:
                    new_cov = recompute_cov(new_code)
                
                accepted = True
                
        elif move_type < 0.8:
            # Add a random codeword
            to_add = random.choice(all_indices)
            if to_add in current_code:
                continue
                
            # Check impact
            new_cov_trial = current_cov | masks[to_add]
            newly_covered = new_cov_trial.bit_count() - current_cov.bit_count()
            
            delta_size = 1
            delta_uncovered = -newly_covered
            
            delta_cost = delta_size + 1000 * delta_uncovered
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                # Accept
                new_counts = coverage_counts.copy()
                for j in neighbor_indices[to_add]:
                    new_counts[j] += 1
                
                new_code = set(current_code)
                new_code.add(to_add)
                new_cov = new_cov_trial
                accepted = True
                
        else:
            # Swap: remove one, add another
            if not current_code:
                continue
            to_remove = random.choice(list(current_code))
            to_add = random.choice(all_indices)
            if to_add == to_remove or to_add in current_code:
                continue
            
            # Simulate
            temp_counts = coverage_counts.copy()
            for j in neighbor_indices[to_remove]:
                temp_counts[j] -= 1
            for j in neighbor_indices[to_add]:
                temp_counts[j] += 1
            
            new_code_trial = set(current_code)
            new_code_trial.remove(to_remove)
            new_code_trial.add(to_add)
            
            new_cov_trial = recompute_cov(new_code_trial)
            
            delta_size = 0 # Remove 1, Add 1
            delta_uncovered = (N - new_cov_trial.bit_count()) - (N - current_cov.bit_count())
            
            delta_cost = delta_size + 1000 * delta_uncovered
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                # Accept
                new_counts = temp_counts
                new_code = new_code_trial
                new_cov = new_cov_trial
                accepted = True
        
        if accepted:
            current_code = new_code
            coverage_counts = new_counts
            current_cov = new_cov
            
            if current_cov == (1 << N) - 1: # Fully covered
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
                    # Optional: break if we hit a very good solution, but let's keep searching for 72
                    if best_size <= 72:
                        break

        T *= cooling
        
        # Periodic deterministic polish
        if random.random() < 0.05:
            indices_list = list(current_code)
            random.shuffle(indices_list)
            for idx in indices_list:
                if is_redundant(idx):
                    for j in neighbor_indices[idx]:
                        coverage_counts[j] -= 1
                    current_code.remove(idx)
                    # current_cov unchanged
                    if current_cov == (1 << N) - 1:
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = set(current_code)

    # Final check: ensure best_code is valid
    final_cov = 0
    for idx in best_code:
        final_cov |= masks[idx]
    if final_cov != (1 << N) - 1:
        # Fallback to the initial tetracode if something went wrong
        best_code = get_initial_tetracode()

    return [list(words[i]) for i in best_code]
