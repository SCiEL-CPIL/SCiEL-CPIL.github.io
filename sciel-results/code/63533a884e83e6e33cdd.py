import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(all_k_subs)
    
    # Map t-subset to index
    t_sub_to_idx = {tsub: i for i, tsub in enumerate(all_t_subs)}
    
    # Precompute for each k-subset, the list of t-subset indices it covers
    # k_sub_t_indices[i] is a list of t-subset indices covered by all_k_subs[i]
    k_sub_t_indices = []
    for ksub in all_k_subs:
        indices = []
        for tsub in itertools.combinations(ksub, t):
            indices.append(t_sub_to_idx[tsub])
        k_sub_t_indices.append(indices)
    
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # State
    selected_set = set()
    coverage_count = np.zeros(n_t_subs, dtype=np.int32)
    current_size = 0
    current_uncovered = n_t_subs
    
    # Helper to update coverage
    def add_block(idx):
        nonlocal current_size, current_uncovered
        if idx in selected_set:
            return
        selected_set.add(idx)
        current_size += 1
        for t_idx in k_sub_t_indices[idx]:
            if coverage_count[t_idx] == 0:
                current_uncovered -= 1
            coverage_count[t_idx] += 1
            
    def remove_block(idx):
        nonlocal current_size, current_uncovered
        if idx not in selected_set:
            return
        selected_set.remove(idx)
        current_size -= 1
        for t_idx in k_sub_t_indices[idx]:
            coverage_count[t_idx] -= 1
            if coverage_count[t_idx] == 0:
                current_uncovered += 1

    # 1. Initialization: Randomized Greedy
    # We want to cover all t-subsets
    while current_uncovered > 0:
        best_gain = -1
        best_candidates = []
        # To speed up, we can sample or iterate all. 50k is small enough to iterate if we do it once.
        # But we do this ~150 times. 150 * 50k = 7.5M ops. This is fine in Python?
        # It might be borderline. Let's try.
        for i in range(n_k_subs):
            if i in selected_set:
                continue
            gain = 0
            for t_idx in k_sub_t_indices[i]:
                if coverage_count[t_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_candidates = [i]
            elif gain == best_gain:
                best_candidates.append(i)
        
        if best_gain <= 0:
            # Should not happen if a solution exists
            break
            
        # Pick randomly from best candidates to add diversity
        chosen = rng.choice(best_candidates)
        add_block(chosen)
        
        if time.time() - start_time > time_limit:
            break

    # Initial Local Optimization: Remove redundant blocks
    def local_optimize():
        nonlocal current_size, current_uncovered
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                return
            improved = False
            # Try to remove each block
            for idx in list(selected_set):
                # Check if idx is redundant
                # A block is redundant if all t-subsets it covers are covered by other blocks
                # i.e., coverage_count[t_idx] > 1 for all t_idx in k_sub_t_indices[idx]
                is_redundant = True
                for t_idx in k_sub_t_indices[idx]:
                    if coverage_count[t_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    remove_block(idx)
                    improved = True
    
    local_optimize()
    
    best_size = current_size
    best_set = set(selected_set)
    
    # 2. Simulated Annealing
    # Energy function: Size + Lambda * Uncovered
    # We want to minimize size, but must ensure Uncovered == 0.
    # If Uncovered > 0, it's invalid. We penalize heavily.
    LAMBDA = 10000.0
    
    def get_energy():
        return current_size + LAMBDA * current_uncovered
    
    T = 100.0
    cooling = 0.99
    min_T = 0.01
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Propose Move
        move_type = rng.random()
        if move_type < 0.3:
            # Add random block
            # Pick random index not in set
            while True:
                idx = rng.randint(0, n_k_subs - 1)
                if idx not in selected_set:
                    break
            old_energy = get_energy()
            add_block(idx)
            new_energy = get_energy()
            delta = new_energy - old_energy
            if delta < 0 or rng.random() < np.exp(-delta / T):
                pass # Accept
            else:
                remove_block(idx) # Reject
        elif move_type < 0.6:
            # Remove random block
            if current_size > 0:
                idx = rng.choice(list(selected_set))
                old_energy = get_energy()
                remove_block(idx)
                new_energy = get_energy()
                delta = new_energy - old_energy
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    pass # Accept
                else:
                    add_block(idx) # Reject
        else:
            # Swap
            if current_size > 0:
                idx_in = rng.choice(list(selected_set))
                while True:
                    idx_out = rng.randint(0, n_k_subs - 1)
                    if idx_out not in selected_set:
                        break
                old_energy = get_energy()
                remove_block(idx_in)
                add_block(idx_out)
                new_energy = get_energy()
                delta = new_energy - old_energy
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    pass # Accept
                else:
                    remove_block(idx_out)
                    add_block(idx_in) # Reject
        
        # Track best
        if current_uncovered == 0 and current_size < best_size:
            best_size = current_size
            best_set = set(selected_set)
            
        # Cooling
        T *= cooling
        
        # Occasionally do local optimization to clean up
        if rng.random() < 0.1:
            local_optimize()
            if current_uncovered == 0 and current_size < best_size:
                best_size = current_size
                best_set = set(selected_set)

    # Final cleanup
    # Restore best set
    selected_set.clear()
    coverage_count[:] = 0
    current_size = 0
    current_uncovered = n_t_subs
    for idx in best_set:
        add_block(idx)
        
    local_optimize()
    
    # Return blocks
    return [all_k_subs[i] for i in selected_set]
