import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    # Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(k_blocks)
    
    # Build sparse representation
    # block_subsets[i] = list of subset indices covered by block i
    # subset_blocks[j] = list of block indices covering subset j
    block_subsets = [[] for _ in range(n_k_blocks)]
    subset_blocks = [[] for _ in range(n_t_subsets)]
    
    for i, b in enumerate(k_blocks):
        for ts in itertools.combinations(b, t):
            idx = subset_to_idx[ts]
            block_subsets[i].append(idx)
            subset_blocks[idx].append(i)
            
    # 1. Greedy Construction
    counts = np.zeros(n_t_subsets, dtype=np.int32)
    chosen_block_indices = []
    uncovered_count = n_t_subsets
    uncovered_indices = list(range(n_t_subsets)) # To quickly pick an uncovered one
    
    # We need to maintain a list of uncovered indices to pick from efficiently
    # Or just scan counts? Scanning 11k is fast.
    
    while uncovered_count > 0:
        if time.time() > deadline - 15.0:
            break
            
        # Pick a random uncovered subset to break symmetry
        # Or pick the one with fewest candidates?
        # Let's pick a random one from the uncovered set.
        # To do this efficiently, we can keep a list of uncovered indices.
        
        # Find an uncovered index
        # Since we update counts, we can just find the first 0.
        # But finding the first 0 in a 11k array is fast.
        unc_idx = -1
        for i in range(n_t_subsets):
            if counts[i] == 0:
                unc_idx = i
                break
                
        if unc_idx == -1:
            break # Should not happen if uncovered_count > 0
            
        # Get candidates for this subset
        candidates = subset_blocks[unc_idx]
        
        best_bi = -1
        best_score = -1
        
        for bi in candidates:
            # Score: number of subsets in bi that are currently uncovered
            score = 0
            for s_idx in block_subsets[bi]:
                if counts[s_idx] == 0:
                    score += 1
            if score > best_score:
                best_score = score
                best_bi = bi
                
        if best_bi == -1 or best_score == 0:
            # Error: no candidate covers the uncovered subset?
            break
            
        # Accept best_bi
        chosen_block_indices.append(best_bi)
        
        # Update counts
        for s_idx in block_subsets[best_bi]:
            if counts[s_idx] == 0:
                uncovered_count -= 1
            counts[s_idx] += 1
            
    current_indices = np.array(chosen_block_indices, dtype=np.int32)
    
    # 2. Local Search: Simulated Annealing
    # Reuse block_subsets for SA
    
    def calc_energy(indices, counts_arr):
        if len(indices) == 0:
            return 0
        uncovered = (counts_arr == 0).sum()
        return len(indices) + 100 * uncovered

    current_counts = counts.copy()
    current_energy = calc_energy(current_indices, current_counts)
    best_indices = current_indices.copy()
    best_energy = current_energy
    best_counts = current_counts.copy()
    
    T = 5.0
    T_min = 0.05
    alpha = 0.999
    rng = random.Random(42)
    
    selected_set = set(current_indices.tolist())
    
    while time.time() < deadline - 1.0:
        if T < T_min:
            T = T_min
            
        move_type = rng.random()
        
        if move_type < 0.6:
            # Swap
            if len(current_indices) > 0:
                out_pos = rng.randint(0, len(current_indices) - 1)
                out_idx = current_indices[out_pos]
                
                in_idx = rng.randint(0, n_k_blocks - 1)
                if in_idx not in selected_set and in_idx != out_idx:
                    # Calculate delta energy
                    # Decrement out, increment in
                    out_subs = block_subsets[out_idx]
                    in_subs = block_subsets[in_idx]
                    
                    # Count uncovered before
                    # We need to know how many 0s become non-0 and vice versa.
                    # Let's just update a temporary copy of counts for the affected indices?
                    # Or calculate delta directly.
                    
                    # Delta N_uncovered:
                    # For out_subs: if count was 1, it becomes 0 (+1 uncovered).
                    # For in_subs: if count was 0, it becomes 1 (-1 uncovered).
                    # Note: if a subset is in both out_subs and in_subs, it cancels out?
                    # No, we decrement then increment.
                    # If j in intersection: count -> count-1 -> count. No change in uncovered status unless count was 1?
                    # If count was 1: 1->0->1. Uncovered status: No->Yes->No. Net 0.
                    # If count was 0: impossible for out_subs.
                    
                    # So we can calculate:
                    # delta_unc = 0
                    # for j in out_subs: if counts[j] == 1: delta_unc += 1
                    # for j in in_subs: if counts[j] == 0: delta_unc -= 1
                    # Wait, if j is in both, and counts[j]==1:
                    # Out: 1->0 (unc+1). In: 0->1 (unc-1). Net 0.
                    # My formula:
                    # out_subs check: counts[j]==1 -> +1.
                    # in_subs check: counts[j]==0? No, counts[j] is 1. So 0.
                    # Total +1. Incorrect.
                    
                    # Correct logic:
                    # Simulate the change.
                    # We can just apply the change to a small dict or array slice?
                    # Given small size (56), we can just do it.
                    
                    # Let's just update current_counts, check energy, and revert if rejected.
                    # This is simpler and less error-prone.
                    
                    for s in out_subs:
                        current_counts[s] -= 1
                    for s in in_subs:
                        current_counts[s] += 1
                        
                    new_uncovered = (current_counts == 0).sum()
                    new_energy = len(current_indices) + 100 * new_uncovered
                    
                    delta = new_energy - current_energy
                    
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        current_indices[out_pos] = in_idx
                        selected_set.remove(out_idx)
                        selected_set.add(in_idx)
                        current_energy = new_energy
                        
                        if new_uncovered == 0 and current_energy < best_energy:
                            best_energy = current_energy
                            best_indices = current_indices.copy()
                            best_counts = current_counts.copy()
                    else:
                        # Revert
                        for s in out_subs:
                            current_counts[s] += 1
                        for s in in_subs:
                            current_counts[s] -= 1
                        
        elif move_type < 0.85:
            # Remove
            if len(current_indices) > 0:
                out_pos = rng.randint(0, len(current_indices) - 1)
                out_idx = current_indices[out_pos]
                
                out_subs = block_subsets[out_idx]
                for s in out_subs:
                    current_counts[s] -= 1
                    
                new_uncovered = (current_counts == 0).sum()
                new_energy = (len(current_indices) - 1) + 100 * new_uncovered
                
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_indices = np.delete(current_indices, out_pos)
                    selected_set.remove(out_idx)
                    current_energy = new_energy
                    
                    if new_uncovered == 0 and current_energy < best_energy:
                        best_energy = current_energy
                        best_indices = current_indices.copy()
                        best_counts = current_counts.copy()
                else:
                    for s in out_subs:
                        current_counts[s] += 1
                        
        else:
            # Add
            in_idx = rng.randint(0, n_k_blocks - 1)
            if in_idx not in selected_set:
                in_subs = block_subsets[in_idx]
                for s in in_subs:
                    current_counts[s] += 1
                    
                new_uncovered = (current_counts == 0).sum()
                new_energy = (len(current_indices) + 1) + 100 * new_uncovered
                
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_indices = np.append(current_indices, in_idx)
                    selected_set.add(in_idx)
                    current_energy = new_energy
                    
                    if new_uncovered == 0 and current_energy < best_energy:
                        best_energy = current_energy
                        best_indices = current_indices.copy()
                        best_counts = current_counts.copy()
                else:
                    for s in in_subs:
                        current_counts[s] -= 1
                        
        T *= alpha
        
    # Final verification and return
    if len(best_indices) > 0:
        test_counts = np.zeros(n_t_subsets, dtype=np.int32)
        for idx in best_indices:
            for s in block_subsets[idx]:
                test_counts[s] += 1
        if (test_counts == 0).sum() == 0:
            return [list(k_blocks[i]) for i in best_indices]
            
    if len(current_indices) > 0:
        if (current_counts == 0).sum() == 0:
            return [list(k_blocks[i]) for i in current_indices]
            
    if len(chosen_block_indices) > 0:
        test_counts = np.zeros(n_t_subsets, dtype=np.int32)
        for idx in chosen_block_indices:
            for s in block_subsets[idx]:
                test_counts[s] += 1
        if (test_counts == 0).sum() == 0:
            return [list(k_blocks[i]) for i in chosen_block_indices]
            
    return []
