import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Build masks and also a numpy boolean incidence matrix for fast gain computation
    masks = [0] * n_k
    # incidence: shape (n_k, n_t), bool
    incidence = np.zeros((n_k, n_t), dtype=np.bool_)
    
    for i, k_sub in enumerate(all_k_subsets):
        m = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            m |= (1 << idx)
            incidence[i, idx] = True
        masks[i] = m

    full_mask = (1 << n_t) - 1
    
    # Precompute bit indices for each block for fast reduction
    # block_t_indices: list of lists of t-indices
    block_t_indices_all = []
    for i in range(n_k):
        m = masks[i]
        indices = []
        while m:
            lsb = m & -m
            indices.append(lsb.bit_length() - 1)
            m ^= lsb
        block_t_indices_all.append(indices)

    # Helper: Greedy cover for a specific uncovered mask
    def greedy_cover(uncovered_mask, rng=None):
        if rng is None:
            rng = random.Random()
        
        selected = []
        current_uncovered = uncovered_mask
        
        while current_uncovered:
            # Convert uncovered mask to boolean array for numpy
            # This is O(n_t) per iteration which is ~2024, acceptable
            uncovered_bool = np.zeros(n_t, dtype=np.bool_)
            m = current_uncovered
            while m:
                lsb = m & -m
                uncovered_bool[lsb.bit_length() - 1] = True
                m ^= lsb
            
            # Compute gain for all blocks: count of uncovered triples covered by each block
            # gain[i] = number of True elements in incidence[i] & uncovered_bool
            gains = np.count_nonzero(incidence & uncovered_bool[np.newaxis, :], axis=1)
            
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
            
            # Find all blocks with max gain
            max_indices = np.where(gains == max_gain)[0]
            
            # Random tie-break among max-gain blocks
            if len(max_indices) == 1:
                best_idx = int(max_indices[0])
            else:
                # Randomly pick from max-gain blocks
                best_idx = int(max_indices[rng.randint(0, len(max_indices) - 1)])
            
            selected.append(best_idx)
            current_uncovered &= ~masks[best_idx]
            
            # Early stop if time is tight
            if time.time() > deadline - 5:
                break
        
        return selected

    def reduce_cover(blocks):
        # blocks is a list of indices
        # Returns a reduced list of indices
        if not blocks:
            return []
            
        # Compute coverage count for each triple
        cov_counts = [0] * n_t
        
        # Use precomputed block_t_indices
        current_blocks = list(blocks)
        current_block_t_indices = [block_t_indices_all[b_idx] for b_idx in current_blocks]
        
        for b_idx in current_blocks:
            for idx in block_t_indices_all[b_idx]:
                cov_counts[idx] += 1
        
        improved = True
        rng_reduce = random.Random()
        while improved:
            improved = False
            order = list(range(len(current_blocks)))
            rng_reduce.shuffle(order)
            
            for bi in order:
                if bi >= len(current_blocks):
                    continue
                
                # Check if redundant
                redundant = True
                for idx in current_block_t_indices[bi]:
                    if cov_counts[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for idx in current_block_t_indices[bi]:
                        cov_counts[idx] -= 1
                    
                    current_blocks.pop(bi)
                    current_block_t_indices.pop(bi)
                    improved = True
                    break
        
        return current_blocks

    # Main ILS Loop
    # 1. Initial Greedy
    rng_main = random.Random(42)
    init_sel = greedy_cover(full_mask, rng=rng_main)
    best_blocks = reduce_cover(init_sel)
    best_len = len(best_blocks)
    
    while time.time() < deadline - 15:
        # Perturb
        current_blocks = list(best_blocks)
        
        # Remove 2 to 5 random blocks
        n_remove = rng_main.randint(2, 5)
        if len(current_blocks) <= n_remove:
            continue
            
        indices_to_remove = set(rng_main.sample(range(len(current_blocks)), n_remove))
        
        removed_mask = 0
        new_current = []
        for i, b_idx in enumerate(current_blocks):
            if i in indices_to_remove:
                removed_mask |= masks[b_idx]
            else:
                new_current.append(b_idx)
                
        # The uncovered mask is the part of removed_mask not covered by new_current
        covered_by_current = 0
        for b_idx in new_current:
            covered_by_current |= masks[b_idx]
            
        uncovered = removed_mask & ~covered_by_current
        
        if uncovered == 0:
            if len(new_current) < best_len:
                best_len = len(new_current)
                best_blocks = new_current
            continue
        
        repair_sel = greedy_cover(uncovered, rng=rng_main)
        
        candidate_blocks = new_current + repair_sel
        
        # Reduce
        candidate_blocks = reduce_cover(candidate_blocks)
        
        if len(candidate_blocks) < best_len:
            best_len = len(candidate_blocks)
            best_blocks = candidate_blocks
    
    # Final polish: try to remove individual blocks or do 1-for-1 swaps
    if time.time() < deadline - 3:
        # Try removing single blocks
        for i in range(len(best_blocks)):
            if time.time() > deadline - 2:
                break
            trial = best_blocks[:i] + best_blocks[i+1:]
            # Check if still valid
            covered = 0
            for b_idx in trial:
                covered |= masks[b_idx]
            if covered == full_mask:
                best_blocks = trial
                best_len -= 1
                break  # Restart single-removal pass
    
    # Convert indices to blocks
    return [list(all_k_subsets[i]) for i in best_blocks]
