import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Precomputation
    all_blocks = [list(c) for c in itertools.combinations(range(v), k)]
    n_blocks = len(all_blocks)
    
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    t_sub_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    block_to_tsubs = []
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    
    for b_idx, block in enumerate(all_blocks):
        covered = []
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_idx[t_sub]
            covered.append(idx)
            tsub_to_blocks[idx].append(b_idx)
        block_to_tsubs.append(np.array(covered, dtype=np.int32))
        
    def time_ok():
        return time.time() < deadline

    def fast_greedy(covered_count_np, existing_blocks=None):
        # covered_count_np: numpy array of shape (n_tsubs,)
        # existing_blocks: list of block indices already selected
        # Returns: updated covered_count_np and list of new block indices added
        
        if existing_blocks is None:
            existing_blocks = []
            
        new_blocks = []
        curr_cov = covered_count_np.copy()
        
        while time_ok():
            # Check if all covered
            # np.any(curr_cov == 0) is fast
            if np.all(curr_cov > 0):
                break
                
            # Find an uncovered t-subset
            # To speed this up, we can pick a random index where cov==0
            # Or just find the first one.
            # np.where is O(n).
            uncovered_mask = curr_cov == 0
            if not np.any(uncovered_mask):
                break
            unc_indices = np.where(uncovered_mask)[0]
            # Pick random from uncovered
            unc_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]
            
            cands = tsub_to_blocks[unc_idx]
            
            best_b = -1
            best_g = -1
            
            for b_idx in cands:
                # Calculate gain: number of 0s in curr_cov for indices in block_to_tsubs[b_idx]
                # curr_cov[block_to_tsubs[b_idx]] gives the values
                # Count how many are 0
                g = np.sum(curr_cov[block_to_tsubs[b_idx]] == 0)
                
                if g > best_g:
                    best_g = g
                    best_b = b_idx
                elif g == best_g and random.random() < 0.1:
                    best_b = b_idx
                    
            if best_b == -1:
                break
                
            new_blocks.append(best_b)
            # Update coverage
            curr_cov[block_to_tsubs[best_b]] += 1
            
        return curr_cov, new_blocks

    # Initial Solution
    init_cov = np.zeros(n_tsubs, dtype=np.int8)
    _, init_blocks = fast_greedy(init_cov)
    
    best_blocks = init_blocks
    
    # Reduction Function
    def reduce_cover(blocks_indices):
        if not blocks_indices:
            return []
        
        changed = True
        while changed and time_ok():
            changed = False
            # Calculate counts
            counts = np.zeros(n_tsubs, dtype=np.int16)
            for b_idx in blocks_indices:
                counts[block_to_tsubs[b_idx]] += 1
            
            new_indices = []
            for b_idx in blocks_indices:
                # If any tsub in this block has count == 1, keep it
                if np.any(counts[block_to_tsubs[b_idx]] == 1):
                    new_indices.append(b_idx)
                else:
                    changed = True
            
            blocks_indices = new_indices
        return blocks_indices

    best_blocks = reduce_cover(best_blocks)
    best_size = len(best_blocks)
    
    # ILS Loop: Perturb and Repair
    while time_ok():
        if random.random() < 0.1 and len(best_blocks) > 0:
            # Small perturbation: remove 1 block
            idx_to_remove = random.randint(0, len(best_blocks) - 1)
            curr_blocks = best_blocks.copy()
            curr_blocks.pop(idx_to_remove)
        else:
            # Medium perturbation: remove 2 blocks? Or restart?
            # Let's try removing 2
            if len(best_blocks) >= 2:
                idx1, idx2 = random.sample(range(len(best_blocks)), 2)
                curr_blocks = best_blocks.copy()
                # Remove largest index first to keep indices valid
                curr_blocks.pop(max(idx1, idx2))
                curr_blocks.pop(min(idx1, idx2))
            else:
                curr_blocks = best_blocks.copy()
                
        # Calculate coverage of perturbed set
        pert_cov = np.zeros(n_tsubs, dtype=np.int8)
        for b_idx in curr_blocks:
            pert_cov[block_to_tsubs[b_idx]] += 1
            
        # Repair
        pert_cov, repair_blocks = fast_greedy(pert_cov, curr_blocks)
        all_curr_blocks = curr_blocks + repair_blocks
        
        # Reduce
        reduced_blocks = reduce_cover(all_curr_blocks)
        
        if len(reduced_blocks) < best_size:
            best_blocks = reduced_blocks
            best_size = len(best_blocks)
            
    return [all_blocks[i] for i in best_blocks]
