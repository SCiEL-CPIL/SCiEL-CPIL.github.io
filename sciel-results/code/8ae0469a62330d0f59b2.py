import itertools
import random
import time
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    
    t_index = {t_set: i for i, t_set in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute t-coverage for each block as numpy arrays
    block_t_indices = []
    for block in all_blocks:
        t_subs = list(itertools.combinations(block, t))
        indices = np.array([t_index[ts] for ts in t_subs], dtype=np.int32)
        block_t_indices.append(indices)
    
    # Precompute: for each t-subset index, which blocks cover it
    t_to_blocks = [[] for _ in range(n_t)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_to_blocks[idx].append(b_idx)
    
    def _run_greedy(seed):
        """Run one greedy pass with given seed."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=np.bool_)
        counts = np.zeros(n_t, dtype=np.int32)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline - 25:
                break
            
            # Pick a random uncovered t-subset
            uncov_indices = np.where(uncovered)[0]
            if len(uncov_indices) == 0:
                break
            target = uncov_indices[rng.randrange(len(uncov_indices))]
            
            # Find best block containing this t-subset
            candidates = t_to_blocks[target]
            if not candidates:
                break
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = np.count_nonzero(uncovered[block_t_indices[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1 or best_gain == 0:
                break
            
            selected.append(best_block_idx)
            
            # Update uncovered and counts
            block_cov = block_t_indices[best_block_idx]
            counts[block_cov] += 1
            uncovered[block_cov] = (counts[block_cov] > 1)
        
        return selected, counts
    
    def _prune_redundant(selected, counts):
        """Remove redundant blocks."""
        if not selected:
            return selected, counts
        counts = counts.copy()
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                b_idx = selected[i]
                block_cov = block_t_indices[b_idx]
                redundant = True
                for idx in block_cov:
                    if counts[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    selected.pop(i)
                    counts[block_cov] -= 1
                    changed = True
        return selected, counts
    
    def _local_search_swaps(selected, counts, max_rounds=3):
        """Try to improve by swapping blocks."""
        if not selected:
            return selected, counts
        counts = counts.copy()
        rng = random.Random(12345)
        
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline - 5:
                    break
                b_idx = selected[i]
                block_cov = block_t_indices[b_idx]
                
                # Find critical t-subsets (only covered by this block)
                critical = [idx for idx in block_cov if counts[idx] == 1]
                if not critical:
                    continue
                
                critical_set = set(critical)
                
                # Try to find a replacement block that covers all critical t-subsets
                # and provides additional coverage
                best_new_block = -1
                best_new_gain = -1
                
                # Sample some candidate blocks
                sample_size = min(50, n_blocks)
                candidates = rng.sample(range(n_blocks), sample_size)
                
                for nb_idx in candidates:
                    if nb_idx == b_idx:
                        continue
                    nb_cov = block_t_indices[nb_idx]
                    nb_cov_set = set(nb_cov)
                    
                    # Check if it covers all critical t-subsets
                    if not critical_set.issubset(nb_cov_set):
                        continue
                    
                    # Compute net gain: new uncovered t-subsets covered
                    # minus any that would become uncovered
                    new_gain = 0
                    for idx in nb_cov:
                        if counts[idx] == 0:
                            new_gain += 1
                    
                    # Check if removing old block would uncover anything
                    # that new block doesn't cover
                    would_uncover = 0
                    for idx in block_cov:
                        if counts[idx] == 1 and idx not in nb_cov_set:
                            would_uncover += 1
                    
                    if would_uncover == 0 and new_gain > best_new_gain:
                        best_new_gain = new_gain
                        best_new_block = nb_idx
                
                if best_new_block != -1 and best_new_gain > 0:
                    # Perform swap
                    old_cov = block_t_indices[b_idx]
                    new_cov = block_t_indices[best_new_block]
                    counts[old_cov] -= 1
                    counts[new_cov] += 1
                    selected[i] = best_new_block
                    improved = True
            
            if not improved:
                break
        
        return selected, counts
    
    # Run multiple greedy restarts and keep the best
    best_solution = None
    best_count = float('inf')
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        seed = 42 + r * 1000
        selected, counts = _run_greedy(seed)
        
        # Prune redundant blocks
        selected, counts = _prune_redundant(selected, counts)
        
        # Local search
        selected, counts = _local_search_swaps(selected, counts)
        
        # Final prune
        selected, counts = _prune_redundant(selected, counts)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_solution = selected
    
    if best_solution is None:
        # Fallback: run one more greedy with remaining time
        selected, counts = _run_greedy(99999)
        selected, counts = _prune_redundant(selected, counts)
        best_solution = selected
    
    # Convert to actual blocks
    blocks = [list(all_blocks[i]) for i in best_solution]
    
    return blocks
