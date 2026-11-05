import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    # Shape: (num_blocks, C(k,t))
    # C(7,4) = 35
    block_tsubs = np.zeros((num_blocks, 35), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_tsubs[i, j] = t_sub_to_idx[sub]
            
    # Precompute for each t-subset, the list of block indices covering it
    # tsub_to_blocks[i] is a list of block indices
    tsub_to_blocks = [[] for _ in range(num_t)]
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            tsub_to_blocks[t_sub_to_idx[sub]].append(i)
            
    # Greedy Initial Solution
    def _greedy_init():
        uncovered = np.ones(num_t, dtype=bool)
        selected_blocks = []
        while uncovered.any():
            if time.time() > deadline - 20.0:
                break
            # Calculate gain for each block
            # uncovered[block_tsubs] gives a (num_blocks, 35) bool array
            # summing over axis 1 gives (num_blocks,) int array
            gains = uncovered[block_tsubs].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_blocks.append(best_idx)
            # Update uncovered
            uncovered[block_tsubs[best_idx]] = False
            
        # Convert indices to blocks
        return [list(all_blocks[i]) for i in selected_blocks]

    # Polish: Remove redundant blocks
    def _remove_redundant(blocks):
        if not blocks:
            return blocks
        # Map block to its t-subsets
        # We need to check if removing a block leaves any t-subset uncovered.
        # Efficient way: Count coverage of each t-subset by the current set of blocks.
        # Then a block is redundant if all its t-subsets have coverage count > 1.
        
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 5.0:
                break
            
            # Calculate coverage counts
            coverage_count = np.zeros(num_t, dtype=np.int32)
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    coverage_count[t_sub_to_idx[sub]] += 1
            
            # Check for redundant blocks
            redundant_idx = None
            for i, b in enumerate(blocks):
                is_essential = False
                for sub in itertools.combinations(b, t):
                    if coverage_count[t_sub_to_idx[sub]] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    redundant_idx = i
                    break
            
            if redundant_idx is not None:
                blocks.pop(redundant_idx)
                changed = True
                
        return blocks

    # Repair: Given a set of blocks and a set of uncovered t-subsets, add blocks to cover them
    def _repair(blocks, uncovered_mask):
        # blocks is a list of block indices
        # uncovered_mask is a bool array
        # We want to add blocks to cover all True entries in uncovered_mask
        # We use a greedy approach: pick block covering most uncovered subsets
        
        # To avoid duplicates, we can track selected indices
        # But we might want to allow re-adding if it helps? No, distinct blocks required.
        # So we should only pick from blocks NOT in `blocks`.
        
        # However, checking "not in blocks" for 50k blocks is slow if done naively.
        # We can maintain a set of selected block indices.
        selected_set = set(blocks)
        
        while uncovered_mask.any():
            if time.time() > deadline - 5.0:
                break
            
            # Calculate gains for all blocks
            gains = uncovered_mask[block_tsubs].sum(axis=1)
            
            # Mask out already selected blocks
            for idx in selected_set:
                gains[idx] = 0
                
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                # Should not happen if solution is possible
                break
                
            blocks.append(best_idx)
            selected_set.add(best_idx)
            uncovered_mask[block_tsubs[best_idx]] = False
            
        return blocks

    # ILS
    best_blocks = _greedy_init()
    best_blocks = _remove_redundant(best_blocks)
    best_count = len(best_blocks)
    
    current_blocks = list(best_blocks)
    
    # Convert to indices for faster manipulation?
    # We store blocks as indices in all_blocks for speed in ILS.
    
    def _blocks_to_indices(blocks):
        return [all_blocks.index(b) for b in blocks]
        
    # Note: all_blocks.index is O(N). For 50k blocks, this is slow if called often.
    # Better to store indices directly in the solution representation.
    
    # Let's restart the representation using indices.
    
    def _greedy_init_idx():
        uncovered = np.ones(num_t, dtype=bool)
        selected_indices = []
        while uncovered.any():
            if time.time() > deadline - 20.0:
                break
            gains = uncovered[block_tsubs].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_tsubs[best_idx]] = False
        return selected_indices

    def _remove_redundant_idx(indices):
        if not indices:
            return indices
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 5.0:
                break
            coverage_count = np.zeros(num_t, dtype=np.int32)
            for idx in indices:
                for sub_idx in block_tsubs[idx]:
                    coverage_count[sub_idx] += 1
            
            redundant_idx = None
            for i, idx in enumerate(indices):
                is_essential = False
                for sub_idx in block_tsubs[idx]:
                    if coverage_count[sub_idx] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    redundant_idx = i
                    break
            
            if redundant_idx is not None:
                indices.pop(redundant_idx)
                changed = True
        return indices

    def _repair_idx(indices, uncovered_mask):
        selected_set = set(indices)
        while uncovered_mask.any():
            if time.time() > deadline - 5.0:
                break
            gains = uncovered_mask[block_tsubs].sum(axis=1)
            for idx in selected_set:
                gains[idx] = 0
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            indices.append(best_idx)
            selected_set.add(best_idx)
            uncovered_mask[block_tsubs[best_idx]] = False
        return indices

    best_indices = _greedy_init_idx()
    best_indices = _remove_redundant_idx(best_indices)
    best_count = len(best_indices)
    
    current_indices = list(best_indices)
    
    # ILS Loop
    perturbation_strength = 2
    
    while time.time() < deadline - 10.0:
        # Perturb
        perturbed = list(current_indices)
        if len(perturbed) > perturbation_strength:
            # Remove random blocks
            for _ in range(perturbation_strength):
                if not perturbed: break
                idx_to_remove = random.randrange(len(perturbed))
                perturbed.pop(idx_to_remove)
                
        # Calculate uncovered
        coverage = np.zeros(num_t, dtype=np.int32)
        for idx in perturbed:
            for sub_idx in block_tsubs[idx]:
                coverage[sub_idx] += 1
        uncovered_mask = (coverage == 0)
        
        # Repair
        repaired = _repair_idx(perturbed, uncovered_mask)
        
        # Polish
        polished = _remove_redundant_idx(repaired)
        
        if len(polished) < best_count:
            best_count = len(polished)
            best_indices = polished
            current_indices = polished
            # Reset perturbation strength if we found a better solution
            perturbation_strength = 2
        else:
            # Simple hill climbing: revert if not better
            # Or accept with probability? Let's try increasing perturbation if stuck.
            # We can track number of steps without improvement.
            # For now, just revert.
            current_indices = best_indices
            # Increase perturbation strength occasionally to escape local minima
            if random.random() < 0.1:
                perturbation_strength = min(perturbation_strength + 1, 5)
            else:
                perturbation_strength = max(2, perturbation_strength - 1)
                
    # Convert best_indices back to blocks
    return [list(all_blocks[i]) for i in best_indices]
