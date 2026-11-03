import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    # Also store the list of t-subset indices for each block for fast iteration
    block_masks = [0] * n_blocks
    block_t_indices = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(k_subsets):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices
        
    def greedy_cover():
        uncovered = all_mask
        selected = []
        used = [False] * n_blocks
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            # Optimization: If we have many blocks, this loop is slow.
            # We can break early if gain is 0.
            for i in range(n_blocks):
                if used[i]:
                    continue
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if gain == uncovered.bit_count():
                        break
            if best_idx == -1:
                break
            selected.append(best_idx)
            used[best_idx] = True
            uncovered &= ~block_masks[best_idx]
        return selected

    def optimize_solution(selected):
        # Remove redundancies using coverage counts
        # selected is a list of block indices
        if not selected:
            return selected
        
        # Count coverage for each t-subset
        cov_count = [0] * n_t_subsets
        for b_idx in selected:
            for t_idx in block_t_indices[b_idx]:
                cov_count[t_idx] += 1
        
        changed = True
        while changed:
            changed = False
            # Iterate backwards to allow safe removal
            for i in range(len(selected) - 1, -1, -1):
                b_idx = selected[i]
                # Check if any t-subset in this block has cov_count == 1
                redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if cov_count[t_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block
                    for t_idx in block_t_indices[b_idx]:
                        cov_count[t_idx] -= 1
                    selected.pop(i)
                    changed = True
                    break # Restart the loop to update counts
        
        return selected

    def local_search_1_1(selected):
        # Try to swap one block for another if it allows subsequent removal?
        # Or just try to find a block that covers the "unique" part of another?
        # This is complex. Let's stick to redundancy removal which is proven.
        pass

    start_time = time.time()
    best_solution = greedy_cover()
    best_solution = optimize_solution(best_solution)
    best_size = len(best_solution)
    
    # Multi-start with slight randomization
    # We will try a few restarts.
    for restart in range(50):
        if time.time() - start_time > 100:
            break
            
        # Randomized Greedy: Pick from top K candidates
        uncovered = all_mask
        selected = []
        used = [False] * n_blocks
        
        while uncovered:
            # Calculate gains for all unused blocks?
            # To speed up, we can just do standard greedy but with random tie-breaking
            # or pick from top 5.
            # For simplicity and speed, let's just do standard greedy but
            # maybe shuffle the order of blocks if they have the same gain?
            # Or just run standard greedy again?
            # Standard greedy is deterministic.
            # Let's try "Greedy with Penalty": 
            # gain = (mask & uncovered).bit_count() - penalty * (mask & covered).bit_count()?
            # No, standard greedy is usually good.
            
            # Let's try a simple randomization: 
            # With prob 0.1, pick a random block with gain > 0? No, too risky.
            # Let's just run the standard greedy. The improvement comes from the optimizer.
            # But if greedy is deterministic, restarts are identical.
            # We need to vary the greedy.
            
            # Variation: Weighted Greedy
            # Initialize weights to 1.
            # After picking a block, increase weight of blocks that overlap with it?
            # Or decrease?
            
            # Let's try: Pick best gain. If tie, pick random.
            best_gain = -1
            candidates = []
            for i in range(n_blocks):
                if used[i]:
                    continue
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if not candidates:
                break
                
            # Pick random from candidates
            idx = random.choice(candidates)
            selected.append(idx)
            used[idx] = True
            uncovered &= ~block_masks[idx]
            
        selected = optimize_solution(selected)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected[:]
            
    return [list(k_subsets[i]) for i in best_solution]
