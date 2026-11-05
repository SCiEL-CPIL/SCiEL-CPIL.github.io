import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast counting
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets and their t-subsets
    # For v=19, k=7, there are 50388 blocks.
    # Each block has 35 t-subsets.
    # Total mappings: 50388 * 35 ~ 1.7M. This is feasible to precompute or compute on the fly.
    # Computing on the fly might be faster for memory, but precomputing lists of t-subset indices for each block
    # allows for very fast gain calculation.
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    
    # block_t_indices[block_idx] = list of indices of t-subsets covered by this block
    block_t_indices = []
    for block in all_k_subsets:
        t_indices = []
        for t_sub in itertools.combinations(block, t):
            t_indices.append(subset_to_idx[t_sub])
        block_t_indices.append(t_indices)
    
    best_blocks = None
    best_score = float('inf')
    
    # Greedy with random tie-breaking and Deletion Heuristic
    while time.time() < deadline:
        # 1. Greedy Construction
        uncovered = [True] * num_t_subsets
        blocks_in_cover = []
        
        # To speed up, we track counts of coverage
        # But for greedy, we just need to know if it's uncovered.
        
        while any(uncovered):
            # Pick a random uncovered t-subset to start the block
            # This introduces randomness
            uncovered_indices = [i for i, u in enumerate(uncovered) if u]
            if not uncovered_indices:
                break
            target_idx = random.choice(uncovered_indices)
            target_subset = all_t_subsets[target_idx]
            
            # Find the best k-subset that contains target_subset
            best_block = None
            best_gain = -1
            
            # Candidates: k-subsets containing target_subset
            # We can iterate through all_k_subsets and check if they contain target_subset
            # Or generate them. Generating is better: C(v-t, k-t)
            # For v=19, t=4, k=7: C(15, 3) = 455 candidates.
            
            # To avoid generating combinations repeatedly, we could precompute candidates for each t-subset.
            # But 3876 t-subsets * 455 candidates is ~1.7M pairs.
            # Let's just filter all_k_subsets. 50k * 3876 is too slow.
            # We should precompute a mapping from t-subset index to list of block indices that cover it.
            
            # Let's do that precomputation once outside the restart loop if not done.
            # But we are inside the function. Let's do it here if not cached.
            # Since v, k, t are fixed, we can cache it.
            # However, the function is called once.
            
            # Optimization: Precompute candidates for each t-subset index
            # This is O(num_t_subsets * C(v-t, k-t))
            
            # Let's implement a simple filter for now, it might be slow if done naively.
            # Better: Precompute candidates_map
            # candidates_map[t_idx] = list of block_indices
            
            # We will compute this once.
            # Note: This might take a few seconds for v=19.
            # 3876 * 455 ~ 1.7M operations. Fast.
            
            # To avoid code duplication, we'll assume we can build this.
            # But we are in a tight loop.
            
            # Let's restructure: Precompute candidates_map BEFORE the restart loop.
            
            pass # Placeholder for logic restructuring

    # The above loop structure is messy. Let's rewrite cleanly.
    
    # Precomputation
    # Map t-subset index to list of block indices covering it
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_t_indices):
        for t_idx in t_indices:
            t_to_blocks[t_idx].append(b_idx)
    
    # Now the restart loop
    while time.time() < deadline:
        # 1. Greedy
        uncovered = [True] * num_t_subsets
        blocks_in_cover = []
        
        while any(uncovered):
            # Pick random uncovered
            # Optimization: maintain a list of uncovered indices
            # For simplicity, scan. 3876 items is fast.
            uncovered_indices = [i for i, u in enumerate(uncovered) if u]
            if not uncovered_indices: break
            target_idx = random.choice(uncovered_indices)
            
            # Get candidates
            candidates = t_to_blocks[target_idx]
            
            # Evaluate gain for each candidate
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1
                # Random tie-breaking
                if gain > best_gain or (gain == best_gain and random.random() < 0.5):
                    best_gain = gain
                    best_block_idx = b_idx
            
            # Add block
            blocks_in_cover.append(best_block_idx)
            for t_idx in block_t_indices[best_block_idx]:
                uncovered[t_idx] = False
        
        # 2. Deletion Heuristic
        # Count coverage of each t-subset by the current blocks
        coverage_count = [0] * num_t_subsets
        for b_idx in blocks_in_cover:
            for t_idx in block_t_indices[b_idx]:
                coverage_count[t_idx] += 1
        
        # Try to remove blocks
        # We iterate and check if a block is redundant.
        # A block is redundant if for all t_idx in block, coverage_count[t_idx] > 1.
        # If we remove it, we must decrement counts.
        # If we remove it, we might make other blocks redundant.
        # So we do a pass, and if any block is removed, we restart the pass.
        
        changed = True
        current_blocks = list(blocks_in_cover)
        
        while changed:
            changed = False
            # Update coverage counts for current_blocks
            # This is O(N * 35). N~150. 5250 ops. Fast.
            coverage_count = [0] * num_t_subsets
            for b_idx in current_blocks:
                for t_idx in block_t_indices[b_idx]:
                    coverage_count[t_idx] += 1
            
            # Find a redundant block
            # We can iterate in random order to break symmetry
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                b_idx = current_blocks[i]
                redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if coverage_count[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    current_blocks.pop(i)
                    changed = True
                    break # Restart the scan with updated counts
        
        # 3. Update Best
        if len(current_blocks) < best_score:
            best_score = len(current_blocks)
            best_blocks = [all_k_subsets[b_idx] for b_idx in current_blocks]
            
    return best_blocks if best_blocks else []

# Wait, the precomputation `t_to_blocks` and `block_t_indices` is inside the function.
# I need to move the precomputation to the top of the function, before the while loop.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    block_t_indices = []
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_subsets):
        t_indices = []
        for t_sub in itertools.combinations(block, t):
            t_idx = subset_to_idx[t_sub]
            t_indices.append(t_idx)
            t_to_blocks[t_idx].append(b_idx)
        block_t_indices.append(t_indices)
        
    best_blocks = None
    best_score = float('inf')
    
    while time.time() < deadline:
        # Greedy
        uncovered = [True] * num_t_subsets
        blocks_in_cover = []
        
        while any(uncovered):
            uncovered_indices = [i for i, u in enumerate(uncovered) if u]
            if not uncovered_indices: break
            target_idx = random.choice(uncovered_indices)
            
            candidates = t_to_blocks[target_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1
                if gain > best_gain or (gain == best_gain and random.random() < 0.5):
                    best_gain = gain
                    best_block_idx = b_idx
            
            blocks_in_cover.append(best_block_idx)
            for t_idx in block_t_indices[best_block_idx]:
                uncovered[t_idx] = False
        
        # Deletion
        current_blocks = list(blocks_in_cover)
        changed = True
        while changed:
            changed = False
            coverage_count = [0] * num_t_subsets
            for b_idx in current_blocks:
                for t_idx in block_t_indices[b_idx]:
                    coverage_count[t_idx] += 1
            
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                b_idx = current_blocks[i]
                redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if coverage_count[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    current_blocks.pop(i)
                    changed = True
                    break
        
        if len(current_blocks) < best_score:
            best_score = len(current_blocks)
            best_blocks = [all_k_subsets[b_idx] for b_idx in current_blocks]
            
    return best_blocks if best_blocks else []
