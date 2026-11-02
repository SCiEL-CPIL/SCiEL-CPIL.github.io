import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    # Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage for each block: list of t-subset indices
    # Also precompute the set of t-subset indices for faster intersection checks?
    # For v=19, k=8, C(8,5)=56. List is fine.
    block_coverage = []
    for block in all_blocks:
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(t_subset_to_idx[ts])
        block_coverage.append(covered)
        
    # Inverse index: for each t-subset, which blocks cover it?
    # This is crucial for efficient repair and swap selection.
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, cov in enumerate(block_coverage):
        for idx in cov:
            t_to_blocks[idx].append(b_idx)
            
    # State variables
    # counts[i] = number of blocks in current solution covering t_subset i
    counts = [0] * num_t_subsets
    current_blocks = [] # List of block indices
    current_block_set = set() # For O(1) membership check
    
    def add_block(b_idx):
        current_blocks.append(b_idx)
        current_block_set.add(b_idx)
        for idx in block_coverage[b_idx]:
            counts[idx] += 1
            
    def remove_block(b_idx):
        current_blocks.remove(b_idx)
        current_block_set.discard(b_idx)
        for idx in block_coverage[b_idx]:
            counts[idx] -= 1
            if counts[idx] < 0:
                raise ValueError("Count negative")
                
    def is_valid():
        # Check if all counts are >= 1
        # This is O(num_t_subsets). C(19,5) = 11628. Fast enough.
        for c in counts:
            if c < 1:
                return False
        return True
        
    def count_uncovered():
        unc = 0
        for c in counts:
            if c < 1:
                unc += 1
        return unc

    # 1. Greedy Initialization
    # We want to cover all t-subsets.
    uncovered = set(range(num_t_subsets))
    
    # To speed up greedy, sort blocks by coverage of uncovered? 
    # Standard greedy: pick block covering most uncovered.
    # With 75k blocks and 11k subsets, iterating all blocks each step is 75k * 300 = 22.5M ops. 
    # In Python, this might be tight but doable if optimized.
    # Optimization: Only consider blocks that cover at least one uncovered subset.
    
    # Precompute for each block the list of covered indices (already done)
    
    # Better Greedy: 
    # 1. Find the t-subset with the fewest covering blocks (hardest to cover).
    # 2. Pick a block that covers it.
    # 3. Repeat.
    
    # Let's use the "hardest subset" heuristic for initialization.
    # Calculate initial frequency of each t-subset in the universe of all blocks?
    # No, just use the inverse index.
    
    while uncovered:
        # Find the uncovered subset with the minimum number of candidate blocks
        # To avoid scanning all uncovered subsets every time (which is 11k * 300 = 3.3M),
        # we can just pick a random uncovered subset if the list is large, 
        # or find the min. Let's find the min for better quality.
        
        min_count = float('inf')
        hardest_subset_idx = -1
        
        # Optimization: Sample if uncovered is large? 
        # For v=19, 11k subsets is small. Scanning is fine.
        for idx in uncovered:
            # Number of blocks that cover idx AND are not yet in solution?
            # Actually, we just need ANY block. The count in t_to_blocks is static.
            # But we should prefer blocks that cover many OTHER uncovered subsets too.
            # Standard greedy: pick block maximizing new coverage.
            # Heuristic: pick block covering the hardest subset, tie-break by total new coverage.
            
            # Let's stick to the standard "max gain" greedy but optimized.
            # It's safer for quality.
            pass
            
    # Let's restart with the standard max-gain greedy but with a trick:
    # Precompute for each block its coverage list.
    # In each step, iterate all blocks, calculate gain = len(block_coverage[b] intersect uncovered).
    # Pick max gain.
    
    uncovered = set(range(num_t_subsets))
    
    while uncovered:
        best_gain = -1
        best_block_idx = -1
        
        # Optimization: 
        # If we have many uncovered, we can sample a few blocks? No, need quality.
        # Iterate all blocks.
        for b_idx in range(num_blocks):
            gain = 0
            # Fast intersection check
            for idx in block_coverage[b_idx]:
                if idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            break # Should not happen
            
        add_block(best_block_idx)
        for idx in block_coverage[best_block_idx]:
            if counts[idx] == 1:
                uncovered.discard(idx)
                
    # 2. Initial Reduction
    # Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(current_blocks)):
            b_idx = current_blocks[i]
            redundant = True
            for idx in block_coverage[b_idx]:
                if counts[idx] == 1:
                    redundant = False
                    break
            if redundant:
                remove_block(b_idx)
                changed = True
                break
                
    best_size = len(current_blocks)
    best_blocks_copy = current_blocks[:]
    best_counts_copy = counts[:]
    
    # 3. Iterated Local Search (ILS)
    # Loop until time runs out
    # Each iteration:
    # a) Local Search: Try to reduce size via swaps/deletes that maintain validity.
    # b) If stuck, Perturb: Remove random blocks, then Greedy Repair.
    
    def local_search(max_iterations=100):
        # Returns new blocks list and counts
        # Uses global state
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            # Try Delete
            # Pick a random block, check if redundant
            if current_blocks:
                i = random.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                redundant = True
                for idx in block_coverage[b_idx]:
                    if counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(b_idx)
                    improved = True
                    continue
            
            # Try Swap
            # Pick a random block to replace
            if current_blocks:
                i = random.randint(0, len(current_blocks) - 1)
                old_b_idx = current_blocks[i]
                
                # Find unique subsets of old_b_idx
                unique_indices = []
                for idx in block_coverage[old_b_idx]:
                    if counts[idx] == 1:
                        unique_indices.append(idx)
                        
                if not unique_indices:
                    # It's redundant, should have been deleted, but maybe we missed it?
                    remove_block(old_b_idx)
                    improved = True
                    continue
                    
                # We need a new block that covers ALL unique_indices
                # Candidate blocks must cover the first unique index
                first_unique = unique_indices[0]
                candidates = t_to_blocks[first_unique]
                
                # Filter candidates
                # We want a candidate that is NOT in current_block_set
                # And covers all unique_indices
                # And ideally reduces the "excess" coverage of other subsets?
                # For now, just find ANY valid replacement.
                
                found = False
                random.shuffle(candidates) # To avoid bias
                for cand_idx in candidates:
                    if cand_idx in current_block_set:
                        continue
                    
                    # Check if cand covers all unique_indices
                    valid = True
                    # Optimization: Check if cand covers all unique indices
                    # Since unique_indices is small, loop is fast.
                    for u_idx in unique_indices:
                        if u_idx not in block_coverage[cand_idx]:
                            valid = False
                            break
                            
                    if valid:
                        # Perform swap
                        remove_block(old_b_idx)
                        add_block(cand_idx)
                        improved = True
                        found = True
                        break
                        
        return improved
        
    def greedy_repair():
        # Current state might be invalid (some counts < 1)
        # Add blocks until valid
        # Use greedy: pick block covering most uncovered
        while True:
            uncovered = [i for i in range(num_t_subsets) if counts[i] < 1]
            if not uncovered:
                break
            
            best_gain = -1
            best_block_idx = -1
            
            # Optimize: Only look at blocks that cover at least one uncovered subset
            # We can pre-filter? Or just iterate all.
            # Since we just removed a few, many blocks are still good.
            # To speed up, we can sample? No, need to ensure we find a block.
            
            # Heuristic: Pick an uncovered subset, pick a block covering it.
            # This is faster but might be lower quality.
            # Let's try the "max gain" but restricted to blocks that cover the "hardest" uncovered subset.
            
            # Find uncovered subset with min candidates?
            # Or just pick a random uncovered subset and pick a random block covering it?
            # "Random Repair" is often fast and effective in ILS.
            
            # Let's use a hybrid: 
            # 1. Identify uncovered subsets.
            # 2. Pick one at random.
            # 3. Pick a random block that covers it (and is not already in set? Or is it? 
            #    If it's in the set, counts would be >= 1 unless we removed it. 
            #    We removed blocks, so some blocks might be re-added? 
            #    If a block is in current_block_set, its subsets have counts >= 1 from it. 
            #    If count < 1, the block covering it is NOT in current_block_set.
            #    So we can safely pick from t_to_blocks[idx] and check membership.
            
            target_idx = random.choice(uncovered)
            candidates = t_to_blocks[target_idx]
            
            # Filter out blocks already in set (should be none for this target if count<1? 
            # Wait, if count < 1, NO block in current_set covers it. 
            # So ALL blocks in t_to_blocks[target_idx] are not in current_set.
            # So we can just pick one randomly!
            
            new_block_idx = random.choice(candidates)
            add_block(new_block_idx)
            
            # Check if this block covers other uncovered subsets?
            # The loop will handle remaining uncovered in next iteration.
            
        # After repair, do a quick reduction
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                redundant = True
                for idx in block_coverage[b_idx]:
                    if counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(b_idx)
                    changed = True
                    break
                    
    # Main ILS Loop
    while time.time() - start_time < time_limit:
        # 1. Local Search
        # Try to improve current solution
        local_search(max_iterations=50)
        
        # Update best
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks_copy = current_blocks[:]
            best_counts_copy = counts[:]
            
        # 2. Perturbation
        # Remove a small number of random blocks to break local optimum
        # Number to remove: proportional to size? Or fixed?
        # Let's remove 1-3 blocks.
        num_remove = random.randint(1, 3)
        
        # Ensure we don't remove all
        if len(current_blocks) > num_remove:
            # Pick random blocks to remove
            remove_indices = random.sample(range(len(current_blocks)), num_remove)
            # Remove in reverse order to keep indices valid
            for i in sorted(remove_indices, reverse=True):
                b_idx = current_blocks[i]
                remove_block(b_idx)
                
            # 3. Repair
            greedy_repair()
            
            # 4. Local Search again on the repaired solution
            local_search(max_iterations=50)
            
            # Update best
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks_copy = current_blocks[:]
                best_counts_copy = counts[:]
                
    # Return the best solution found
    return [list(all_blocks[i]) for i in best_blocks_copy]
