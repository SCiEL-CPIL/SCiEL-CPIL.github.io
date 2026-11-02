import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Greedy Set Cover (Whole Blocks) 
    followed by Local Search Minimization.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s for safety
    
    # 1. Pre-compute all possible k-blocks and their covered t-subsets
    # C(24, 7) is ~496k. This is manageable.
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_subsets = {}
    for b in all_blocks:
        subs = tuple(sorted(itertools.combinations(b, t)))
        block_to_subsets[b] = subs
        
    # 2. Greedy Set Cover
    all_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_subsets)
    blocks = []
    
    # To speed up greedy, we can't iterate all 496k blocks every time if we have many steps.
    # But C(24,3) is 2024. The number of blocks in the covering is ~76.
    # So we do ~76 iterations.
    # 76 * 496k = 37M operations. In Python, this might be slow (seconds).
    # We need to optimize the "find best block" step.
    
    # Optimization: 
    # We can maintain a list of "active" blocks? 
    # Or just iterate. 37M simple set intersections might be ~5-10 seconds in Python.
    # Let's try to optimize by pre-computing subsets as sets?
    # Or using bitmasks?
    # v=24, t=3. Subsets are small.
    
    # Let's use a simpler heuristic for the first few blocks, then switch?
    # No, let's just run the greedy.
    
    # To make it faster, we can stop iterating if we find a block that covers all remaining? Unlikely.
    # We can also randomize the order of blocks to break ties?
    
    # Let's try to speed up the intersection.
    # uncovered is a set of tuples.
    # block_to_subsets[b] is a tuple of tuples.
    
    # Alternative: 
    # Instead of iterating all blocks, we can pick a random uncovered subset S.
    # Then consider only blocks that contain S.
    # How many blocks contain a specific t-subset?
    # C(v-t, k-t). For v=24, k=7, t=3: C(21, 4) = 5985.
    # This is much smaller than 496k!
    
    # Greedy Strategy:
    # 1. Pick an uncovered subset S (e.g., the first one, or random).
    # 2. Find all blocks that cover S.
    # 3. Among those, pick the one that covers the most *other* uncovered subsets.
    
    # This is much faster.
    
    # Pre-compute: map from t-subset to list of blocks covering it.
    subset_to_blocks = {s: [] for s in all_subsets}
    for b, subs in block_to_subsets.items():
        for s in subs:
            subset_to_blocks[s].append(b)
            
    # Now Greedy:
    while uncovered:
        # Pick a subset to focus on. 
        # To be deterministic and fast, pick the first one in the set? 
        # Or the one with the fewest covering blocks (most constrained)?
        # Let's pick the first one for simplicity, or random to add diversity.
        # Random is better for local search variety later? 
        # Let's pick the one with the smallest number of available blocks (heuristic).
        # This requires counting how many blocks covering S are still "available" (not used? No, blocks can be reused? No, set cover uses each block at most once? 
        # Wait, in set cover, we pick distinct blocks. 
        # But here, we are constructing a covering design. Blocks must be distinct.
        # So we cannot pick the same block twice.
        
        # So we need to track which blocks are already used.
        used_blocks = set()
        
        # To find the "most constrained" uncovered subset:
        # For each s in uncovered, count how many blocks in subset_to_blocks[s] are not in used_blocks.
        # This is expensive if we do it for all uncovered.
        
        # Simpler: Just pick the first uncovered subset.
        s = next(iter(uncovered))
        
        # Get candidates
        candidates = subset_to_blocks[s]
        
        best_block = None
        best_gain = -1
        
        for b in candidates:
            if b in used_blocks:
                continue
            # Calculate gain: how many uncovered subsets does b cover?
            gain = 0
            for sub in block_to_subsets[b]:
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
                
        if best_block is None:
            # Should not happen if valid
            break
            
        blocks.append(best_block)
        used_blocks.add(best_block)
        
        # Update uncovered
        for sub in block_to_subsets[best_block]:
            uncovered.discard(sub)
            
    # 3. Local Search Minimization
    # Same as before: remove redundant blocks, then swap points to enable removal.
    
    # Data structures
    coverage_counts = {}
    for b in blocks:
        for sub in block_to_subsets[b]:
            coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
            
    def is_redundant(block, coverage_counts):
        for sub in block_to_subsets[block]:
            if coverage_counts[sub] <= 1:
                return False
        return True
        
    def remove_block(block, coverage_counts, blocks):
        for sub in block_to_subsets[block]:
            coverage_counts[sub] -= 1
        blocks.remove(block)
        
    def add_block(block, coverage_counts, blocks):
        for sub in block_to_subsets[block]:
            coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
        blocks.append(block)
        
    # Minimization Loop
    while time.time() - start_time < time_limit:
        removed_any = False
        
        # Pass 1: Try to remove any redundant block
        for i in range(len(blocks) - 1, -1, -1):
            if is_redundant(blocks[i], coverage_counts):
                remove_block(blocks[i], coverage_counts, blocks)
                removed_any = True
                break
                
        if removed_any:
            continue
            
        # Pass 2: Try to swap points to enable removal
        if not blocks:
            break
            
        block_idx = random.randint(0, len(blocks) - 1)
        block = blocks[block_idx]
        
        # Try swapping a point in this block
        x_idx = random.randint(0, k - 1)
        x = block[x_idx]
        
        candidates = [y for y in range(v) if y not in block]
        if not candidates:
            continue
        y = random.choice(candidates)
        
        new_block_list = [p for p in block if p != x]
        new_block_list.append(y)
        new_block_tuple = tuple(sorted(new_block_list))
        
        # Check if new block is valid (distinct points) - already ensured by construction
        # Check if we can swap without breaking coverage
        # Subsets in old block containing x must have coverage > 1
        can_swap = True
        for sub in block_to_subsets[block]:
            if x in sub:
                if coverage_counts[sub] == 1:
                    can_swap = False
                    break
                    
        if not can_swap:
            continue
            
        # Simulate swap
        # Remove old block counts
        for sub in block_to_subsets[block]:
            coverage_counts[sub] -= 1
            
        # Add new block counts
        # Note: new_block_tuple might not be in block_to_subsets if it wasn't pre-computed?
        # It IS in block_to_subsets because we pre-computed ALL C(v,k) blocks.
        for sub in block_to_subsets[new_block_tuple]:
            coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
            
        # Check for redundant block
        found_redundant = -1
        for i in range(len(blocks)):
            b = blocks[i]
            if i == block_idx:
                b = new_block_tuple
            if is_redundant(b, coverage_counts):
                found_redundant = i
                break
                
        if found_redundant != -1:
            # Accept swap
            blocks[block_idx] = list(new_block_tuple)
            
            # Remove redundant block
            b_to_remove = blocks[found_redundant]
            if found_redundant == block_idx:
                b_to_remove = new_block_tuple
                
            for sub in block_to_subsets[b_to_remove]:
                coverage_counts[sub] -= 1
            del blocks[found_redundant]
            
        else:
            # Revert
            for sub in block_to_subsets[new_block_tuple]:
                coverage_counts[sub] -= 1
            for sub in block_to_subsets[block]:
                coverage_counts[sub] += 1

    return [list(b) for b in blocks]
