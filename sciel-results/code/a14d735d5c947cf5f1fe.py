"""
Constructs the Steiner system S(5,6,12) using an Exact Cover solver.
The problem is to find 132 blocks of size 6 on 12 points such that every 
5-subset is contained in exactly one block. This is equivalent to an exact cover 
of the set of all 792 five-subsets by the 924 six-subsets.
"""

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Points are 0..11
    n_points = 12
    block_size = 6
    subset_size = 5
    
    from itertools import combinations
    
    # 1. Generate all possible blocks (6-subsets)
    all_blocks = list(combinations(range(n_points), block_size))
    num_blocks = len(all_blocks)
    
    # 2. Generate all possible 5-subsets
    all_5_subsets = list(combinations(range(n_points), subset_size))
    num_5_subsets = len(all_5_subsets)
    
    # Map each 5-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # 3. Build the incidence matrix: which blocks contain which 5-subsets
    # block_to_subsets[i] is a list of indices of 5-subsets contained in block i
    block_to_subsets = [None] * num_blocks
    # subset_to_blocks[j] is a list of indices of blocks containing 5-subset j
    subset_to_blocks = [[] for _ in range(num_5_subsets)]
    
    for i, b in enumerate(all_blocks):
        subsets_in_b = []
        for s in combinations(b, subset_size):
            idx = subset_to_idx[s]
            subsets_in_b.append(idx)
            subset_to_blocks[idx].append(i)
        block_to_subsets[i] = subsets_in_b

    # State for backtracking
    # covered[j] = True if 5-subset j is already covered
    # selected_blocks = list of block indices selected so far
    # We use a recursive backtracking solver with MRV (Minimum Remaining Values) heuristic.
    
    covered = [False] * num_5_subsets
    selected = []
    
    # To speed up, we can precompute the count of available blocks for each subset dynamically, 
    # but for n=12, a simple check is fast enough if we prune well.
    
    def solve():
        if time.time() > deadline:
            return False
            
        # Find the uncovered 5-subset with the fewest available blocks (MRV)
        min_avail = float('inf')
        target_subset_idx = -1
        
        for j in range(num_5_subsets):
            if not covered[j]:
                # Count how many blocks can still cover this subset
                # A block can cover it if:
                # 1. It contains the subset
                # 2. It doesn't conflict with already selected blocks (i.e., it doesn't share any other 5-subset)
                # Actually, in Exact Cover, if we select a block, we cover ALL its 5-subsets.
                # So a block is "available" for subset j if:
                # - It contains j
                # - All other 5-subsets in that block are NOT yet covered (otherwise conflict)
                
                avail_count = 0
                for b_idx in subset_to_blocks[j]:
                    # Check if block b_idx is compatible with current selection
                    compatible = True
                    for s_idx in block_to_subsets[b_idx]:
                        if covered[s_idx]:
                            compatible = False
                            break
                    if compatible:
                        avail_count += 1
                        if avail_count < min_avail:
                            min_avail = avail_count
                            if min_avail == 0:
                                return False # Dead end
                            
                if min_avail == 0:
                    return False
                    
                if min_avail < min_avail: # This logic is slightly off, let's just find the min
                    pass
                
        # Re-do the MRV selection properly
        min_avail = float('inf')
        target_subset_idx = -1
        best_block_options = []
        
        for j in range(num_5_subsets):
            if not covered[j]:
                avail_count = 0
                options = []
                for b_idx in subset_to_blocks[j]:
                    compatible = True
                    for s_idx in block_to_subsets[b_idx]:
                        if covered[s_idx]:
                            compatible = False
                            break
                    if compatible:
                        avail_count += 1
                        options.append(b_idx)
                        
                if avail_count == 0:
                    return False
                    
                if avail_count < min_avail:
                    min_avail = avail_count
                    target_subset_idx = j
                    best_block_options = options
                    
                if min_avail == 1:
                    break # Can't get better than 1
        
        if target_subset_idx == -1:
            # All subsets covered
            return True
            
        # Try each block option for the target subset
        for b_idx in best_block_options:
            # Select block b_idx
            selected.append(b_idx)
            # Mark all its 5-subsets as covered
            for s_idx in block_to_subsets[b_idx]:
                covered[s_idx] = True
                
            if solve():
                return True
                
            # Backtrack
            selected.pop()
            for s_idx in block_to_subsets[b_idx]:
                covered[s_idx] = False
                
        return False

    # Optimization: The search space is symmetric. We can fix the first block to break symmetry.
    # The block (0,1,2,3,4,5) is in the Witt design.
    # Let's force the first selected block to be one containing 0, or specifically (0,1,2,3,4,5) if we know it exists.
    # Actually, standard MRV is usually fast enough for this size.
    
    # To ensure we find the optimal 132, we just need *any* valid exact cover.
    # If no exact cover is found, we might need a different strategy, but S(5,6,12) exists.
    
    # Initial setup to help MRV:
    # We can arbitrarily pick the first uncovered subset.
    
    if solve():
        # Convert block indices back to lists of points
        result = [list(all_blocks[i]) for i in selected]
        return result
    else:
        # Fallback: If solver fails or times out, return empty or best effort
        # Given the constraints, it should find it.
        return []

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the Steiner system S(5,6,12) using an Exact Cover solver.
    """
    import time
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n_points = 12
    block_size = 6
    subset_size = 5
    
    from itertools import combinations
    
    all_blocks = list(combinations(range(n_points), block_size))
    num_blocks = len(all_blocks)
    
    all_5_subsets = list(combinations(range(n_points), subset_size))
    num_5_subsets = len(all_5_subsets)
    
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    block_to_subsets = [None] * num_blocks
    subset_to_blocks = [[] for _ in range(num_5_subsets)]
    
    for i, b in enumerate(all_blocks):
        subsets_in_b = []
        for s in combinations(b, subset_size):
            idx = subset_to_idx[s]
            subsets_in_b.append(idx)
            subset_to_blocks[idx].append(i)
        block_to_subsets[i] = subsets_in_b

    covered = [False] * num_5_subsets
    selected = []
    
    # Optimization: Precompute the list of 5-subsets for each block as a bitmask or set for faster intersection checks?
    # For n=12, 792 subsets, simple list iteration is fine.
    
    # To speed up MRV, we can maintain a count of available blocks for each subset.
    # But updating counts is complex. Let's try the simple version first with a time guard.
    
    def solve():
        if time.time() > deadline:
            return False
            
        min_avail = float('inf')
        target_subset_idx = -1
        best_block_options = []
        
        # Find the most constrained uncovered subset
        for j in range(num_5_subsets):
            if not covered[j]:
                avail_count = 0
                options = []
                for b_idx in subset_to_blocks[j]:
                    # Check compatibility
                    compatible = True
                    for s_idx in block_to_subsets[b_idx]:
                        if covered[s_idx]:
                            compatible = False
                            break
                    if compatible:
                        avail_count += 1
                        options.append(b_idx)
                        
                if avail_count == 0:
                    return False
                    
                if avail_count < min_avail:
                    min_avail = avail_count
                    target_subset_idx = j
                    best_block_options = options
                    
                if min_avail == 1:
                    break
        
        if target_subset_idx == -1:
            return True # All covered
            
        for b_idx in best_block_options:
            selected.append(b_idx)
            for s_idx in block_to_subsets[b_idx]:
                covered[s_idx] = True
                
            if solve():
                return True
                
            selected.pop()
            for s_idx in block_to_subsets[b_idx]:
                covered[s_idx] = False
                
        return False

    if solve():
        return [list(all_blocks[i]) for i in selected]
    else:
        return []
# EVOLVE-BLOCK-END
