import itertools
import sys

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks and compute their clause lists
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute clause indices for each block
    block_clauses = []
    for block in k_blocks:
        clauses = []
        for t_sub in itertools.combinations(block, t):
            clauses.append(sub_to_idx[t_sub])
        block_clauses.append(clauses)
        
    # Precompute sets of clauses for each block for fast subset checks
    # This uses memory but speeds up swap checks significantly
    block_clause_sets = [set(clauses) for clauses in block_clauses]

    # 3. Greedy Set Cover for initial solution
    uncovered = (1 << n_t_subs) - 1
    selected = []
    
    # We need masks for greedy. Let's generate them on the fly or precompute.
    # Precomputing masks for 75k blocks of 11628 bits is memory heavy (75k * 1.4KB ~ 100MB).
    # Python ints are arbitrary precision, so a mask with 11628 bits is ~1.5KB.
    # 75k * 1.5KB = 112MB. Might be too high or slow to allocate.
    # Alternative: Greedy without masks?
    # Gain = number of uncovered clauses in block.
    # We can maintain a `covered` boolean array or set.
    # `uncovered_set` = set of all clause indices.
    # For each block, gain = len(block_clauses[b] & uncovered_set).
    # This is O(56) per block. 75k * 56 = 4.2M ops per step.
    # Steps ~ 300. 1.2B ops. Too slow in Python.
    
    # So we need masks for Greedy.
    # Let's try to use numpy? Or just hope memory is ok.
    # Or, use a list of 0/1 for uncovered?
    # Let's stick to masks but maybe only for active blocks?
    # Actually, 112MB is probably fine for modern systems, but let's try to be safe.
    # We can compute masks lazily? No, we need them for all blocks in each step.
    
    # Let's use the mask approach but be mindful.
    block_masks = [0] * n_k_blocks
    for i, clauses in enumerate(block_clauses):
        mask = 0
        for c in clauses:
            mask |= (1 << c)
        block_masks[i] = mask

    while uncovered:
        best_idx = -1
        max_gain = 0
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # 4. Local Search
    # State: selected list, counts array, unique_clauses list
    counts = [0] * n_t_subs
    for idx in selected:
        for c in block_clauses[idx]:
            counts[c] += 1
            
    # unique_clauses[r] = [c for c in block_clauses[r] if counts[c] == 1]
    def compute_unique_clauses():
        unique = []
        for idx in selected:
            u = []
            for c in block_clauses[idx]:
                if counts[c] == 1:
                    u.append(c)
            unique.append(u)
        return unique
        
    unique_clauses = compute_unique_clauses()
    
    # Helper to update counts and unique_clauses after a change
    # This is complex to do incrementally. 
    # Given small size, we can recompute counts and unique_clauses from scratch after each swap/deletion.
    # Cost: O(|S| * 56) for counts. O(|S| * 56) for unique.
    # |S| ~ 300. 300*56 = 16800 ops. Very fast.
    
    def recalc():
        nonlocal counts, unique_clauses
        counts = [0] * n_t_subs
        for idx in selected:
            for c in block_clauses[idx]:
                counts[c] += 1
        unique_clauses = []
        for idx in selected:
            u = []
            for c in block_clauses[idx]:
                if counts[c] == 1:
                    u.append(c)
            unique_clauses.append(u)
            
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Deletion
        changed = True
        while changed:
            changed = False
            new_selected = []
            for i, idx in enumerate(selected):
                if not unique_clauses[i]:
                    # Redundant
                    changed = True
                    improved = True
                else:
                    new_selected.append(idx)
            if changed:
                selected = new_selected
                recalc()
                
        # Phase 2: Swap
        # Find best swap
        best_b = -1
        best_replacements = []
        best_gain = 0
        
        selected_set = set(selected)
        
        # Iterate through all blocks not in selected
        for b in range(n_k_blocks):
            if b in selected_set:
                continue
                
            set_b = block_clause_sets[b]
            replacements = []
            
            # Check which selected blocks become redundant
            for i, r in enumerate(selected):
                u_r = unique_clauses[i]
                # Check if u_r is subset of set_b
                # If u_r is empty, it would have been deleted in Phase 1
                if all(c in set_b for c in u_r):
                    replacements.append(i)
                    
            if len(replacements) > 1:
                gain = len(replacements) - 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
                    best_replacements = replacements
                    
        if best_gain > 0:
            # Apply swap
            # Remove best_replacements
            new_selected = []
            for i, idx in enumerate(selected):
                if i not in best_replacements:
                    new_selected.append(idx)
            new_selected.append(best_b)
            selected = new_selected
            recalc()
            improved = True
            # Loop back to Deletion phase

    return [list(k_blocks[i]) for i in selected]
