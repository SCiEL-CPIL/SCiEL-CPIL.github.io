import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for i, B in enumerate(all_kblocks):
        B_set = set(B)
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            
    # 1. Weighted Greedy Construction
    # Weights are initialized based on scarcity (1/degree) and updated dynamically
    selected_indices = []
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    
    # Precompute degree of each t-subset (how many blocks cover it)
    # This is the sum of the column
    tsub_degrees = Cover.sum(axis=0).astype(np.float64)
    # Avoid division by zero
    tsub_degrees[tsub_degrees == 0] = 1.0
    tsub_weights = 1.0 / tsub_degrees
    
    # Current weights for t-subsets
    current_weights = tsub_weights.copy()
    
    remaining_time_limit = deadline - 30.0  # Leave 30s for local optimization
    
    while np.sum(uncovered) > 0:
        if time.time() > remaining_time_limit:
            break
            
        # Gain = sum of weights of uncovered t-subsets covered by block i
        # We only care about uncovered ones
        gains = (Cover * current_weights).sum(axis=1)
        
        # Only consider blocks that cover at least one uncovered t-subset
        # Actually, gains will be 0 for blocks that don't cover any uncovered if weights are 0 for covered?
        # No, current_weights are non-zero for all.
        # We should mask out covered t-subsets in the calculation or just let gains be high for those covering uncovered.
        # Let's re-calculate gains properly: sum of weights of UNCOVERED t-subsets in block i
        # gains = (Cover * (current_weights * uncovered)).sum(axis=1)
        active_weights = current_weights * uncovered
        gains = (Cover * active_weights).sum(axis=1)
        
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        
        # Update uncovered
        newly_covered = Cover[best_idx] & uncovered
        uncovered = uncovered & (Cover[best_idx] == 0)
        
        # Update weights: increase weight of uncovered t-subsets that are "scarce"
        # A common heuristic is to increase weight of uncovered items.
        # Let's increase weight of newly uncovered items? No, they are now covered.
        # Increase weight of items that are still uncovered?
        # Standard weighted set cover: w_j <- w_j + 1/deg(j) for uncovered j?
        # Or just increase weights of all uncovered items to prioritize them.
        # Let's try: w_j = w_j * 1.1 for uncovered j? Or add a constant?
        # Let's stick to a simple dynamic adjustment:
        # Increase weight of t-subsets that are still uncovered.
        current_weights[uncovered == 1] += 0.1
        
    # 2. Post-processing: Remove Redundant Blocks
    if time.time() < deadline - 10.0:
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected_indices:
            cover_counts += Cover[idx]
            
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            new_selected_indices = []
            for idx in selected_indices:
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_selected_indices.append(idx)
                else:
                    cover_counts -= Cover[idx]
                    changed = True
            selected_indices = new_selected_indices
            
    # 3. Local Optimization with Simulated Annealing
    if time.time() < deadline - 2.0:
        selected_indices = _local_optimize(
            selected_indices, Cover, n_tsubs, 
            deadline - 2.0, start_time
        )
    
    blocks = [all_kblocks[i] for i in selected_indices]
    return blocks

def _local_optimize(selected_indices, Cover, n_tsubs, deadline, start_time):
    """Simulated annealing with remove/swap/add moves to minimize block count."""
    n_blocks_total = Cover.shape[0]
    current_indices = list(selected_indices)
    current_count = len(current_indices)
    
    # Compute initial cover counts
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in current_indices:
        cover_counts += Cover[idx]
    
    best_indices = list(current_indices)
    best_count = current_count
    
    # SA parameters
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    
    rng = np.random.default_rng(42)
    
    # Precompute list of selected indices for fast access
    current_set = set(current_indices)
    
    while time.time() < deadline and T > min_T:
        # Choose move type: remove (20%), swap (60%), add (20%)
        move_type = rng.random()
        
        if move_type < 0.2 and len(current_indices) > 1:
            # Try removing a block
            i = rng.integers(0, len(current_indices))
            idx_to_remove = current_indices[i]
            
            # Check if removing this block keeps all t-subsets covered
            removable = not np.any((Cover[idx_to_remove] == 1) & (cover_counts == 1))
            
            if removable:
                # Remove it
                cover_counts -= Cover[idx_to_remove]
                current_indices.pop(i)
                current_set.remove(idx_to_remove)
                current_count -= 1
                
                if current_count < best_count:
                    best_indices = list(current_indices)
                    best_count = current_count
            else:
                # Roll back (didn't change)
                pass
                
        elif move_type < 0.8 and len(current_indices) > 0:
            # Try swap: replace one block with another
            i = rng.integers(0, len(current_indices))
            idx_current = current_indices[i]
            
            # Find a candidate to swap with
            # Prefer blocks that cover many uncovered-or-singly-covered t-subsets
            # But we need a valid swap: after swap, all tsubs still covered
            
            # Simple approach: try random other blocks
            num_attempts = 5
            swapped = False
            for _ in range(num_attempts):
                idx_new = rng.integers(0, n_blocks_total)
                if idx_new == idx_current or idx_new in current_set:
                    continue
                
                # Check if swapping is valid
                # After removing idx_current, cover_counts decreases
                # After adding idx_new, cover_counts increases
                # We need: for all j, (cover_counts[j] - Cover[idx_current,j] + Cover[idx_new,j]) >= 1
                
                # Equivalent: for all j where Cover[idx_current,j]==1 and cover_counts[j]==1,
                # we need Cover[idx_new,j]==1
                
                critical = (Cover[idx_current] == 1) & (cover_counts == 1)
                if not np.any(critical):
                    # Safe to swap (removal alone is safe)
                    valid = True
                else:
                    # Check if idx_new covers all critical tsubs
                    valid = np.all(Cover[idx_new][critical] == 1)
                
                if valid:
                    # Perform swap
                    cover_counts -= Cover[idx_current]
                    cover_counts += Cover[idx_new]
                    current_indices[i] = idx_new
                    current_set.remove(idx_current)
                    current_set.add(idx_new)
                    swapped = True
                    break
            
        else:
            # Try add: add a new block (might enable removals later)
            # Only add if we are stuck or to escape local minima?
            # Adding increases count, so we should only accept if it leads to a better state quickly.
            # For simplicity in this SA, we might skip add or use it rarely.
            # Let's try adding a block that covers many "critical" (count==1) t-subsets?
            # No, that doesn't help minimization directly.
            # Let's skip add for now to keep it simple, or implement a 2-exchange later.
            pass
        
        # Cool down
        T *= cooling
        
    return best_indices
