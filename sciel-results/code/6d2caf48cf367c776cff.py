import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Precompute all k-blocks and t-subsets
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # 2. Build coverage matrix
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for i, B in enumerate(all_kblocks):
        B_set = set(B)
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            
    # 3. Greedy Construction
    selected_indices = []
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    remaining_time_limit = deadline - 20.0  # Leave 20s for local optimization
    
    while np.sum(uncovered) > 0:
        if time.time() > remaining_time_limit:
            break
            
        gains = Cover @ uncovered
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
    
    # 4. Post-processing: Remove Redundant Blocks
    if time.time() < deadline - 5.0:
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected_indices:
            cover_counts += Cover[idx]
            
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected_indices = []
            for idx in selected_indices:
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_selected_indices.append(idx)
                else:
                    cover_counts -= Cover[idx]
                    changed = True
            selected_indices = new_selected_indices
    
    # 5. Local Optimization with Simulated Annealing
    if time.time() < deadline - 2.0:
        selected_indices = _local_optimize(
            selected_indices, Cover, n_tsubs, 
            deadline - 2.0, start_time
        )
    
    blocks = [all_kblocks[i] for i in selected_indices]
    return blocks

def _local_optimize(selected_indices, Cover, n_tsubs, deadline, start_time):
    """Simulated annealing with remove/swap moves to minimize block count."""
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
    
    while time.time() < deadline and T > min_T:
        # Choose move type: remove (30%) or swap (70%)
        move_type = rng.random()
        
        if move_type < 0.3 and len(current_indices) > 1:
            # Try removing a block
            i = rng.integers(0, len(current_indices))
            idx_to_remove = current_indices[i]
            
            # Check if removing this block keeps all t-subsets covered
            # A block is removable if no t-subset it covers has count == 1
            removable = not np.any((Cover[idx_to_remove] == 1) & (cover_counts == 1))
            
            if removable:
                # Remove it
                cover_counts -= Cover[idx_to_remove]
                new_indices = current_indices[:i] + current_indices[i+1:]
                new_count = current_count - 1
                
                if new_count < best_count:
                    best_indices = list(new_indices)
                    best_count = new_count
                    current_indices = new_indices
                    current_count = new_count
                else:
                    # Roll back
                    cover_counts += Cover[idx_to_remove]
            else:
                # Can't remove, try swap
                pass
        
        # Try swap: replace one block with another
        if len(current_indices) > 0:
            i = rng.integers(0, len(current_indices))
            idx_current = current_indices[i]
            
            # Find a candidate to swap with
            # Prefer blocks that cover many uncovered-or-singly-covered t-subsets
            # But we need a valid swap: after swap, all tsubs still covered
            
            # Simple approach: try random other blocks
            num_attempts = 3
            for _ in range(num_attempts):
                idx_new = rng.integers(0, n_blocks_total)
                if idx_new == idx_current:
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
                    new_indices = list(current_indices)
                    new_indices[i] = idx_new
                    
                    # Count doesn't change, but might enable future removals
                    # Accept with some probability based on whether it enables removal
                    # For now, just accept if it doesn't hurt (count same)
                    current_indices = new_indices
                    break
        
        # Cool down
        T *= cooling
    
    return best_indices
