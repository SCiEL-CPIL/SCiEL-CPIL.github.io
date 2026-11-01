import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            
    # Weighted Greedy Construction: prioritize subsets that are hard to cover
    selected_indices = []
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    
    # Weight each t-subset by how many blocks cover it (inverse frequency)
    # This makes rare subsets more important
    block_counts = Cover.sum(axis=0).astype(np.float64)
    # Avoid division by zero, use 1/(count+1) so rarer subsets get higher weight
    weights = 1.0 / (block_counts + 1.0)
    weights = weights / weights.sum()  # Normalize
    
    # Precompute weighted coverage for each block
    weighted_cover = Cover.astype(np.float64) * weights
    
    while np.sum(uncovered) > 0 and time.time() < deadline - 30.0:
        gains = weighted_cover @ uncovered
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
        
    # If we ran out of time during greedy, we have an invalid covering.
    # But greedy is usually fast. Let's assume it finishes.
    
    # Local Optimization with Add-Minimize moves
    if time.time() < deadline - 10.0:
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected_indices:
            cover_counts += Cover[idx]
            
        selected_indices, cover_counts = _local_optimize_v3(selected_indices, Cover, cover_counts, n_blocks, deadline - 5.0)
        
    blocks = [all_kblocks[i] for i in selected_indices]
    return blocks

def _local_optimize_v3(selected_indices, Cover, cover_counts, n_blocks, deadline):
    """
    Enhanced local optimization with Add-Minimize moves.
    1. Remove redundant blocks.
    2. Try to replace blocks with better ones.
    3. Add-Minimize: temporarily add a block, then remove multiple blocks.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        
        # Phase 1: Remove redundant blocks
        new_selected_indices = []
        for idx in selected_indices:
            if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                new_selected_indices.append(idx)
            else:
                cover_counts -= Cover[idx]
                changed = True
        selected_indices = new_selected_indices
        
        if time.time() > deadline:
            break
            
        # Phase 2: Try replacements
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected_indices = []
            for idx in selected_indices:
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    unique_tsubs = np.where((Cover[idx] == 1) & (cover_counts == 1))[0]
                    
                    if len(unique_tsubs) == 0:
                        new_selected_indices.append(idx)
                        continue
                    
                    cand_mask = np.all(Cover[:, unique_tsubs] == 1, axis=1)
                    
                    sel_mask = np.zeros(n_blocks, dtype=bool)
                    for s_idx in selected_indices:
                        sel_mask[s_idx] = True
                    cand_mask[sel_mask] = False
                    
                    candidates = np.where(cand_mask)[0]
                    
                    if len(candidates) > 0:
                        gains = Cover[candidates].sum(axis=1)
                        best_candidate = candidates[np.argmax(gains)]
                        
                        cover_counts -= Cover[idx]
                        cover_counts += Cover[best_candidate]
                        
                        new_selected_indices.append(best_candidate)
                        improved = True
                    else:
                        new_selected_indices.append(idx)
                else:
                    if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                        new_selected_indices.append(idx)
                    else:
                        cover_counts -= Cover[idx]
                        changed = True
                        improved = True
            
            selected_indices = new_selected_indices
            
        # Phase 3: Add-Minimize moves
        # Temporarily add a block, then try to remove 2+ blocks
        if time.time() < deadline - 10.0:
            add_minimize_improved = True
            while add_minimize_improved and time.time() < deadline - 10.0:
                add_minimize_improved = False
                best_removal_count = 0
                best_add_idx = None
                best_remove_indices = None
                
                # Try adding each non-selected block
                selected_set = set(selected_indices)
                for add_idx in range(n_blocks):
                    if add_idx in selected_set:
                        continue
                    
                    # Simulate adding
                    sim_cover_counts = cover_counts + Cover[add_idx]
                    
                    # Try to remove as many blocks as possible
                    removable = []
                    for idx in selected_indices:
                        if not np.any((Cover[idx] == 1) & (sim_cover_counts == 1)):
                            removable.append(idx)
                    
                    if len(removable) > best_removal_count:
                        best_removal_count = len(removable)
                        best_add_idx = add_idx
                        best_remove_indices = removable
                
                if best_removal_count >= 2 and best_add_idx is not None:
                    # Apply the move
                    for idx in best_remove_indices:
                        cover_counts -= Cover[idx]
                    cover_counts += Cover[best_add_idx]
                    
                    new_selected = [idx for idx in selected_indices if idx not in set(best_remove_indices)]
                    new_selected.append(best_add_idx)
                    selected_indices = new_selected
                    changed = True
                    add_minimize_improved = True
                else:
                    break
            
    return selected_indices, cover_counts
