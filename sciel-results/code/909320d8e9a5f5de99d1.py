import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute all k-blocks and t-subsets
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix: Cover[i, j] = 1 if t-subset j is in block i
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1

    # Greedy construction
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected_indices = []
    greedy_deadline = deadline - 35.0  # Leave time for ILS
    
    while np.sum(uncovered) > 0:
        if time.time() > greedy_deadline:
            break
        gains = Cover @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)

    # Compute cover counts for current solution
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in selected_indices:
        cover_counts += Cover[idx]
    
    # Local optimization: remove redundancies and try replacements
    def _remove_redundant(selected_indices, cover_counts, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for idx in selected_indices:
                # Block is essential if it uniquely covers any t-subset
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_selected.append(idx)
                else:
                    cover_counts -= Cover[idx]
                    changed = True
            selected_indices = new_selected
        return selected_indices, cover_counts

    def _try_replacement(selected_indices, cover_counts, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = []
            for idx in selected_indices:
                # Check if essential
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_selected.append(idx)
                    continue
                
                # Try to remove this block
                cover_counts -= Cover[idx]
                uncovered_tsubs = np.where(cover_counts == 0)[0]
                
                if len(uncovered_tsubs) == 0:
                    # Redundant, keep removed
                    improved = True
                    continue
                
                # Try to find a replacement block that covers all uncovered tsubs
                if len(uncovered_tsubs) <= 15:
                    best_rep = -1
                    best_gain = 0
                    for cand in range(n_blocks):
                        if cand in selected_indices:
                            continue
                        if np.all(Cover[cand, uncovered_tsubs] == 1):
                            gain = np.sum(Cover[cand] == 1)
                            if gain > best_gain:
                                best_gain = gain
                                best_rep = cand
                    if best_rep >= 0:
                        cover_counts += Cover[best_rep]
                        new_selected.append(best_rep)
                        improved = True
                        continue
                
                # No replacement found, restore
                cover_counts += Cover[idx]
                new_selected.append(idx)
            
            selected_indices = new_selected
        return selected_indices, cover_counts

    def _local_optimize(selected_indices, cover_counts, deadline):
        selected_indices, cover_counts = _remove_redundant(selected_indices, cover_counts, deadline)
        selected_indices, cover_counts = _try_replacement(selected_indices, cover_counts, deadline)
        selected_indices, cover_counts = _remove_redundant(selected_indices, cover_counts, deadline)
        return selected_indices, cover_counts

    # Initial local optimization
    selected_indices, cover_counts = _local_optimize(selected_indices, cover_counts, deadline - 10.0)
    best_solution = list(selected_indices)
    best_count = len(best_solution)
    
    # ILS: Iterated Local Search with constrained swaps
    ils_deadline = deadline - 5.0
    random.seed(42)
    
    while time.time() < ils_deadline:
        # Perturb: swap one or two blocks with random ones
        if len(selected_indices) < 2:
            break
        
        perturbed = list(selected_indices)
        n_swaps = random.choice([1, 1, 2])  # Mostly single swaps
        
        for _ in range(n_swaps):
            if len(perturbed) == 0:
                break
            # Pick a random block to remove
            remove_pos = random.randint(0, len(perturbed) - 1)
            removed_idx = perturbed[remove_pos]
            perturbed.pop(remove_pos)
            
            # Recompute cover counts after removal
            temp_counts = np.zeros(n_tsubs, dtype=np.int32)
            for idx in perturbed:
                temp_counts += Cover[idx]
            
            # Find uncovered tsubs
            uncovered_tsubs = np.where(temp_counts == 0)[0]
            
            if len(uncovered_tsubs) == 0:
                # Removal was safe, don't add anything back
                continue
            
            # Try to find a block that covers all uncovered tsubs
            if len(uncovered_tsubs) <= 20:
                candidates = [c for c in range(n_blocks) if c not in perturbed]
                random.shuffle(candidates)
                added = False
                for cand in candidates:
                    if np.all(Cover[cand, uncovered_tsubs] == 1):
                        perturbed.append(cand)
                        added = True
                        break
                if not added:
                    # Restore the removed block
                    perturbed.append(removed_idx)
            else:
                # Too many uncovered, restore
                perturbed.append(removed_idx)
        
        # Local optimize the perturbed solution
        if time.time() > ils_deadline - 2.0:
            break
        
        # Compute cover counts for perturbed
        pert_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in perturbed:
            pert_counts += Cover[idx]
        
        # Verify it's a valid cover
        if np.any(pert_counts == 0):
            continue
        
        perturbed, pert_counts = _local_optimize(perturbed, pert_counts, ils_deadline - 1.0)
        
        # Verify validity again
        if np.any(pert_counts == 0):
            continue
        
        if len(perturbed) < best_count:
            best_count = len(perturbed)
            best_solution = list(perturbed)
            # Update selected_indices to the new best for future perturbations
            selected_indices = list(perturbed)
            cover_counts = pert_counts.copy()
    
    return [all_kblocks[i] for i in best_solution]
