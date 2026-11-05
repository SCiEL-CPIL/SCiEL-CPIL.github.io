import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    # M[i, j] = 1 if all_kblocks[i] covers all_tsubs[j]
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    
    # Precompute the list of tsub indices for each block
    # This is useful for updating counts quickly if we use a different approach, 
    # but for now, we rely on M.
    # To speed up matrix construction, we can use a loop.
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)

    # Helper: Greedy construction using matrix
    def greedy_construction():
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        selected_indices = []
        
        while np.any(uncovered):
            if time.time() > deadline - 5.0:
                break
            # Gains: number of uncovered tsubs covered by each block
            gains = M @ uncovered
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            # Update uncovered
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
            
        return selected_indices

    # Helper: Remove redundant blocks using matrix
    def remove_redundant(indices):
        indices_set = set(indices)
        if not indices_set:
            return []
        # Calculate counts
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        changed = True
        while changed:
            changed = False
            # Find blocks that are redundant
            # A block is redundant if for all tsubs it covers, count > 1
            to_remove = []
            for idx in indices_set:
                row = M[idx]
                covered = np.where(row == 1)[0]
                # If all covered tsubs have count > 1, it's redundant
                if np.all(counts[covered] > 1):
                    to_remove.append(idx)
            
            for idx in to_remove:
                indices_set.remove(idx)
                counts -= M[idx]
                changed = True
                
        return list(indices_set)

    # Helper: Local Search
    # We try to swap a block with another block that covers its critical t-subsets
    # and potentially allows other blocks to be removed.
    def local_search(indices):
        indices_set = set(indices)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        improved = True
        while improved:
            improved = False
            # First pass: remove redundant
            new_indices = remove_redundant(list(indices_set))
            if len(new_indices) < len(indices_set):
                indices_set = set(new_indices)
                # Recalculate counts
                counts = np.zeros(num_tsubs, dtype=np.int32)
                for idx in indices_set:
                    counts += M[idx]
                improved = True
                continue
            
            # Second pass: try swaps
            # We iterate through blocks and try to replace one with a better one
            idx_list = list(indices_set)
            rng.shuffle(idx_list)
            
            for b_idx in idx_list:
                if time.time() > deadline - 5.0:
                    return list(indices_set)
                
                # Critical tsubs: those covered ONLY by b_idx
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                critical_mask = counts == 1
                critical_indices = covered_indices[critical_mask[covered_indices]]
                
                if len(critical_indices) == 0:
                    # Redundant, should have been removed
                    indices_set.remove(b_idx)
                    counts -= row
                    improved = True
                    break
                
                # We need a block that covers all critical_indices
                # candidates_mask = M[:, critical_indices].all(axis=1)
                # This is expensive if critical_indices is large.
                # If critical_indices is too large, it's unlikely a single block covers all.
                if len(critical_indices) > 10:
                    continue
                
                candidates_mask = M[:, critical_indices].all(axis=1)
                candidates = np.where(candidates_mask)[0]
                
                # Filter out current blocks
                current_set = set(indices_set)
                candidates = [c for c in candidates if c not in current_set]
                
                if len(candidates) > 0:
                    # Pick a random candidate
                    c_idx = candidates[rng.randrange(len(candidates))]
                    
                    # Perform swap
                    indices_set.remove(b_idx)
                    counts -= M[b_idx]
                    indices_set.add(c_idx)
                    counts += M[c_idx]
                    improved = True
                    break
                    
        return list(indices_set)

    # Phase 1: Initial Greedy
    initial_sol = greedy_construction()
    if initial_sol:
        initial_sol = remove_redundant(initial_sol)
        best_solution = initial_sol
        best_len = len(initial_sol)

    # Phase 2: Iterated Local Search
    current_sol = best_solution[:]
    
    # We will run ILS until time runs out
    while time.time() < deadline - 10.0:
        # Local Search
        polished_sol = local_search(current_sol)
        
        if len(polished_sol) < best_len:
            best_len = len(polished_sol)
            best_solution = polished_sol[:]
            current_sol = polished_sol[:]
            
        # Perturbation
        # Remove a random block and add a random block? 
        # Or just swap a random block with a random other block?
        if current_sol:
            # Swap a random block with a random different block
            idx_to_swap = rng.randrange(len(current_sol))
            old_block = current_sol[idx_to_swap]
            
            # Find a random block not in current_sol
            all_blocks_set = set(current_sol)
            candidate_blocks = [i for i in range(num_kblocks) if i not in all_blocks_set]
            if candidate_blocks:
                new_block = rng.choice(candidate_blocks)
                current_sol[idx_to_swap] = new_block
        
        # Re-polish after perturbation
        polished_sol = local_search(current_sol)
        
        # Accept if better or same (to avoid getting stuck)
        if len(polished_sol) <= best_len:
            if len(polished_sol) < best_len:
                best_len = len(polished_sol)
                best_solution = polished_sol[:]
            current_sol = polished_sol[:]
        else:
            # Revert to best or keep current? 
            # Standard ILS keeps the perturbed state to explore further, 
            # but we should track the global best.
            # Let's keep the perturbed state as current_sol to explore its neighborhood
            pass

    # Final cleanup
    if best_solution:
        best_solution = remove_redundant(best_solution)
    
    if best_solution is None:
        # Fallback if no time was left for restarts
        rng = random.Random(0)
        best_solution = greedy_construction()
        best_solution = remove_redundant(best_solution)
        
    return get_blocks(best_solution)

