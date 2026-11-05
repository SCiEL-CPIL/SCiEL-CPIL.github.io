import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    # Using uint8 for efficiency
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    
    # Also build reverse map: for each tsub, which blocks cover it?
    # This is useful for swap heuristics
    # But M is dense enough, we can just use M.
    
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)

    # Helper: Greedy construction
    def greedy_construction():
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        selected_indices = []
        
        while np.any(uncovered):
            if time.monotonic() > deadline - 5.0:
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

    # Helper: Remove redundant blocks
    def remove_redundant(indices):
        indices_set = set(indices)
        # Calculate counts
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        changed = True
        while changed:
            changed = False
            # Find blocks that are redundant
            # A block is redundant if for all tsubs it covers, count > 1
            # We iterate and check
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

    # Helper: Try to swap blocks to reduce count
    # This is the core local search step
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
            # For each block, check if it can be swapped
            # To make it faster, we only try a subset or random order
            idx_list = list(indices_set)
            rng.shuffle(idx_list)
            
            for b_idx in idx_list:
                if time.monotonic() > deadline - 5.0:
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
                # And is not in indices_set
                # Check M[:, critical_indices].all(axis=1)
                # This is expensive if done for every block.
                # Optimization: Only check if critical_indices is small?
                # Or use a precomputed list of blocks for each tsub?
                
                # Let's try to find candidates
                # Intersection of sets of blocks covering each critical tsub
                # Since M is dense, we can do:
                # candidates_mask = M[:, critical_indices].all(axis=1)
                # candidates_mask &= ~np.isin(np.arange(num_kblocks), list(indices_set))
                
                # To speed up, we can just pick a random candidate from the union? No, must cover ALL.
                # Let's just do the numpy op, it's vectorized.
                if len(critical_indices) > 50:
                    # Too many constraints, unlikely to find a single block covering all
                    # Skip this block for swap
                    continue
                    
                candidates_mask = M[:, critical_indices].all(axis=1)
                # Exclude current blocks
                # Creating a boolean array of length num_kblocks every time is slow.
                # We can maintain a 'selected' boolean array.
                # But we are in a loop.
                
                # Let's optimize by maintaining selected_mask
                # This requires passing it around.
                
                # For now, let's just check candidates
                candidates = np.where(candidates_mask)[0]
                # Filter out current
                current_list = list(indices_set)
                current_set = set(current_list)
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
        best_solution = initial_sol
        best_len = len(initial_sol)

    # Phase 2: ILS
    current_sol = best_solution[:]
    
    # Precompute selected_mask for faster swap checks?
    # Let's try a simpler loop first.
    
    for _ in range(1000): # Max iterations, bounded by time
        if time.monotonic() > deadline - 10.0:
            break
            
        # Local Search
        polished_sol = local_search(current_sol)
        
        if len(polished_sol) < best_len:
            best_len = len(polished_sol)
            best_solution = polished_sol[:]
            
        # Perturbation
        # Remove 5-10 random blocks
        n_remove = min(5 + rng.randint(0, 5), len(current_sol))
        if n_remove > 0:
            to_remove = rng.sample(current_sol, n_remove)
            current_sol_set = set(current_sol)
            for b in to_remove:
                current_sol_set.remove(b)
            current_sol = list(current_sol_set)
            
            # Greedy repair
            # Calculate current coverage
            counts = np.zeros(num_tsubs, dtype=np.int32)
            for idx in current_sol:
                counts += M[idx]
                
            uncovered_mask = counts == 0
            
            while np.any(uncovered_mask):
                if time.monotonic() > deadline - 5.0:
                    break
                gains = M @ uncovered_mask
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                current_sol.append(best_idx)
                counts += M[best_idx]
                uncovered_mask = counts == 0
                
            # If repair failed or got too big, revert to best
            if len(current_sol) > best_len + 10:
                current_sol = best_solution[:]

    if best_solution is None:
        return []
        
    return get_blocks(best_solution)
