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
    # M[i, j] = 1 if block i covers t-subset j
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    def time_left():
        return deadline - time.monotonic()

    # Helper: Greedy construction with diversity tie-breaking
    def greedy_construction(rng, div_factor=0.0):
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        selected_indices = []
        
        while np.any(uncovered):
            if time_left() < 5.0:
                break
            # Gains: number of uncovered tsubs covered by each block
            gains = M @ uncovered
            
            # Apply diversity penalty: prefer blocks that are "far" from already selected ones?
            # Simpler: just add random noise to gains to break ties differently
            if div_factor > 0:
                noise = rng.random(num_kblocks) * div_factor
                gains = gains + noise
            
            best_idx = int(np.argmax(gains))
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
        if not indices_set:
            return []
        
        # Calculate counts: how many blocks cover each tsub
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        changed = True
        while changed:
            changed = False
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

    # Helper: Local Search with Swaps
    def local_search(indices):
        indices_set = set(indices)
        if not indices_set:
            return []
        
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        # First, remove redundancies
        indices_set = set(remove_redundant(list(indices_set)))
        if not indices_set:
            return []
        
        # Recalculate counts after removal
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
            
        improved = True
        while improved:
            improved = False
            if time_left() < 5.0:
                break
                
            # Try to swap each block
            idx_list = list(indices_set)
            random.shuffle(idx_list)
            
            for b_idx in idx_list:
                if time_left() < 5.0:
                    return list(indices_set)
                
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                
                # Find critical tsubs: those covered ONLY by b_idx
                critical_mask = counts == 1
                critical_indices = covered_indices[critical_mask[covered_indices]]
                
                if len(critical_indices) == 0:
                    # Redundant, remove it
                    indices_set.remove(b_idx)
                    counts -= row
                    improved = True
                    break
                
                # We need a replacement block that covers all critical_indices
                # And is not in indices_set
                # Optimization: If too many criticals, skip
                if len(critical_indices) > 20: 
                    continue

                # Find candidates: blocks that cover all critical_indices
                # M[:, critical_indices].all(axis=1)
                candidates_mask = M[:, critical_indices].all(axis=1)
                candidates = np.where(candidates_mask)[0]
                
                # Filter out current blocks
                current_list = list(indices_set)
                current_set = set(current_list)
                candidates = [c for c in candidates if c not in current_set]
                
                if not candidates:
                    continue
                
                # Evaluate candidates
                # We want to minimize the number of NEW tsubs covered? No, that doesn't help reduction directly.
                # Actually, any valid swap maintains coverage. 
                # However, a "better" swap might create more redundancy, allowing future removals.
                # Heuristic: Pick a candidate that shares the most tsubs with the current set (max overlap), 
                # or simply pick a random one to explore different local minima.
                # Let's try to pick one that maximizes overlap with other blocks to increase redundancy potential.
                
                best_cand = None
                best_score = -1
                
                # To save time, only check a subset of candidates if there are many
                if len(candidates) > 20:
                    candidates = random.sample(candidates, 20)
                
                for c_idx in candidates:
                    # Score: number of tsubs covered by c_idx that are already covered by other blocks
                    # This indicates how "redundant" the new block will be.
                    # If a block is very redundant, it might be removed later.
                    # Alternatively, we just want to keep the size same, but enable future removals.
                    # Let's try: maximize overlap with the union of other blocks.
                    # This is expensive. 
                    # Simpler heuristic: Random selection often works well in ILS.
                    # Or: Pick the one with max gain (covers most uncovered? No, nothing is uncovered).
                    
                    # Let's try: Pick the candidate that minimizes the "uniqueness" of the other blocks it affects?
                    # Too complex. Let's stick to random selection from top-k overlaps or just random.
                    # Random is fast and provides diversity.
                    best_cand = random.choice(candidates)
                    break # Just pick one random candidate for speed
                
                if best_cand is not None:
                    # Perform swap
                    indices_set.remove(b_idx)
                    counts -= M[b_idx]
                    indices_set.add(best_cand)
                    counts += M[best_cand]
                    improved = True
                    break
                    
        return list(indices_set)

    # Phase 1: Initial Greedy
    rng = random.Random(42)
    best_solution = greedy_construction(rng)
    best_len = len(best_solution)
    
    # Phase 2: ILS
    current_sol = best_solution[:]
    
    for _ in range(2000): # Max iterations, bounded by time
        if time_left() < 10.0:
            break
            
        # Local Search
        polished_sol = local_search(current_sol)
        
        if len(polished_sol) < best_len:
            best_len = len(polished_sol)
            best_solution = polished_sol[:]
            current_sol = polished_sol[:] # Continue from new best
            
        # Perturbation
        # Remove 3-7 random blocks
        n_remove = min(3 + rng.randint(0, 4), len(current_sol))
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
                if time_left() < 5.0:
                    break
                gains = M @ uncovered_mask
                best_idx = int(np.argmax(gains))
                if gains[best_idx] == 0:
                    break
                current_sol.append(best_idx)
                counts += M[best_idx]
                uncovered_mask = counts == 0
                
            # If repair failed or got too big, revert to best
            if len(current_sol) > best_len + 15:
                current_sol = best_solution[:]
            else:
                # Try to polish the repaired solution
                current_sol = local_search(current_sol)
                if len(current_sol) < best_len:
                    best_len = len(current_sol)
                    best_solution = current_sol[:]

    if best_solution is None:
        return []
        
    return get_blocks(best_solution)
