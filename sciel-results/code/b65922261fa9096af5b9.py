import time
import numpy as np
import random
import itertools

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs) using uint8
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    
    # Also build reverse map: for each tsub, which blocks cover it?
    t_to_blocks = [[] for _ in range(num_tsubs)]
    
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1
            t_to_blocks[idx].append(i)
    
    t_to_blocks = [np.array(lst, dtype=np.int32) for lst in t_to_blocks]

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)

    # Helper: Greedy construction with lookahead
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
                if len(covered) == 0:
                    continue
                # If all covered tsubs have count > 1, it's redundant
                if np.all(counts[covered] > 1):
                    to_remove.append(idx)
            
            for idx in to_remove:
                indices_set.remove(idx)
                counts -= M[idx]
                changed = True
                
        return list(indices_set)

    # Helper: Dual local search - optimize coverage counts directly
    def dual_local_search(indices, max_time=None):
        if max_time is None:
            max_time = deadline - 5.0
        
        indices_set = set(indices)
        if not indices_set:
            return []
        
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_set:
            counts += M[idx]
        
        # Precompute: for each block, the set of tsubs it covers
        block_covered = {}
        for idx in indices_set:
            row = M[idx]
            block_covered[idx] = np.where(row == 1)[0]
        
        improved = True
        while improved:
            improved = False
            
            # Pass 1: Remove redundant blocks
            to_remove = []
            for idx in list(indices_set):
                covered = block_covered[idx]
                if len(covered) == 0:
                    to_remove.append(idx)
                    continue
                if np.all(counts[covered] > 1):
                    to_remove.append(idx)
            
            for idx in to_remove:
                indices_set.remove(idx)
                counts -= M[idx]
                del block_covered[idx]
                improved = True
            
            if not improved:
                # Pass 2: Try swaps to improve minimum coverage
                # For each block, find critical tsubs (count == 1)
                # Try to swap with a block that maximizes min coverage after swap
                idx_list = list(indices_set)
                rng.shuffle(idx_list)
                
                for b_idx in idx_list:
                    if time.monotonic() > max_time:
                        return list(indices_set)
                    
                    covered = block_covered[b_idx]
                    if len(covered) == 0:
                        continue
                    
                    # Critical tsubs: those with count == 1 that this block covers
                    critical_mask = counts == 1
                    critical_indices = covered[critical_mask[covered]]
                    
                    if len(critical_indices) == 0:
                        # Redundant - should have been caught in pass 1
                        indices_set.remove(b_idx)
                        counts -= M[b_idx]
                        del block_covered[b_idx]
                        improved = True
                        break
                    
                    # If too many critical tsubs, skip (hard to find replacement)
                    if len(critical_indices) > 20:
                        continue
                    
                    # Find candidates that cover all critical tsubs
                    # and are not in indices_set
                    candidates_mask = M[:, critical_indices].all(axis=1)
                    candidates = np.where(candidates_mask)[0]
                    
                    # Filter out current blocks
                    current_set = indices_set
                    candidates = [c for c in candidates if c not in current_set]
                    
                    if len(candidates) == 0:
                        continue
                    
                    # Score candidates by how much they improve min coverage
                    # After removing b_idx, critical tsubs will have count 0
                    # After adding candidate, they'll have count 1
                    # We want to maximize the minimum count after swap
                    # Approximate: count how many tsubs the candidate covers that are currently under-covered
                    best_candidate = None
                    best_score = -1
                    
                    # Sample candidates for speed
                    sample_size = min(15, len(candidates))
                    sampled = rng.sample(candidates, sample_size)
                    
                    for cand in sampled:
                        cand_covered = np.where(M[cand] == 1)[0]
                        
                        # Count tsubs where candidate helps:
                        # - Covers critical tsubs (required)
                        # - Covers tsubs with count 1 (increases to 2)
                        # - Avoids creating new count==1 situations
                        help_count = np.sum(counts[cand_covered] == 1)
                        
                        # Penalty: tsubs that would go from count 2 to 1 after removing b_idx
                        # and are covered by candidate
                        # This is complex, use simple heuristic
                        score = help_count
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = cand
                    
                    if best_candidate is not None and best_score >= 0:
                        # Perform swap
                        indices_set.remove(b_idx)
                        counts -= M[b_idx]
                        del block_covered[b_idx]
                        
                        indices_set.add(best_candidate)
                        counts += M[best_candidate]
                        block_covered[best_candidate] = np.where(M[best_candidate] == 1)[0]
                        
                        improved = True
                        break
        
        return list(indices_set)

    # Phase 1: Initial Greedy
    initial_sol = greedy_construction()
    if initial_sol:
        best_solution = initial_sol
        best_len = len(initial_sol)

    # Phase 2: Dual local search
    if time.monotonic() < deadline - 10.0:
        polished = dual_local_search(best_solution, max_time=deadline - 8.0)
        if len(polished) < best_len:
            best_len = len(polished)
            best_solution = polished

    # Phase 3: Additional local search passes with different random seeds
    for seed in [123, 456, 789]:
        if time.monotonic() > deadline - 5.0:
            break
        
        rng2 = random.Random(seed)
        # Temporarily use rng2 for diversity
        orig_rng = rng
        rng = rng2
        
        polished = dual_local_search(best_solution, max_time=deadline - 5.0)
        if len(polished) < best_len:
            best_len = len(polished)
            best_solution = polished
        
        rng = orig_rng

    # Final reduction
    best_solution = remove_redundant(best_solution)
    
    # Convert to output
    return get_blocks(best_solution)
