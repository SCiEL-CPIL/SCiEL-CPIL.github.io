"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def _greedy_construct(v, k, t, deadline):
    """
    Fast greedy covering algorithm using vectorized gain computation.
    """
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    covered = np.zeros(n_subs, dtype=bool)
    blocks = []

    # Precompute: for each (t-1)-subset, the index of the t-subset formed by adding each point
    # Key: (tuple of t-1 points) -> dict mapping point -> subset index (if valid)
    from collections import defaultdict
    tminus1_to_additions = defaultdict(dict)
    for sub_idx, sub in enumerate(all_subs):
        for i in range(t):
            rest = tuple(sorted(sub[:i] + sub[i+1:]))
            x = sub[i]
            tminus1_to_additions[rest][x] = sub_idx
    
    while not covered.all():
        if time.time() > deadline:
            break
        
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[0]
        start_sub = all_subs[start_idx]
        
        block = list(start_sub)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_point = -1
            best_gain = -1
            
            # For each candidate point x, compute gain = number of uncovered t-subsets
            # that contain x and whose other t-1 elements are in block
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    rest_key = tuple(sorted(rest))
                    if x in tminus1_to_additions.get(rest_key, {}):
                        idx = tminus1_to_additions[rest_key][x]
                        if not covered[idx]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                break
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            covered[idx] = True
    
    return blocks


def _remove_redundant_blocks(blocks, v, t, deadline):
    """Remove blocks that don't uniquely cover any t-subset. Iterates until no more can be removed."""
    if len(blocks) <= 1:
        return blocks
    
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    # Precompute block subset indices
    block_sub_indices = []
    for block in blocks:
        idxs = []
        for sub in itertools.combinations(block, t):
            idxs.append(sub_to_idx[sub])
        block_sub_indices.append(np.array(idxs, dtype=np.int32))
    
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        
        coverage = np.zeros(n_subs, dtype=np.int32)
        for idxs in block_sub_indices:
            np.add.at(coverage, idxs, 1)
        
        new_blocks = []
        new_block_sub_indices = []
        for i, idxs in enumerate(block_sub_indices):
            has_unique = np.any(coverage[idxs] == 1)
            if not has_unique:
                np.add.at(coverage, idxs, -1)
                changed = True
            else:
                new_blocks.append(blocks[i])
                new_block_sub_indices.append(idxs)
        
        if new_blocks != blocks:
            blocks = new_blocks
            block_sub_indices = new_block_sub_indices
        else:
            changed = False
    
    return blocks


def _local_search(blocks, v, k, t, deadline, seed=42):
    """
    Local search to reduce block count.
    Moves: remove a redundant block, or swap a block for a better one.
    """
    rng = np.random.RandomState(seed)
    current_blocks = list(blocks)
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    # Precompute block subset indices
    def block_to_indices(block):
        return np.array([sub_to_idx[sub] for sub in itertools.combinations(block, t)], dtype=np.int32)
    
    block_indices = [block_to_indices(b) for b in current_blocks]
    
    coverage = np.zeros(n_subs, dtype=np.int32)
    for idxs in block_indices:
        np.add.at(coverage, idxs, 1)
    
    # Precompute for greedy candidate generation: for each (t-1)-subset, map point -> t-subset index
    tminus1_map = defaultdict(dict)
    for sub_idx, sub in enumerate(all_subs):
        for i in range(t):
            rest = tuple(sorted(sub[:i] + sub[i+1:]))
            x = sub[i]
            tminus1_map[rest][x] = sub_idx
    
    def greedy_candidate(coverage_arr, num_candidates=5):
        """Generate a few greedy candidates to replace a block."""
        candidates = []
        for _ in range(num_candidates):
            # Start from a random uncovered or under-covered subset
            if np.any(coverage_arr == 0):
                uncov = np.where(coverage_arr == 0)[0]
                start_idx = uncov[rng.randint(0, len(uncov))]
            else:
                start_idx = rng.randint(0, n_subs)
            start_sub = all_subs[start_idx]
            block = list(start_sub)
            block_set = set(block)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        rest_key = tuple(sorted(rest))
                        if x in tminus1_map.get(rest_key, {}):
                            idx = tminus1_map[rest_key][x]
                            if coverage_arr[idx] == 0:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    break
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            candidates.append(block)
        return candidates
    
    temp = 1.0
    cooling = 0.999
    min_temp = 0.001
    iteration = 0
    
    while time.time() < deadline and temp > min_temp:
        iteration += 1
        move_type = rng.randint(0, 3)
        
        if move_type <= 1 and len(current_blocks) > 1:
            # Try to remove a block
            idx = rng.randint(0, len(current_blocks))
            idxs = block_indices[idx]
            
            can_remove = np.all(coverage[idxs] > 1)
            
            if can_remove:
                np.add.at(coverage, idxs, -1)
                current_blocks.pop(idx)
                block_indices.pop(idx)
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
                
                temp *= cooling
                continue
            else:
                temp *= cooling
                continue
        
        elif move_type == 2 and len(current_blocks) > 1:
            # Swap: replace a block with a better one
            idx = rng.randint(0, len(current_blocks))
            old_idxs = block_indices[idx]
            
            can_remove = np.all(coverage[old_idxs] > 1)
            
            if not can_remove:
                temp *= cooling
                continue
            
            # Temporarily remove
            np.add.at(coverage, old_idxs, -1)
            
            # Generate greedy candidates
            candidates = greedy_candidate(coverage, num_candidates=5)
            
            best_block = None
            best_gain = -1
            for cand in candidates:
                cand_idxs = block_to_indices(cand)
                gain = np.sum(coverage[cand_idxs] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block = cand
            
            if best_block is not None and best_gain > 0:
                cand_idxs = block_to_indices(best_block)
                np.add.at(coverage, cand_idxs, 1)
                current_blocks[idx] = best_block
                block_indices[idx] = cand_idxs
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
            else:
                # Revert
                np.add.at(coverage, old_idxs, 1)
            
            temp *= cooling
            continue
        
        else:
            # Perturb: modify a random block
            if len(current_blocks) == 0:
                temp *= cooling
                continue
            
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_idxs = block_indices[idx]
            
            new_block = list(old_block)
            num_changes = rng.randint(1, min(2, k))
            for _ in range(num_changes):
                pos = rng.randint(0, k)
                new_val = rng.randint(0, v)
                if new_val not in new_block:
                    new_block[pos] = new_val
            
            new_block.sort()
            
            if new_block == old_block:
                temp *= cooling
                continue
            
            new_idxs = block_to_indices(new_block)
            new_sub_set = set(new_idxs)
            
            # Check validity: no old-only subset loses its last cover
            valid = True
            for sidx in old_idxs:
                if coverage[sidx] <= 1 and sidx not in new_sub_set:
                    valid = False
                    break
            
            if valid:
                np.add.at(coverage, old_idxs, -1)
                np.add.at(coverage, new_idxs, 1)
                current_blocks[idx] = new_block
                block_indices[idx] = new_idxs
                
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
            
            temp *= cooling
    
    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for final validation
    
    # Step 1: Greedy construction
    blocks = _greedy_construct(v, k, t, deadline)
    
    # Step 2: Remove redundant blocks (iterative)
    blocks = _remove_redundant_blocks(blocks, v, t, deadline)
    
    # Step 3: Local search with multiple seeds if time permits
    if time.time() < deadline - 20:
        blocks = _local_search(blocks, v, k, t, deadline, seed=42)
    
    # Step 4: Second local search pass with different seed if time permits
    if time.time() < deadline - 10:
        blocks = _local_search(blocks, v, k, t, deadline, seed=123)
    
    # Final cleanup: remove any redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, t, deadline)
    
    # Ensure validity: check all t-subsets are covered
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    
    # If not fully covered, add more blocks greedily
    while not (coverage > 0).all() and time.time() < deadline:
        uncovered_idx = np.where(coverage == 0)[0][0]
        sub = all_subs[uncovered_idx]
        block = list(sub)
        block_set = set(block)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in sub_to_idx:
                        sidx = sub_to_idx[candidate]
                        if coverage[sidx] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                break
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        for sub2 in itertools.combinations(block, t):
            sidx = sub_to_idx[sub2]
            coverage[sidx] += 1
    
    return blocks
# EVOLVE-BLOCK-END
