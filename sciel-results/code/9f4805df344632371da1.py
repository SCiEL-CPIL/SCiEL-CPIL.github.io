"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _greedy_construct(v, k, t, deadline):
    """
    Fast greedy covering with precomputed lookup for gain computation.
    """
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    covered = np.zeros(n_subs, dtype=bool)
    blocks = []

    # Precompute: for each (t-1)-subset (as sorted tuple), map to {x: sub_idx}
    # where sub_idx is the index of the t-subset formed by adding x
    t_minus_1 = t - 1
    rest_to_x_to_idx = {}
    for sub_idx, sub in enumerate(all_subs):
        for j in range(t):
            rest = sub[:j] + sub[j+1:]
            x = sub[j]
            key = tuple(sorted(rest))
            if key not in rest_to_x_to_idx:
                rest_to_x_to_idx[key] = {}
            rest_to_x_to_idx[key][x] = sub_idx
    
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
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Count uncovered t-subsets S where x ∈ S and S\{x} ⊆ block
                gain = 0
                for rest in itertools.combinations(block, t_minus_1):
                    key = tuple(sorted(rest))
                    if key in rest_to_x_to_idx:
                        idx = rest_to_x_to_idx[key].get(x, -1)
                        if idx >= 0 and not covered[idx]:
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
    """Remove blocks that don't uniquely cover any t-subset."""
    if len(blocks) <= 1:
        return blocks
    
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    
    new_blocks = []
    for i, block in enumerate(blocks):
        has_unique = False
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            if coverage[idx] == 1:
                has_unique = True
                break
        
        if not has_unique:
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                coverage[idx] -= 1
        else:
            new_blocks.append(block)
        
        if time.time() > deadline:
            return new_blocks + blocks[i + 1:]
    
    return new_blocks


def _local_search(blocks, v, k, t, deadline, seed=42):
    """
    Local search to reduce block count with multiple move types.
    """
    rng = np.random.RandomState(seed)
    current_blocks = list(blocks)
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    coverage = np.zeros(n_subs, dtype=np.int32)
    for block in current_blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            coverage[idx] += 1
    
    def get_block_subs(block):
        return list(itertools.combinations(block, t))
    
    def try_remove_block(idx):
        """Try to remove block at index idx. Returns True if removed."""
        block = current_blocks[idx]
        block_subs = get_block_subs(block)
        for sub in block_subs:
            sidx = sub_to_idx[sub]
            if coverage[sidx] <= 1:
                return False
        for sub in block_subs:
            sidx = sub_to_idx[sub]
            coverage[sidx] -= 1
        current_blocks.pop(idx)
        return True
    
    def try_replace_block(idx, new_block):
        """Replace block at idx with new_block if coverage maintained."""
        old_block = current_blocks[idx]
        old_subs = get_block_subs(old_block)
        new_subs = get_block_subs(new_block)
        new_sub_set = set(new_subs)
        
        for sub in old_subs:
            sidx = sub_to_idx[sub]
            if coverage[sidx] <= 1 and sub not in new_sub_set:
                return False
        
        for sub in old_subs:
            sidx = sub_to_idx[sub]
            coverage[sidx] -= 1
        for sub in new_subs:
            sidx = sub_to_idx[sub]
            coverage[sidx] += 1
        current_blocks[idx] = new_block
        return True
    
    temp = 1.0
    cooling = 0.9995
    min_temp = 0.001
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and temp > min_temp:
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            # Reset to best and restart with higher temp
            current_blocks = list(best_blocks)
            coverage = np.zeros(n_subs, dtype=np.int32)
            for block in current_blocks:
                for sub in itertools.combinations(block, t):
                    idx = sub_to_idx[sub]
                    coverage[idx] += 1
            no_improve_count = 0
            temp = max(temp, 0.5)
        
        move_type = rng.randint(0, 3)
        
        if move_type == 0 and len(current_blocks) > 1:
            # Try to remove a block
            idx = rng.randint(0, len(current_blocks))
            if try_remove_block(idx):
                new_size = len(current_blocks)
                if new_size < best_size:
                    best_blocks = list(current_blocks)
                    best_size = new_size
                    no_improve_count = 0
                temp *= cooling
                continue
        
        elif move_type == 1 and len(current_blocks) > 1:
            # Replace a block with a greedy-optimized one
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            old_subs = get_block_subs(old_block)
            
            # Check if we can temporarily remove
            can_remove = True
            for sub in old_subs:
                sidx = sub_to_idx[sub]
                if coverage[sidx] <= 1:
                    can_remove = False
                    break
            
            if not can_remove:
                temp *= cooling
                continue
            
            # Temporarily remove
            for sub in old_subs:
                sidx = sub_to_idx[sub]
                coverage[sidx] -= 1
            
            # Build a good replacement block greedily
            # Start with an uncovered subset if possible, else random
            uncovered = np.where(coverage == 0)[0]
            if len(uncovered) > 0:
                start_sub = all_subs[rng.randint(0, len(uncovered))]
            else:
                start_sub = tuple(sorted(rng.choice(v, size=t, replace=False).tolist()))
            
            new_block = list(start_sub)
            block_set = set(new_block)
            
            # Greedily add points to maximize coverage of currently-uncovered subsets
            t_minus_1 = t - 1
            while len(new_block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(new_block, t_minus_1):
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
                new_block.append(best_point)
                block_set.add(best_point)
            
            new_block.sort()
            
            # Add back the new block's coverage
            for sub in itertools.combinations(new_block, t):
                sidx = sub_to_idx[sub]
                coverage[sidx] += 1
            current_blocks[idx] = new_block
            
            new_size = len(current_blocks)
            if new_size < best_size:
                best_blocks = list(current_blocks)
                best_size = new_size
                no_improve_count = 0
            
            temp *= cooling
            continue
        
        elif move_type == 2:
            # Perturb: modify a random block by changing 1-2 elements
            if len(current_blocks) == 0:
                temp *= cooling
                continue
            
            idx = rng.randint(0, len(current_blocks))
            old_block = current_blocks[idx]
            new_block = list(old_block)
            num_changes = rng.randint(1, min(2, k))
            for _ in range(num_changes):
                pos = rng.randint(0, k)
                new_val = rng.randint(0, v)
                if new_val not in new_block:
                    new_block[pos] = new_val
            new_block.sort()
            
            if new_block != old_block:
                if try_replace_block(idx, new_block):
                    new_size = len(current_blocks)
                    if new_size < best_size:
                        best_blocks = list(current_blocks)
                        best_size = new_size
                        no_improve_count = 0
            
            temp *= cooling
            continue
        
        else:
            # Add a new block to fill gaps, then try to remove a redundant one
            if len(current_blocks) < 300:  # Safety cap
                uncovered = np.where(coverage == 0)[0]
                if len(uncovered) > 0:
                    start_sub = all_subs[rng.randint(0, len(uncovered))]
                    new_block = list(start_sub)
                    block_set = set(new_block)
                    t_minus_1 = t - 1
                    while len(new_block) < k:
                        best_point = -1
                        best_gain = -1
                        for x in range(v):
                            if x in block_set:
                                continue
                            gain = 0
                            for rest in itertools.combinations(new_block, t_minus_1):
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
                        new_block.append(best_point)
                        block_set.add(best_point)
                    new_block.sort()
                    
                    # Add the new block
                    for sub in itertools.combinations(new_block, t):
                        sidx = sub_to_idx[sub]
                        coverage[sidx] += 1
                    current_blocks.append(new_block)
                    
                    # Now try to remove a redundant block
                    for _ in range(3):
                        idx = rng.randint(0, len(current_blocks))
                        if try_remove_block(idx):
                            new_size = len(current_blocks)
                            if new_size < best_size:
                                best_blocks = list(current_blocks)
                                best_size = new_size
                                no_improve_count = 1
                            break
            
            temp *= cooling
            continue
    
    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for final validation
    
    # Step 1: Greedy construction
    blocks = _greedy_construct(v, k, t, deadline)
    
    # Step 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, t, deadline)
    
    # Step 3: Local search
    if time.time() < deadline - 10:
        blocks = _local_search(blocks, v, k, t, deadline, seed=42)
    
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
