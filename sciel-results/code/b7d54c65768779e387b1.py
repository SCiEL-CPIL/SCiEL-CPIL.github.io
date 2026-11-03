import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    # Deadline guard: 150s limit, use 145s to be safe
    deadline = time.time() + 145
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute bitmask for each block? No, blocks are dynamic.
    # But we can precompute the t-subsets for any potential block? 
    # There are C(19,8) = 75582 possible blocks.
    # We can precompute the bitmask of t-subsets covered by each possible block.
    # This will speed up greedy selection and local search significantly.
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage mask for each possible block
    # mask[b] is an integer where bit i is 1 if tsub i is in block b
    block_masks = [0] * len(all_blocks)
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # Also precompute the list of t-subset indices for each block for quick access if needed
    # But bitmasks are faster for set operations.
    
    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)
    
    def _remove_redundant(blocks, deadline):
        if not blocks:
            return blocks
            
        # Compute coverage counts using bitmasks is tricky for counts, 
        # so we stick to counting or using a frequency array.
        # n_tsubs is ~11628, which is small enough for an array.
        cover_count = [0] * n_tsubs
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        
        for bi in block_indices:
            mask = block_masks[bi]
            # Iterate set bits
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                bi = block_indices[i]
                mask = block_masks[bi]
                m = mask
                redundant = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    # Remove block i
                    mask = block_masks[bi]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                        
                    blocks.pop(i)
                    block_indices.pop(i)
                    changed = True
                    break # Restart loop
                    
        return blocks

    def _local_search(blocks, deadline):
        if not blocks:
            return blocks
            
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs
        
        for bi in block_indices:
            mask = block_masks[bi]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        improved = True
        max_iterations = 50
        iter_count = 0
        
        while improved and time.time() < deadline - 10 and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            # Try to remove a block
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                
                bi = block_indices[i]
                mask = block_masks[bi]
                
                # Find critical t-subsets (covered only by block i)
                criticals = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        criticals.append(idx)
                    m ^= lsb
                
                if not criticals:
                    # Block i is redundant
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                        
                    blocks.pop(i)
                    block_indices.pop(i)
                    improved = True
                    break
                
                # If not redundant, try to swap a point in another block to cover ALL criticals?
                # Actually, we just need to cover them.
                # Strategy: Pick a critical t-subset. Find a block j that can be modified to cover it.
                # If modifying j also covers other criticals, that's a bonus.
                
                # To speed up, we try to fix the most "constrained" block first?
                # Or just iterate.
                
                found_swap = False
                # Sort criticals by how hard they are to cover?
                # For now, pick the first one.
                crit_idx = criticals[0]
                crit_tsub = all_tsubs[crit_idx]
                crit_set = set(crit_tsub)
                
                # Find candidate blocks j that are "close" to covering crit_tsub
                # A block j covers crit_tsub if it contains crit_tsub.
                # We want to modify j to contain crit_tsub.
                # If j contains t-1 elements of crit_tsub, we can swap 1 element.
                # If j contains t-2 elements, we can swap 2 elements.
                
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if time.time() > deadline - 10:
                        break
                    if found_swap:
                        break
                    
                    bj = blocks[j]
                    bj_set = set(bj)
                    common = bj_set.intersection(crit_set)
                    
                    if len(common) >= t - 1:
                        missing = [p for p in crit_tsub if p not in bj_set]
                        if len(missing) == 1:
                            extra = [p for p in bj if p not in common]
                            for rem_p in extra:
                                if found_swap:
                                    break
                                new_bj = [p if p != rem_p else missing[0] for p in bj]
                                new_bj.sort()
                                if new_bj == bj:
                                    continue
                                new_bi = block_to_idx.get(tuple(new_bj))
                                if new_bi is None:
                                    continue
                                
                                # Check if this swap is valid (doesn't break unique coverage of block j)
                                old_mask = block_masks[block_indices[j]]
                                new_mask = block_masks[new_bi]
                                
                                # T-subsets lost by j
                                lost = old_mask & ~new_mask
                                # T-subsets gained by j
                                gained = new_mask & ~old_mask
                                
                                valid = True
                                m = lost
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    if cover_count[idx] == 1:
                                        valid = False
                                        break
                                    m ^= lsb
                                
                                if valid:
                                    # Apply swap
                                    # Update cover counts
                                    m = lost
                                    while m:
                                        lsb = m & -m
                                        idx = lsb.bit_length() - 1
                                        cover_count[idx] -= 1
                                        m ^= lsb
                                        
                                    m = gained
                                    while m:
                                        lsb = m & -m
                                        idx = lsb.bit_length() - 1
                                        cover_count[idx] += 1
                                        m ^= lsb
                                        
                                    blocks[j] = new_bj
                                    block_indices[j] = new_bi
                                    improved = True
                                    found_swap = True
                                    break
                            
                            # Try 2-point swaps
                            if not found_swap and len(common) >= t - 2:
                                missing = [p for p in crit_tsub if p not in bj_set]
                                if len(missing) == 2:
                                    extra = [p for p in bj if p not in common]
                                    if len(extra) >= 2:
                                        for rem1, rem2 in itertools.combinations(extra, 2):
                                            if found_swap:
                                                break
                                            new_bj = [p for p in bj if p != rem1 and p != rem2] + missing
                                            new_bj.sort()
                                            if len(set(new_bj)) != k:
                                                continue
                                            new_bi = block_to_idx.get(tuple(new_bj))
                                            if new_bi is None:
                                                continue
                                            
                                            old_mask = block_masks[block_indices[j]]
                                            new_mask = block_masks[new_bi]
                                            
                                            lost = old_mask & ~new_mask
                                            gained = new_mask & ~old_mask
                                            
                                            valid = True
                                            m = lost
                                            while m:
                                                lsb = m & -m
                                                idx = lsb.bit_length() - 1
                                                if cover_count[idx] == 1:
                                                    valid = False
                                                    break
                                                m ^= lsb
                                            
                                            if valid:
                                                m = lost
                                                while m:
                                                    lsb = m & -m
                                                    idx = lsb.bit_length() - 1
                                                    cover_count[idx] -= 1
                                                    m ^= lsb
                                                    
                                                m = gained
                                                while m:
                                                    lsb = m & -m
                                                    idx = lsb.bit_length() - 1
                                                    cover_count[idx] += 1
                                                    m ^= lsb
                                                    
                                                blocks[j] = new_bj
                                                block_indices[j] = new_bi
                                                improved = True
                                                found_swap = True
                                                break
                if found_swap:
                    break

        return blocks

    # Main Loop: Randomized Greedy with Bitmasks
    for restart in range(100):
        if time.time() > deadline - 20:
            break

        uncovered_mask = (1 << n_tsubs) - 1
        blocks = []
        
        # Randomize the order of t-subsets to break symmetry
        order = list(range(n_tsubs))
        rng.shuffle(order)
        
        while uncovered_mask:
            if time.time() > deadline - 10:
                break
            
            # Find the first uncovered t-subset in the randomized order
            seed_idx = -1
            for idx in order:
                if (uncovered_mask >> idx) & 1:
                    seed_idx = idx
                    break
                    
            if seed_idx == -1:
                break
                
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_gain = -1
                best_point = -1
                tie_points = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain: how many new t-subsets does adding x cover?
                    # A t-subset is covered if it contains x and t-1 points from current block.
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None:
                            if (uncovered_mask >> idx) & 1:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        tie_points = [x]
                    elif gain == best_gain and gain > -1:
                        tie_points.append(x)
                        
                if best_point >= 0:
                    if len(tie_points) > 1:
                        best_point = rng.choice(tie_points)
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining)
                        block.append(best_point)
                        block_set.add(best_point)
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            bi = block_to_idx[tuple(block)]
            block_mask = block_masks[bi]
            uncovered_mask &= ~block_mask
            
        if time.time() > deadline - 10 or uncovered_mask:
            continue
            
        # Post-processing
        blocks = _remove_redundant(blocks, deadline)
        
        if time.time() < deadline - 15:
            blocks = _local_search(blocks, deadline)
            blocks = _remove_redundant(blocks, deadline)
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if best_count <= 335:
                break

    if best_blocks is None:
        # Fallback: Simple greedy
        uncovered_mask = (1 << n_tsubs) - 1
        blocks = []
        while uncovered_mask:
            seed_idx = 0
            while (uncovered_mask >> seed_idx) & 1 == 0:
                seed_idx += 1
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None:
                            if (uncovered_mask >> idx) & 1:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point >= 0:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(remaining[0])
                        block_set.add(block[-1])
                    else:
                        break
            block.sort()
            blocks.append(block)
            bi = block_to_idx[tuple(block)]
            uncovered_mask &= ~block_masks[bi]
        best_blocks = blocks

    # Final cleanup
    best_blocks = _remove_redundant(best_blocks, deadline)
    
    return best_blocks
