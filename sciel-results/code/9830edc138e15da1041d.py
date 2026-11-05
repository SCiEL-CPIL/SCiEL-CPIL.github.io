import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precomputation
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    T = len(all_triples)
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    
    # For fast mask generation for a block
    # We need to map a block (tuple of k points) to a bit mask
    # Since k=7, v=24, there are C(24,7) = 346104 possible blocks.
    # We can precompute masks for all possible blocks?
    # 346k masks of 2024 bits. 2024 bits ~ 253 bytes. 346k * 253 ~ 87 MB.
    # This might be too much memory or slow to precompute.
    # Better to compute on the fly or cache.
    
    def get_block_mask(block):
        mask = 0
        for tr in itertools.combinations(block, t):
            mask |= (1 << triple_to_idx[tr])
        return mask

    # Greedy Construction
    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered_mask = (1 << T) - 1
        blocks = []
        
        while uncovered_mask:
            # Find an uncovered triple
            # Get lowest set bit
            lsb = uncovered_mask & -uncovered_mask
            idx = lsb.bit_length() - 1
            seed_triple = all_triples[idx]
            
            block = list(seed_triple)
            
            # Greedily add points
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in points:
                    if x in block:
                        continue
                    gain = 0
                    # Check pairs in block
                    for u, v in itertools.combinations(block, t - 1):
                        tr = tuple(sorted((x, u, v)))
                        # Check if this triple is uncovered
                        # triple_to_idx[tr] is the index
                        if (uncovered_mask >> triple_to_idx[tr]) & 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        if gain == 0: # No point covers new triples? Should not happen if uncovered_mask != 0
                            # But if we are stuck, pick arbitrary
                            pass
                
                if best_x == -1:
                    # Fallback
                    for x in points:
                        if x not in block:
                            best_x = x
                            break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            b_mask = get_block_mask(block)
            uncovered_mask &= ~b_mask
            
        return blocks

    # Local Search
    def local_search(blocks):
        # Calculate counts
        counts = [0] * T
        block_masks = []
        for b in blocks:
            m = get_block_mask(b)
            block_masks.append(m)
            for tr in itertools.combinations(b, t):
                counts[triple_to_idx[tr]] += 1
        
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant blocks
            i = 0
            while i < len(blocks):
                b = blocks[i]
                m = block_masks[i]
                redundant = True
                # Check if all triples in b are covered by others
                # We can iterate through the triples in b
                for tr in itertools.combinations(b, t):
                    if counts[triple_to_idx[tr]] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for tr in itertools.combinations(b, t):
                        counts[triple_to_idx[tr]] -= 1
                    blocks.pop(i)
                    block_masks.pop(i)
                    improved = True
                    # Do not increment i, check next block at same index
                else:
                    i += 1
            
            if improved:
                continue # Restart loop as counts changed significantly
            
            # 2. Try swaps
            swapped = False
            for i in range(len(blocks)):
                b = blocks[i]
                # Try swapping each point in b
                for x_idx, x in enumerate(b):
                    # Try replacing x with y
                    for y in points:
                        if y in b:
                            continue
                        
                        # Check validity of swap
                        # Lost triples: {x, u, v} where u, v in b \ {x}
                        # Gained triples: {y, u, v} where u, v in b \ {x}
                        
                        other_points = [p for j, p in enumerate(b) if j != x_idx]
                        
                        valid = True
                        lost_triples = []
                        gained_triples = []
                        
                        for u, v in itertools.combinations(other_points, t - 1):
                            tr_lost = tuple(sorted((x, u, v)))
                            tr_gained = tuple(sorted((y, u, v)))
                            
                            lost_triples.append(tr_lost)
                            gained_triples.append(tr_gained)
                            
                            # If a lost triple is only covered by this block, swap is invalid
                            if counts[triple_to_idx[tr_lost]] == 1:
                                valid = False
                                break
                        
                        if valid:
                            # Apply swap
                            # Update counts
                            for tr in lost_triples:
                                counts[triple_to_idx[tr]] -= 1
                            for tr in gained_triples:
                                counts[triple_to_idx[tr]] += 1
                            
                            # Update block
                            b[x_idx] = y
                            b.sort()
                            blocks[i] = b
                            
                            # Update mask
                            block_masks[i] = get_block_mask(b)
                            
                            swapped = True
                            break # Restart swap search? Or just continue?
                            # Restarting is safer to handle cascading effects
                        if swapped:
                            break
                    if swapped:
                        break
                if swapped:
                    break
            
            if swapped:
                improved = True
        
        return blocks

    # Initial Solution
    current_blocks = greedy_cover(42)
    current_blocks = local_search(current_blocks)
    best_blocks = current_blocks[:]
    best_len = len(current_blocks)
    
    # Iterated Local Search
    while time.time() < deadline:
        # Perturb
        perturbed = best_blocks[:]
        
        # Remove a random block
        if perturbed:
            idx_to_remove = random.randint(0, len(perturbed) - 1)
            removed_block = perturbed.pop(idx_to_remove)
            
            # Re-greedy fill the gaps
            # We need to know what's uncovered now.
            # This is expensive to recompute from scratch.
            # Alternative: Just run a full greedy again with a random seed?
            # That might lose the good structure.
            
            # Let's try a "Repair" step:
            # Compute uncovered mask for perturbed
            covered_mask = 0
            for b in perturbed:
                covered_mask |= get_block_mask(b)
            uncovered_mask = ((1 << T) - 1) & ~covered_mask
            
            # Greedy add blocks to cover uncovered
            while uncovered_mask:
                lsb = uncovered_mask & -uncovered_mask
                idx = lsb.bit_length() - 1
                seed_triple = all_triples[idx]
                
                block = list(seed_triple)
                while len(block) < k:
                    best_x = -1
                    best_gain = -1
                    for x in points:
                        if x in block:
                            continue
                        gain = 0
                        for u, v in itertools.combinations(block, t - 1):
                            tr = tuple(sorted((x, u, v)))
                            if (uncovered_mask >> triple_to_idx[tr]) & 1:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x == -1:
                        for x in points:
                            if x not in block:
                                best_x = x
                                break
                    block.append(best_x)
                
                block.sort()
                perturbed.append(block)
                b_mask = get_block_mask(block)
                uncovered_mask &= ~b_mask
        
        # Local Search on perturbed
        perturbed = local_search(perturbed)
        
        if len(perturbed) < best_len:
            best_len = len(perturbed)
            best_blocks = perturbed[:]
            
    return best_blocks
