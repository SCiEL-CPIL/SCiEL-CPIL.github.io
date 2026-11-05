import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145 # 5s safety margin
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip
        
    # Precompute all blocks and their masks
    all_blocks = [tuple(b) for b in itertools.combinations(range(v), k)]
    n_blocks = len(all_blocks)
    
    # Helper to get mask of a block
    def get_block_mask(block):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        return mask

    # Precompute masks for all blocks
    block_masks = [get_block_mask(b) for b in all_blocks]
    
    # Precompute triplet ids for each block for faster counting in optimization
    block_triplet_ids = []
    for b in all_blocks:
        ids = []
        for triplet in itertools.combinations(b, t):
            ids.append(triplet_to_id[triplet])
        block_triplet_ids.append(ids)
        
    # Helper to enumerate blocks containing a set of points (for local search)
    def blocks_containing(points):
        points = set(points)
        if len(points) > k:
            return []
        remaining = [p for p in range(v) if p not in points]
        need = k - len(points)
        if need < 0:
            return []
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points.union(comb)))

    # Local Greedy Construction (Faster, restricted)
    def greedy_local(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triplet
            # Find a set bit
            lsb = uncovered & -uncovered
            idx = lsb.bit_length() - 1
            start_triplet = id_to_triplet[idx]
            
            # Find best block containing start_triplet
            best_block = None
            best_gain = -1
            best_mask = 0
            
            for block in blocks_containing(start_triplet):
                b_mask = get_block_mask(block)
                gain = (b_mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_mask = b_mask
                    
            if best_block:
                blocks.append(best_block)
                uncovered &= ~best_mask
            else:
                # Fallback: shouldn't happen
                break
                
        return blocks

    # Local Search Optimizer
    def optimize(blocks):
        # Use lists of triplet ids for faster count updates
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def update_counts_add(block_idx):
            for tid in block_triplet_ids[block_idx]:
                counts[tid] += 1
                
        def update_counts_sub(block_idx):
            for tid in block_triplet_ids[block_idx]:
                counts[tid] -= 1
                
        # Initialize counts
        # Note: blocks are tuples, need to map to index or just use the tuple directly?
        # It is faster to keep track of indices into all_blocks if possible, 
        # but blocks can be new. 
        # Let's stick to computing ids on the fly for new blocks, but use precomputed for existing?
        # Actually, just computing ids on the fly is fast enough for local search steps.
        
        def get_ids(block):
            return [triplet_to_id[triplet] for triplet in itertools.combinations(block, t)]

        # Recalculate counts from scratch for initial blocks
        for b in blocks:
            for tid in get_ids(b):
                counts[tid] += 1
        
        max_iters = 10000
        no_improve_count = 0
        current_blocks = list(blocks)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            n = len(current_blocks)
            
            # 1. Try Remove Redundant
            for i in range(n):
                if not improved:
                    b = current_blocks[i]
                    redundant = True
                    for tid in get_ids(b):
                        if counts[tid] == 1:
                            redundant = False
                            break
                    if redundant:
                        current_blocks.pop(i)
                        for tid in get_ids(b):
                            counts[tid] -= 1
                        improved = True
                        break
                        
            # 2. Try 2-for-1
            if not improved:
                # Sample random pairs to save time
                if n > 2:
                    for _ in range(50): # Try 50 random pairs
                        if time.time() > deadline - 10:
                            break
                        i, j = random.sample(range(n), 2)
                        b1 = current_blocks[i]
                        b2 = current_blocks[j]
                        
                        ids_b1 = set(get_ids(b1))
                        ids_b2 = set(get_ids(b2))
                        
                        needed_triplets = []
                        all_involved = ids_b1 | ids_b2
                        for tid in all_involved:
                            if counts[tid] == 1:
                                needed_triplets.append(tid)
                                
                        if not needed_triplets:
                            # Both redundant?
                            # Pop j then i
                            b_rem = current_blocks.pop(j)
                            for tid in get_ids(b_rem): counts[tid] -= 1
                            b_rem = current_blocks.pop(i)
                            for tid in get_ids(b_rem): counts[tid] -= 1
                            improved = True
                            break
                        
                        points_needed = set()
                        for tid in needed_triplets:
                            trip = id_to_triplet[tid]
                            points_needed.update(trip)
                            
                        if len(points_needed) > k:
                            continue
                            
                        found_replacement = False
                        for new_block in blocks_containing(points_needed):
                            if new_block == b1 or new_block == b2:
                                continue
                            
                            new_ids = set(get_ids(new_block))
                            covers = all(tid in new_ids for tid in needed_triplets)
                            
                            if covers:
                                # Swap
                                b_rem = current_blocks.pop(j)
                                for tid in get_ids(b_rem): counts[tid] -= 1
                                b_rem = current_blocks.pop(i)
                                for tid in get_ids(b_rem): counts[tid] -= 1
                                
                                current_blocks.append(new_block)
                                for tid in get_ids(new_block): counts[tid] += 1
                                
                                improved = True
                                found_replacement = True
                                break
                                
                        if improved:
                            break

            # 3. Try 1-for-1
            if not improved:
                if n > 0:
                    i = random.randint(0, n-1)
                    b1 = current_blocks[i]
                    ids_b1 = set(get_ids(b1))
                    
                    needed_triplets = [tid for tid in ids_b1 if counts[tid] == 1]
                    
                    if needed_triplets:
                        points_needed = set()
                        for tid in needed_triplets:
                            trip = id_to_triplet[tid]
                            points_needed.update(trip)
                            
                        if len(points_needed) <= k:
                            for new_block in blocks_containing(points_needed):
                                if new_block == b1:
                                    continue
                                new_ids = set(get_ids(new_block))
                                if all(tid in new_ids for tid in needed_triplets):
                                    # Swap
                                    b_rem = current_blocks.pop(i)
                                    for tid in get_ids(b_rem): counts[tid] -= 1
                                    current_blocks.append(new_block)
                                    for tid in get_ids(new_block): counts[tid] += 1
                                    improved = True
                                    break
                                
            if not improved:
                # Restart optimization with a different random seed or break?
                # Let's just break for this instance, we will have multiple candidates
                break
                
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    candidates = []
    
    # Run Local Greedy many times as it is fast
    # Estimate: ~6k blocks per step * ~80 steps = 480k ops. 
    # 50 runs * 480k = 24M ops. Should be fast enough in Python if optimized.
    # Let's try 50 runs.
    for seed in range(50):
        if time.time() > deadline - 30:
            break
        blocks = greedy_local(seed)
        candidates.append(blocks)
        
    # Optimize all candidates
    for blocks in candidates:
        if time.time() > deadline - 5:
            break
            
        opt_blocks = optimize(blocks)
        
        # Verify
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1
                
        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks
                
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_local(0)
        
    return best_blocks

if __name__ == "__main__":
    pass
