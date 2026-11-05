import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 143  # 7s safety margin
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = all_triplets  # list indexed by triplet id
    
    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the list of triplet IDs it covers
    # This avoids recomputing combinations repeatedly
    block_triplet_ids = []
    for block in all_blocks:
        tids = [triplet_to_id[trip] for trip in itertools.combinations(block, t)]
        block_triplet_ids.append(tids)
    
    # Build a lookup: for each triplet, which blocks contain it
    # triplet_blocks[tid] = list of block indices
    triplet_blocks = [[] for _ in range(n_triplets)]
    for bi, tids in enumerate(block_triplet_ids):
        for tid in tids:
            triplet_blocks[tid].append(bi)
    
    # Helper to enumerate blocks containing a set of points
    def blocks_containing(points):
        points_set = set(points)
        if len(points_set) > k:
            return []
        remaining = [p for p in range(v) if p not in points_set]
        need = k - len(points_set)
        if need < 0:
            return []
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points_set.union(comb)))
    
    # Fast greedy using numpy counts
    def fast_greedy(seed):
        random.seed(seed)
        counts = np.zeros(n_triplets, dtype=np.int32)
        selected = []
        
        while True:
            if time.time() > deadline - 30:
                break
            # Find uncovered triplets
            uncovered_mask = counts == 0
            n_uncovered = uncovered_mask.sum()
            if n_uncovered == 0:
                break
            
            # Compute gain for each block: number of uncovered triplets it covers
            # Use the precomputed block_triplet_ids
            best_gain = -1
            best_block_idx = -1
            
            # Sample a subset of blocks if n_blocks is large to save time
            # For v=24,k=7: n_blocks = 346104, which is manageable
            for bi in range(n_blocks):
                tids = block_triplet_ids[bi]
                gain = 0
                for tid in tids:
                    if counts[tid] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
            
            if best_block_idx < 0:
                break
            
            # Add this block
            selected.append(best_block_idx)
            for tid in block_triplet_ids[best_block_idx]:
                counts[tid] += 1
        
        return [all_blocks[i] for i in selected]
    
    # Optimized local search
    def optimize(blocks):
        # Build counts
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for trip in itertools.combinations(b, t):
                counts[triplet_to_id[trip]] += 1
        
        # Precompute for each block in the current set, its triplet ids
        # and track which triplets are uniquely covered
        n = len(blocks)
        
        # Helper to check if a block is redundant
        def is_redundant(block):
            for trip in itertools.combinations(block, t):
                tid = triplet_to_id[trip]
                if counts[tid] == 1:
                    return False
            return True
        
        # Helper to get triplet ids for a block
        def get_triplet_ids(block):
            return [triplet_to_id[trip] for trip in itertools.combinations(block, t)]
        
        max_iters = 3000
        for it in range(max_iters):
            if time.time() > deadline - 8:
                break
            
            improved = False
            n = len(blocks)
            
            # 1. Remove redundant blocks
            for i in range(n):
                if is_redundant(blocks[i]):
                    b = blocks.pop(i)
                    for tid in get_triplet_ids(b):
                        counts[tid] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try 2-for-1 replacement
            # Sample pairs to limit computation
            if n >= 2:
                # Try a limited number of random pairs
                n_pairs_to_try = min(50, n * (n - 1) // 2)
                tried = set()
                for _ in range(n_pairs_to_try):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j or (i, j) in tried or (j, i) in tried:
                        continue
                    tried.add((i, j))
                    tried.add((j, i))
                    
                    b1 = blocks[i]
                    b2 = blocks[j]
                    ids1 = set(get_triplet_ids(b1))
                    ids2 = set(get_triplet_ids(b2))
                    
                    # Find triplets that would become uncovered
                    needed = []
                    for tid in ids1 | ids2:
                        if counts[tid] == 1:
                            needed.append(tid)
                    
                    if not needed:
                        # Both blocks are redundant
                        for idx in [j, i]:
                            b_rem = blocks.pop(idx)
                            for tid in get_triplet_ids(b_rem):
                                counts[tid] -= 1
                        improved = True
                        break
                    
                    # Find points needed
                    points_needed = set()
                    for tid in needed:
                        points_needed.update(id_to_triplet[tid])
                    
                    if len(points_needed) > k:
                        continue
                    
                    # Try to find a replacement block
                    for new_block in blocks_containing(points_needed):
                        if set(new_block) == set(b1) or set(new_block) == set(b2):
                            continue
                        new_ids = set(get_triplet_ids(new_block))
                        if all(tid in new_ids for tid in needed):
                            # Perform swap
                            for idx in [j, i]:
                                b_rem = blocks.pop(idx)
                                for tid in get_triplet_ids(b_rem):
                                    counts[tid] -= 1
                            blocks.append(new_block)
                            for tid in get_triplet_ids(new_block):
                                counts[tid] += 1
                            improved = True
                            break
                    
                    if improved:
                        break
            
            if not improved:
                # 3. Try 1-for-1 replacement
                if n > 0:
                    i = random.randint(0, n - 1)
                    b1 = blocks[i]
                    ids1 = set(get_triplet_ids(b1))
                    needed = [tid for tid in ids1 if counts[tid] == 1]
                    
                    if needed:
                        points_needed = set()
                        for tid in needed:
                            points_needed.update(id_to_triplet[tid])
                        
                        if len(points_needed) <= k:
                            for new_block in blocks_containing(points_needed):
                                if set(new_block) == set(b1):
                                    continue
                                new_ids = set(get_triplet_ids(new_block))
                                if all(tid in new_ids for tid in needed):
                                    # Replace
                                    for tid in get_triplet_ids(b1):
                                        counts[tid] -= 1
                                    blocks[i] = new_block
                                    for tid in get_triplet_ids(new_block):
                                        counts[tid] += 1
                                    improved = True
                                    break
            
            if not improved:
                break
        
        return blocks
    
    # Ensure validity: fill in any missing triplets
    def ensure_valid(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for trip in itertools.combinations(b, t):
                counts[triplet_to_id[trip]] += 1
        
        # Find uncovered triplets
        uncovered = [i for i in range(n_triplets) if counts[i] == 0]
        if not uncovered:
            return blocks
        
        # Greedily add blocks to cover uncovered triplets
        while uncovered:
            if time.time() > deadline - 5:
                break
            # Pick first uncovered triplet
            tid = uncovered[0]
            trip = id_to_triplet[tid]
            
            # Find best block containing this triplet
            best_block = None
            best_gain = -1
            for bi in triplet_blocks[tid]:
                tids = block_triplet_ids[bi]
                gain = sum(1 for x in tids if counts[x] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block = all_blocks[bi]
            
            if best_block is not None:
                blocks.append(best_block)
                for tid2 in block_triplet_ids[all_blocks.index(best_block)]:
                    counts[tid2] += 1
                # Update uncovered
                uncovered = [i for i in range(n_triplets) if counts[i] == 0]
            else:
                break
        
        return blocks
    
    # Verify a set of blocks is valid
    def is_valid(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for trip in itertools.combinations(b, t):
                counts[triplet_to_id[trip]] += 1
        return np.min(counts) > 0
    
    # Vectorized greedy: use numpy to compute gains for all blocks at once
    def vectorized_greedy(seed):
        random.seed(seed)
        counts = np.zeros(n_triplets, dtype=np.int32)
        selected = []
        
        # Precompute a numpy array of block_triplet_ids for vectorized operations
        # Each row is a block, each column is a triplet id (or -1 for not covered)
        max_tids = max(len(tids) for tids in block_triplet_ids)
        block_tid_matrix = np.full((n_blocks, max_tids), -1, dtype=np.int32)
        for bi, tids in enumerate(block_triplet_ids):
            block_tid_matrix[bi, :len(tids)] = tids
        
        while True:
            if time.time() > deadline - 30:
                break
            
            # Find uncovered triplets
            uncovered_mask = counts == 0
            n_uncovered = uncovered_mask.sum()
            if n_uncovered == 0:
                break
            
            # Compute gain for each block using numpy
            # gain[bi] = number of uncovered triplets covered by block bi
            # Only consider tids that are valid (not -1) and uncovered
            valid_mask = block_tid_matrix != -1
            # For each block, count how many of its triplets are uncovered
            # We need to check if block_tid_matrix[bi, j] is valid and counts[block_tid_matrix[bi, j]] == 0
            # This is tricky with numpy because we need to index counts with block_tid_matrix
            # Let's do it in a loop but with early termination for speed
            
            best_gain = -1
            best_block_idx = -1
            
            for bi in range(n_blocks):
                tids = block_triplet_ids[bi]
                gain = 0
                for tid in tids:
                    if counts[tid] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
            
            if best_block_idx < 0:
                break
            
            # Add this block
            selected.append(best_block_idx)
            for tid in block_triplet_ids[best_block_idx]:
                counts[tid] += 1
        
        return [all_blocks[i] for i in selected]
    
    # Enhanced local search with more aggressive 2-for-1 and 3-for-2
    def optimize_enhanced(blocks):
        # Build counts
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for trip in itertools.combinations(b, t):
                counts[triplet_to_id[trip]] += 1
        
        n = len(blocks)
        
        def is_redundant(block):
            for trip in itertools.combinations(block, t):
                tid = triplet_to_id[trip]
                if counts[tid] == 1:
                    return False
            return True
        
        def get_triplet_ids(block):
            return [triplet_to_id[trip] for trip in itertools.combinations(block, t)]
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline - 8:
                break
            
            improved = False
            n = len(blocks)
            
            # 1. Remove redundant blocks
            for i in range(n):
                if is_redundant(blocks[i]):
                    b = blocks.pop(i)
                    for tid in get_triplet_ids(b):
                        counts[tid] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try 2-for-1 replacement
            if n >= 2:
                n_pairs_to_try = min(100, n * (n - 1) // 2)
                tried = set()
                for _ in range(n_pairs_to_try):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j or (i, j) in tried or (j, i) in tried:
                        continue
                    tried.add((i, j))
                    tried.add((j, i))
                    
                    b1 = blocks[i]
                    b2 = blocks[j]
                    ids1 = set(get_triplet_ids(b1))
                    ids2 = set(get_triplet_ids(b2))
                    
                    needed = []
                    for tid in ids1 | ids2:
                        if counts[tid] == 1:
                            needed.append(tid)
                    
                    if not needed:
                        for idx in [j, i]:
                            b_rem = blocks.pop(idx)
                            for tid in get_triplet_ids(b_rem):
                                counts[tid] -= 1
                        improved = True
                        break
                    
                    points_needed = set()
                    for tid in needed:
                        points_needed.update(id_to_triplet[tid])
                    
                    if len(points_needed) > k:
                        continue
                    
                    for new_block in blocks_containing(points_needed):
                        if set(new_block) == set(b1) or set(new_block) == set(b2):
                            continue
                        new_ids = set(get_triplet_ids(new_block))
                        if all(tid in new_ids for tid in needed):
                            for idx in [j, i]:
                                b_rem = blocks.pop(idx)
                                for tid in get_triplet_ids(b_rem):
                                    counts[tid] -= 1
                            blocks.append(new_block)
                            for tid in get_triplet_ids(new_block):
                                counts[tid] += 1
                            improved = True
                            break
                    
                    if improved:
                        break
            
            if not improved:
                # 3. Try 1-for-1 replacement
                if n > 0:
                    i = random.randint(0, n - 1)
                    b1 = blocks[i]
                    ids1 = set(get_triplet_ids(b1))
                    needed = [tid for tid in ids1 if counts[tid] == 1]
                    
                    if needed:
                        points_needed = set()
                        for tid in needed:
                            points_needed.update(id_to_triplet[tid])
                        
                        if len(points_needed) <= k:
                            for new_block in blocks_containing(points_needed):
                                if set(new_block) == set(b1):
                                    continue
                                new_ids = set(get_triplet_ids(new_block))
                                if all(tid in new_ids for tid in needed):
                                    for tid in get_triplet_ids(b1):
                                        counts[tid] -= 1
                                    blocks[i] = new_block
                                    for tid in get_triplet_ids(new_block):
                                        counts[tid] += 1
                                    improved = True
                                    break
            
            if not improved:
                # 4. Try 3-for-2 replacement (new)
                if n >= 3:
                    n_triples_to_try = min(30, n * (n - 1) * (n - 2) // 6)
                    tried = set()
                    for _ in range(n_triples_to_try):
                        idxs = random.sample(range(n), 3)
                        idxs.sort()
                        key = tuple(idxs)
                        if key in tried:
                            continue
                        tried.add(key)
                        
                        b1, b2, b3 = blocks[idxs[0]], blocks[idxs[1]], blocks[idxs[2]]
                        ids1 = set(get_triplet_ids(b1))
                        ids2 = set(get_triplet_ids(b2))
                        ids3 = set(get_triplet_ids(b3))
                        
                        needed = []
                        for tid in ids1 | ids2 | ids3:
                            if counts[tid] == 1:
                                needed.append(tid)
                        
                        if not needed:
                            # All three are redundant
                            for idx in [idxs[2], idxs[1], idxs[0]]:
                                b_rem = blocks.pop(idx)
                                for tid in get_triplet_ids(b_rem):
                                    counts[tid] -= 1
                            improved = True
                            break
                        
                        points_needed = set()
                        for tid in needed:
                            points_needed.update(id_to_triplet[tid])
                        
                        if len(points_needed) > k:
                            continue
                        
                        # Try to find 2 replacement blocks
                        found_replacements = []
                        for new_block1 in blocks_containing(points_needed):
                            if set(new_block1) == set(b1) or set(new_block1) == set(b2) or set(new_block1) == set(b3):
                                continue
                            new_ids1 = set(get_triplet_ids(new_block1))
                            if not all(tid in new_ids1 for tid in needed):
                                continue
                            
                            # Now find a second block to cover remaining needed triplets
                            remaining_needed = [tid for tid in needed if tid not in new_ids1]
                            if not remaining_needed:
                                # First block covers everything, we just need one more block
                                # But we want 2 blocks to replace 3, so this doesn't help directly
                                # Actually, if one block covers all needed, we can replace 3 with 1
                                # Let's handle that case
                                for idx in [idxs[2], idxs[1], idxs[0]]:
                                    b_rem = blocks.pop(idx)
                                    for tid in get_triplet_ids(b_rem):
                                        counts[tid] -= 1
                                blocks.append(new_block1)
                                for tid in get_triplet_ids(new_block1):
                                    counts[tid] += 1
                                improved = True
                                break
                            
                            # Find second block
                            for new_block2 in blocks_containing(points_needed):
                                if set(new_block2) == set(b1) or set(new_block2) == set(b2) or set(new_block2) == set(b3) or set(new_block2) == set(new_block1):
                                    continue
                                new_ids2 = set(get_triplet_ids(new_block2))
                                if all(tid in new_ids2 for tid in remaining_needed):
                                    # Perform swap
                                    for idx in [idxs[2], idxs[1], idxs[0]]:
                                        b_rem = blocks.pop(idx)
                                        for tid in get_triplet_ids(b_rem):
                                            counts[tid] -= 1
                                    blocks.append(new_block1)
                                    blocks.append(new_block2)
                                    for tid in get_triplet_ids(new_block1):
                                        counts[tid] += 1
                                    for tid in get_triplet_ids(new_block2):
                                        counts[tid] += 1
                                    improved = True
                                    break
                            
                            if improved:
                                break
                        
                        if improved:
                            break
            
            if not improved:
                break
        
        return blocks
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy initializations with enhanced optimization
    n_greedy_runs = 10
    for seed in range(n_greedy_runs):
        if time.time() > deadline - 40:
            break
        blocks = fast_greedy(seed)
        if not blocks:
            continue
        
        # Optimize with enhanced local search
        opt_blocks = optimize_enhanced(blocks)
        
        # Ensure validity
        opt_blocks = ensure_valid(opt_blocks)
        
        if is_valid(opt_blocks):
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks
    
    # If no valid solution found, do a pure greedy
    if best_blocks is None:
        blocks = fast_greedy(0)
        best_blocks = ensure_valid(blocks)
    
    return best_blocks

if __name__ == "__main__":
    pass
