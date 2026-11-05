import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip
        
    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Helper to get mask of a block
    def get_block_mask(block):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        return mask

    # Precompute masks for all blocks and triplet lists
    block_masks = [0] * n_blocks
    block_triplets = [None] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        trips = []
        for triplet in itertools.combinations(b, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
            trips.append(tid)
        block_masks[i] = mask
        block_triplets[i] = trips

    # Precompute rarity (degree) of each triplet
    triplet_degrees = np.zeros(n_triplets, dtype=np.int32)
    for i in range(n_blocks):
        for tid in block_triplets[i]:
            triplet_degrees[tid] += 1
            
    # Weights for weighted greedy: 1 / degree
    triplet_weights = 1.0 / triplet_degrees

    # Precompute blocks containing each point for local search
    point_to_blocks = {p: [] for p in range(v)}
    for i, b in enumerate(all_blocks):
        for p in b:
            point_to_blocks[p].append(i)

    # Global Weighted Greedy Construction
    def greedy_weighted(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        selected_indices = []
        
        while uncovered:
            best_gain = -1
            best_indices = []
            
            for i in range(n_blocks):
                mask = block_masks[i]
                common = mask & uncovered
                if common == 0:
                    continue
                gain = 0.0
                for tid in block_triplets[i]:
                    if (uncovered >> tid) & 1:
                        gain += triplet_weights[tid]
                
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain:
                    best_indices.append(i)
                    
            if not best_indices:
                break
                
            idx = random.choice(best_indices)
            selected_indices.append(idx)
            uncovered &= ~block_masks[idx]
            
        return [all_blocks[i] for i in selected_indices]

    # Local Search Optimizer
    def optimize(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def recalc_counts(b_list):
            nonlocal counts
            counts[:] = 0
            for b in b_list:
                idx = block_to_idx[b]
                for tid in block_triplets[idx]:
                    counts[tid] += 1
        
        recalc_counts(blocks)
        
        max_iters = 2000
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            improved = False
            n = len(blocks)
            
            # 1. Try Remove Redundant
            for i in range(n):
                if not improved:
                    b = blocks[i]
                    idx = block_to_idx[b]
                    redundant = True
                    for tid in block_triplets[idx]:
                        if counts[tid] == 1:
                            redundant = False
                            break
                    if redundant:
                        blocks.pop(i)
                        for tid in block_triplets[idx]:
                            counts[tid] -= 1
                        improved = True
                        break
                        
            # 2. Try 2-for-1
            if not improved:
                if n > 50:
                    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
                    random.shuffle(pairs)
                    pairs_to_try = pairs[:500]
                else:
                    pairs_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
                    
                for i, j in pairs_to_try:
                    if time.time() > deadline - 15:
                        break
                        
                    b1 = blocks[i]
                    b2 = blocks[j]
                    idx1 = block_to_idx[b1]
                    idx2 = block_to_idx[b2]
                    
                    ids_b1 = set(block_triplets[idx1])
                    ids_b2 = set(block_triplets[idx2])
                    
                    needed_triplets = []
                    all_involved = ids_b1 | ids_b2
                    for tid in all_involved:
                        if counts[tid] == 1:
                            needed_triplets.append(tid)
                            
                    if not needed_triplets:
                        b_remove = blocks.pop(j)
                        for tid in block_triplets[idx2]:
                            counts[tid] -= 1
                        b_remove = blocks.pop(i)
                        for tid in block_triplets[idx1]:
                            counts[tid] -= 1
                        improved = True
                        break
                    
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) > k:
                        continue
                        
                    candidate_blocks = set()
                    if points_needed:
                        min_point = min(points_needed, key=lambda p: len(point_to_blocks[p]))
                        candidate_blocks = set(point_to_blocks[min_point])
                        for p in points_needed:
                            if p != min_point:
                                candidate_blocks &= set(point_to_blocks[p])
                                if not candidate_blocks:
                                    break
                    else:
                        candidate_blocks = set(range(n_blocks))
                        
                    found_replacement = False
                    for new_idx in candidate_blocks:
                        new_block = all_blocks[new_idx]
                        if new_block == b1 or new_block == b2:
                            continue
                            
                        new_ids = set(block_triplets[new_idx])
                        covers = all(tid in new_ids for tid in needed_triplets)
                        
                        if covers:
                            b_remove = blocks.pop(j)
                            for tid in block_triplets[idx2]:
                                counts[tid] -= 1
                            b_remove = blocks.pop(i)
                            for tid in block_triplets[idx1]:
                                counts[tid] -= 1
                                
                            blocks.append(new_block)
                            for tid in block_triplets[new_idx]:
                                counts[tid] += 1
                                
                            improved = True
                            found_replacement = True
                            break
                            
                    if improved:
                        break
                        
            # 3. Try 1-for-1
            if not improved:
                i = random.randint(0, n-1)
                b1 = blocks[i]
                idx1 = block_to_idx[b1]
                ids_b1 = set(block_triplets[idx1])
                
                needed_triplets = [tid for tid in ids_b1 if counts[tid] == 1]
                
                if needed_triplets:
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) <= k:
                        candidate_blocks = set()
                        if points_needed:
                            min_point = min(points_needed, key=lambda p: len(point_to_blocks[p]))
                            candidate_blocks = set(point_to_blocks[min_point])
                            for p in points_needed:
                                if p != min_point:
                                    candidate_blocks &= set(point_to_blocks[p])
                                    if not candidate_blocks:
                                        break
                        else:
                            candidate_blocks = set(range(n_blocks))
                            
                        for new_idx in candidate_blocks:
                            if new_idx == idx1:
                                continue
                            new_block = all_blocks[new_idx]
                            new_ids = set(block_triplets[new_idx])
                            if all(tid in new_ids for tid in needed_triplets):
                                b_remove = blocks.pop(i)
                                for tid in block_triplets[idx1]:
                                    counts[tid] -= 1
                                blocks.append(new_block)
                                for tid in block_triplets[new_idx]:
                                    counts[tid] += 1
                                improved = True
                                break
                                
            if not improved:
                break
                
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    candidates = []
    
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        blocks = greedy_weighted(seed)
        candidates.append(blocks)
        
    optimized_candidates = []
    for blocks in candidates:
        if time.time() > deadline - 10:
            break
        opt_blocks = optimize(blocks)
        optimized_candidates.append(opt_blocks)
        
    for blocks in optimized_candidates:
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            idx = block_to_idx[b]
            for tid in block_triplets[idx]:
                counts[tid] += 1
                
        if np.min(counts) > 0:
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
    
    if best_blocks is None:
        best_blocks = greedy_weighted(0)
        
    return best_blocks

if __name__ == "__main__":
    pass
