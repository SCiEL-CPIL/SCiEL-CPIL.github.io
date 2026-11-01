import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 150s kill
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Precompute degrees of t-subsets (number of blocks covering each)
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]

    # Precompute scarcity weights for each block
    # weight[b] = sum of 1/degree[t] for t in block
    # This helps in the "Packing" phase to prefer blocks covering rare subsets
    block_scarcity_weights = [0.0] * num_k_subsets
    for b_idx, indices in enumerate(block_t_indices):
        w = 0.0
        for t_idx in indices:
            deg = t_subset_degrees[t_idx]
            if deg > 0:
                w += 1.0 / deg
        block_scarcity_weights[b_idx] = w

    # 2. Phase 1: Weighted Greedy Packing
    # Select blocks that cover many uncovered items and have high scarcity weight.
    # Conflict constraint: Do not select blocks that share any t-subset with already selected blocks?
    # No, that's too strict for a covering problem. We want a covering, not a packing.
    # However, the inspiration suggests "Greedy Packing" (maximal independent set heuristic) 
    # followed by repair. Let's stick to the inspiration's logic:
    # Select a core of blocks that are "non-conflicting" in terms of t-subsets?
    # If we strictly avoid sharing t-subsets, we are doing a packing, not a covering.
    # The inspiration code removes neighbors from the pool. This creates a set of blocks 
    # that are pairwise disjoint in terms of t-subsets. This is a very sparse core.
    # Then we repair the rest. This is a valid strategy.
    
    pool = set(range(num_k_subsets))
    active_blocks = set()
    cover_count = [0] * num_t_subsets
    uncovered_mask = [1] * num_t_subsets
    
    # We will run the packing phase. 
    # Score: Maximize (uncovered_count * 100 + scarcity_sum)
    # But we must remove conflicts to ensure the "packing" property.
    
    # Optimization: Instead of scanning the whole pool every time (O(N^2)), 
    # we can just scan. N is ~50k, iterations ~100-200. 10M ops. Acceptable.
    
    while pool:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        best_block = -1
        max_score = -1.0
        
        # To speed up, we might not scan the whole pool if it's huge, 
        # but 50k is small enough.
        for b_idx in pool:
            # Calculate score
            cov_uncovered = 0
            scarcity_sum = 0.0
            for t_idx in block_t_indices[b_idx]:
                if uncovered_mask[t_idx]:
                    cov_uncovered += 1
                    scarcity_sum += 1.0 / t_subset_degrees[t_idx]
            
            if cov_uncovered == 0:
                continue
                
            # Combine: Maximize covered count, tie-break by scarcity
            score = cov_uncovered * 100.0 + scarcity_sum
            
            if score > max_score:
                max_score = score
                best_block = b_idx
                
        if best_block == -1:
            break
            
        active_blocks.add(best_block)
        
        # Update cover_count and uncovered_mask
        for t_idx in block_t_indices[best_block]:
            cover_count[t_idx] += 1
            if cover_count[t_idx] == 1:
                uncovered_mask[t_idx] = 0
                
        # Remove best_block and its conflicts from pool
        # Conflicts: Any block sharing a t-subset with best_block
        neighbors_to_remove = set()
        for t_idx in block_t_indices[best_block]:
            neighbors_to_remove.update(t_subset_blocks[t_idx])
        
        pool.difference_update(neighbors_to_remove)
        
        if time.time() - start_time > TIME_LIMIT:
            break

    # 3. Phase 2: Residual Set Cover (Repair) - Optimized
    # Use candidate collection from uncovered t-subsets
    
    uncovered_t_indices = [i for i in range(num_t_subsets) if cover_count[i] == 0]
    
    while uncovered_t_indices:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        candidates = set()
        for t_idx in uncovered_t_indices:
            candidates.update(t_subset_blocks[t_idx])
            
        candidates.difference_update(active_blocks)
        
        if not candidates:
            break
            
        best_block = -1
        max_gain = 0
        
        for b_idx in candidates:
            gain = 0
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx
                
        if best_block == -1 or max_gain == 0:
            break
            
        active_blocks.add(best_block)
        for t_idx in block_t_indices[best_block]:
            if cover_count[t_idx] == 0:
                cover_count[t_idx] = 1
                
        uncovered_t_indices = [i for i in uncovered_t_indices if cover_count[i] == 0]

    # 4. Minimalization Helper
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return active_blocks_set, cover_counts

    # Initial minimalization
    active_blocks, cover_count = minimize(active_blocks, cover_count)
    
    best_count = len(active_blocks)
    best_blocks = set(active_blocks)
    best_cover = list(cover_count)

    # 5. Local Search with Restarts
    current_blocks = set(active_blocks)
    current_cover = list(cover_count)
    
    while time.time() - start_time < TIME_LIMIT:
        improved = False
        current_blocks_list = list(current_blocks)
        random.shuffle(current_blocks_list)
        
        for B in current_blocks_list:
            if time.time() - start_time > TIME_LIMIT:
                break
            if B not in current_blocks:
                continue
            
            uncovered_by_B = []
            for t_idx in block_t_indices[B]:
                if current_cover[t_idx] == 1:
                    uncovered_by_B.append(t_idx)
            
            if not uncovered_by_B:
                continue

            uncovered_by_B.sort(key=lambda x: len(t_subset_blocks[x]))
            
            candidates = None
            for t_idx in uncovered_by_B:
                blocks_list = t_subset_blocks[t_idx]
                if candidates is None:
                    candidates = set(blocks_list)
                else:
                    candidates.intersection_update(blocks_list)
                
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            valid_candidates = [c for c in candidates if c != B and c not in current_blocks]
            
            if not valid_candidates:
                continue
            
            B_new = random.choice(valid_candidates)
            
            current_blocks.remove(B)
            current_blocks.add(B_new)
            
            for t_idx in block_t_indices[B]:
                current_cover[t_idx] -= 1
            for t_idx in block_t_indices[B_new]:
                current_cover[t_idx] += 1
            
            current_blocks, current_cover = minimize(current_blocks, current_cover)
            improved = True
            
            if len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = set(current_blocks)
                best_cover = list(current_cover)

        if not improved:
            # Stuck. Perturb and restart local search
            if time.time() - start_time > TIME_LIMIT - 10:
                break
                
            # Perturbation
            num_adds = random.randint(3, 5)
            for _ in range(num_adds):
                b_idx = random.randint(0, num_k_subsets - 1)
                if b_idx in current_blocks:
                    continue
                current_blocks.add(b_idx)
                for t_idx in block_t_indices[b_idx]:
                    current_cover[t_idx] += 1
                    
            current_blocks, current_cover = minimize(current_blocks, current_cover)

    # Return best result found
    result = [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
    return result
