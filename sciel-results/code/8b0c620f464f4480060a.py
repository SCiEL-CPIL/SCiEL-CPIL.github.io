import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 145s hard deadline (5s safety margin from 150s)
    
    # Precomputation
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
            
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]
    
    def get_neighbors(b_idx):
        neigh = set()
        for t_idx in block_t_indices[b_idx]:
            neigh.update(t_subset_blocks[t_idx])
        neigh.discard(b_idx)
        return neigh

    # Phase 1: Branch and Bound for Maximum Independent Set (Packing)
    best_packing = set()
    time_limit_bnb = 25.0
    
    def branch_and_bound(candidates, current_set, depth=0):
        nonlocal best_packing
        nonlocal start_time
        nonlocal time_limit_bnb
        
        if time.time() - start_time > time_limit_bnb:
            return
            
        if not candidates:
            if len(current_set) > len(best_packing):
                best_packing = set(current_set)
            return
            
        # Pruning
        if len(current_set) + len(candidates) <= len(best_packing):
            return
            
        # Select a vertex to branch on - pick one with high degree in candidate set
        candidates_list = list(candidates)
        random.shuffle(candidates_list)
        
        # Sample to find a high-degree vertex
        sample_size = min(15, len(candidates_list))
        sample = candidates_list[:sample_size]
        
        best_v = -1
        max_neigh_count = -1
        
        for v in sample:
            neigh_v = get_neighbors(v)
            count = 0
            for n in neigh_v:
                if n in candidates:
                    count += 1
            if count > max_neigh_count:
                max_neigh_count = count
                best_v = v
                
        if best_v == -1:
            best_v = random.choice(candidates_list)
            
        # Branch 1: Include best_v
        neigh_best_v = get_neighbors(best_v)
        new_candidates_include = set(candidates)
        new_candidates_include.difference_update(neigh_best_v)
        new_candidates_include.discard(best_v)
        
        new_current_set = set(current_set)
        new_current_set.add(best_v)
        
        branch_and_bound(new_candidates_include, new_current_set, depth + 1)
        
        if time.time() - start_time > time_limit_bnb:
            return
            
        # Branch 2: Exclude best_v
        new_candidates_exclude = set(candidates)
        new_candidates_exclude.discard(best_v)
        
        branch_and_bound(new_candidates_exclude, current_set, depth + 1)

    global_candidates = set(range(num_k_subsets))
    branch_and_bound(global_candidates, set())
    
    if not best_packing:
        pool = set(range(num_k_subsets))
        while pool:
            b = random.choice(list(pool))
            best_packing.add(b)
            neigh = get_neighbors(b)
            pool.difference_update(neigh)
            pool.discard(b)

    # Phase 2: Solve Remaining Set Cover
    active_blocks = set(best_packing)
    cover_count = [0] * num_t_subsets
    
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    still_uncovered = [i for i in range(num_t_subsets) if cover_count[i] == 0]
    
    if still_uncovered:
        residual_uncovered = set(still_uncovered)
        
        while residual_uncovered:
            if time.time() > deadline - 30:
                break
                
            # Pick an uncovered t-subset with fewest covering blocks (most constrained)
            min_degree = float('inf')
            anchor_t = None
            for t_idx in residual_uncovered:
                deg = len(t_subset_blocks[t_idx])
                if deg < min_degree:
                    min_degree = deg
                    anchor_t = t_idx
                if deg <= 1:
                    break
            
            if anchor_t is None:
                anchor_t = random.choice(list(residual_uncovered))
                
            candidate_blocks = t_subset_blocks[anchor_t]
            
            best_block = -1
            max_gain = 0
            
            for b_idx in candidate_blocks:
                if b_idx in active_blocks:
                    continue
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if t_idx in residual_uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                    
            if best_block == -1 or max_gain == 0:
                break
                
            active_blocks.add(best_block)
            for t_idx in block_t_indices[best_block]:
                if t_idx in residual_uncovered:
                    residual_uncovered.remove(t_idx)
                cover_count[t_idx] += 1

    # Phase 3: Minimize - remove redundant blocks
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

    active_blocks, cover_count = minimize(active_blocks, cover_count)

    # Phase 4: Local Search - try swaps and removals
    start_time_ls = time.time()
    time_limit_ls = min(100.0, deadline - start_time_ls - 10)
    
    def try_remove(b_idx):
        nonlocal active_blocks, cover_count
        if b_idx not in active_blocks:
            return False
        for t_idx in block_t_indices[b_idx]:
            if cover_count[t_idx] <= 1:
                return False
        active_blocks.remove(b_idx)
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] -= 1
        return True

    def try_swap(old_idx, new_idx):
        nonlocal active_blocks, cover_count
        if old_idx not in active_blocks or new_idx in active_blocks:
            return False
        
        unique_subsets = []
        for t_idx in block_t_indices[old_idx]:
            if cover_count[t_idx] == 1:
                unique_subsets.append(t_idx)
                
        if unique_subsets:
            for t_idx in unique_subsets:
                if t_idx not in block_t_indices[new_idx]:
                    return False

        active_blocks.remove(old_idx)
        active_blocks.add(new_idx)
        
        for t_idx in block_t_indices[old_idx]:
            cover_count[t_idx] -= 1
        for t_idx in block_t_indices[new_idx]:
            cover_count[t_idx] += 1
        return True

    while time.time() - start_time_ls < time_limit_ls:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if B not in active_blocks:
                continue
            
            if try_remove(B):
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                improved = True
                continue
            
            unique_subsets = []
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 1:
                    unique_subsets.append(t_idx)
            
            if unique_subsets:
                candidates = None
                for t_idx in unique_subsets:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    if not candidates:
                        break
                
                if candidates:
                    valid_candidates = [c for c in candidates if c != B and c not in active_blocks]
                    if valid_candidates:
                        # Pick the candidate that covers the most other uncovered subsets
                        best_new = None
                        best_score = -1
                        for c in valid_candidates[:20]:  # Limit to 20 for speed
                            score = 0
                            for t_idx in block_t_indices[c]:
                                if cover_count[t_idx] == 0:
                                    score += 2
                                elif cover_count[t_idx] == 1:
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_new = c
                        if best_new is not None:
                            if try_swap(B, best_new):
                                active_blocks, cover_count = minimize(active_blocks, cover_count)
                                improved = True
        
        if not improved:
            break

    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    return result
