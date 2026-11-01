import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute 5-subsets for all 8-subsets
    # Map 8-tuple to list of 5-subset indices
    all_k_subsets = list(itertools.combinations(range(v), k))
    k_subset_to_idx = {s: i for i, s in enumerate(all_k_subsets)}
    
    # For each k-subset, store the list of t-subset indices it covers
    k_subsets_t_indices = {}
    for k_sub in all_k_subsets:
        indices = []
        for t_sub in itertools.combinations(k_sub, t):
            indices.append(subset_to_idx[t_sub])
        k_subsets_t_indices[k_sub] = indices
        
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    # With precomputation, greedy step is faster.
    # 300 blocks * 364 candidates * (56 lookups + sum)
    # Sum is still 56 ops.
    # 300 * 364 * 56 = 6M ops.
    # Precomputation cost: 75582 * 56 = 4.2M ops.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        
        is_covered = [False] * num_t_subsets
        coverage_count = [0] * num_t_subsets
        
        uncovered_list = list(range(num_t_subsets))
        uncovered_pos = {i: i for i in range(num_t_subsets)}
        
        blocks = []
        
        while uncovered_list:
            list_idx = random.randint(0, len(uncovered_list) - 1)
            sub_idx = uncovered_list[list_idx]
            t_subset = all_t_subsets[sub_idx]
            
            points_in_t = set(t_subset)
            points_not_in_t = [p for p in range(v) if p not in points_in_t]
            
            best_gain = -1
            best_candidate = None
            
            for combo in itertools.combinations(points_not_in_t, k - t):
                candidate_block = tuple(sorted(t_subset + combo))
                
                # Get precomputed indices
                sub_indices = k_subsets_t_indices[candidate_block]
                
                gain = 0
                for idx in sub_indices:
                    if not is_covered[idx]:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate_block
                    
            if best_candidate is not None:
                blocks.append(best_candidate)
                
                sub_indices = k_subsets_t_indices[best_candidate]
                for idx in sub_indices:
                    coverage_count[idx] += 1
                    if not is_covered[idx]:
                        is_covered[idx] = True
                        pos = uncovered_pos[idx]
                        last_idx = uncovered_list[-1]
                        uncovered_list[pos] = last_idx
                        uncovered_pos[last_idx] = pos
                        uncovered_list.pop()
                        del uncovered_pos[idx]
                        
        # Local Search
        changed = True
        while changed:
            changed = False
            to_remove = []
            for i, block in enumerate(blocks):
                is_redundant = True
                sub_indices = k_subsets_t_indices[block]
                for idx in sub_indices:
                    if coverage_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                for i in reversed(to_remove):
                    block = blocks[i]
                    sub_indices = k_subsets_t_indices[block]
                    for idx in sub_indices:
                        coverage_count[idx] -= 1
                    blocks.pop(i)
                changed = True
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    return best_blocks if best_blocks else []
