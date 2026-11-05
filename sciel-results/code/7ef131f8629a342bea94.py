import itertools
import random
import time

def construct(v, k, t):
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    subset_to_idx = {sub: i for i, sub in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block to subset indices
    block_subsets_idx = []
    for b in all_blocks:
        subs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_subsets_idx.append(subs)
        
    # Precompute subset to block indices (inverse)
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for i, subs in enumerate(block_subsets_idx):
        for s in subs:
            subset_to_blocks[s].append(i)
            
    # Precompute sets for fast membership testing in local search
    block_subsets_set = [set(subs) for subs in block_subsets_idx]

    def weighted_greedy(seed):
        random.seed(seed)
        uncovered = set(range(num_subsets))
        selected_blocks = []
        
        while uncovered:
            best_block_idx = -1
            best_score = -1.0
            
            for i in range(num_blocks):
                subs = block_subsets_idx[i]
                gain = 0
                overlap = 0
                for s in subs:
                    if s in uncovered:
                        gain += 1
                    else:
                        overlap += 1
                
                if gain == 0:
                    continue
                    
                # Weighted score: prefer high gain, low overlap
                score = gain / (1 + overlap)
                
                if score > best_score:
                    best_score = score
                    best_block_idx = i
                elif score == best_score:
                    if random.random() < 0.5:
                        best_block_idx = i

            if best_block_idx == -1:
                break

            selected_blocks.append(best_block_idx)
            for s in block_subsets_idx[best_block_idx]:
                uncovered.discard(s)
                
        return selected_blocks

    def local_search(initial_blocks):
        in_solution = [False] * num_blocks
        for b in initial_blocks:
            in_solution[b] = True
            
        coverage_count = [0] * num_subsets
        for b in initial_blocks:
            for s in block_subsets_idx[b]:
                coverage_count[s] += 1
                
        current_blocks = list(initial_blocks)
        
        max_iters = 20
        for _ in range(max_iters):
            changed = False
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in current_blocks:
                is_redundant = True
                for s in block_subsets_idx[b]:
                    if coverage_count[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(b)
            
            for b in to_remove:
                in_solution[b] = False
                current_blocks.remove(b)
                for s in block_subsets_idx[b]:
                    coverage_count[s] -= 1
                changed = True
            
            if not changed:
                # Phase 2: Try swaps
                random.shuffle(current_blocks)
                for b_in in current_blocks:
                    in_solution[b_in] = False
                    missing = []
                    for s in block_subsets_idx[b_in]:
                        coverage_count[s] -= 1
                        if coverage_count[s] == 0:
                            missing.append(s)
                    
                    if not missing:
                        changed = True
                        continue
                    
                    random.shuffle(missing)
                    s0 = missing[0]
                    found_swap = False
                    
                    for b_out in subset_to_blocks[s0]:
                        if in_solution[b_out]:
                            continue
                        
                        # Check if b_out covers all missing
                        covers_all = True
                        for s in missing:
                            if s not in block_subsets_set[b_out]:
                                covers_all = False
                                break
                        
                        if covers_all:
                            in_solution[b_out] = True
                            current_blocks.append(b_out)
                            for s in block_subsets_idx[b_out]:
                                coverage_count[s] += 1
                            found_swap = True
                            changed = True
                            break
                    
                    if not found_swap:
                        in_solution[b_in] = True
                        for s in block_subsets_idx[b_in]:
                            coverage_count[s] += 1
                    else:
                        try:
                            idx = current_blocks.index(b_in)
                            current_blocks.pop(idx)
                        except ValueError:
                            pass
                        break

            if not changed:
                break
                
        return current_blocks

    best_len = float('inf')
    best_blocks = []
    
    start_time = time.time()
    for seed in range(5):
        if time.time() - start_time > 100:
            break
        init_blocks = weighted_greedy(seed)
        refined_blocks = local_search(init_blocks)
        if len(refined_blocks) < best_len:
            best_len = len(refined_blocks)
            best_blocks = refined_blocks

    return [list(all_blocks[i]) for i in best_blocks]
