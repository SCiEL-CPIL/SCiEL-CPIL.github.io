import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Precompute all t-subsets and map them to indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Precompute all k-blocks and their coverage bitmasks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_masks = []
    block_subs = []
    
    for block in all_blocks:
        mask = 0
        subs = []
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks.append(mask)
        block_subs.append(subs)
        
    # 3. Precompute for each subset, which blocks cover it (as sets for fast intersection)
    sub_to_blocks_list = [[] for _ in range(n_t)]
    for b_idx, subs in enumerate(block_subs):
        for s_idx in subs:
            sub_to_blocks_list[s_idx].append(b_idx)
            
    sub_to_blocks_set = [set(lst) for lst in sub_to_blocks_list]
    
    def time_ok():
        return time.time() < deadline

    def greedy_cover(seed=0):
        random.seed(seed)
        selected = set()
        covered_mask = 0
        
        while True:
            if not time_ok():
                break
            if covered_mask == (1 << n_t) - 1:
                break
                
            # Find subset with min candidates among uncovered
            target_sub = -1
            min_cands = float('inf')
            
            for s in range(n_t):
                if not (covered_mask >> s) & 1:
                    c = len(sub_to_blocks_list[s])
                    if c < min_cands:
                        min_cands = c
                        target_sub = s
                        if c == 1:
                            break
                            
            if target_sub == -1:
                break
                
            candidates = sub_to_blocks_list[target_sub]
            
            best_block = -1
            max_gain = -1
            best_cands = []
            
            for b in candidates:
                if b in selected:
                    continue
                gain_mask = block_masks[b] & ~covered_mask
                gain = bin(gain_mask).count('1')
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
                    best_cands = [b]
                elif gain == max_gain:
                    best_cands.append(b)
                    
            if not best_cands:
                break
                
            chosen = random.choice(best_cands)
            selected.add(chosen)
            covered_mask |= block_masks[chosen]
                
        return selected

    def local_search(selected_set, max_iter=100):
        selected = list(selected_set)
        counts = [0] * n_t
        for b in selected:
            for s in block_subs[b]:
                counts[s] += 1
        
        improved = True
        iterations = 0
        while improved and time_ok() and iterations < max_iter:
            improved = False
            iterations += 1
            
            # Pass 1: Try to remove any block
            for i in range(len(selected)):
                b = selected[i]
                safe = True
                for s in block_subs[b]:
                    if counts[s] == 1:
                        safe = False
                        break
                if safe:
                    for s in block_subs[b]:
                        counts[s] -= 1
                    selected.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Pass 2: Try to swap
            for i in range(len(selected)):
                b = selected[i]
                critical = []
                for s in block_subs[b]:
                    if counts[s] == 1:
                        critical.append(s)
                        
                if not critical:
                    continue
                    
                candidates = sub_to_blocks_set[critical[0]]
                for s in critical[1:]:
                    candidates = candidates & sub_to_blocks_set[s]
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Swap with a random candidate
                b_prime = random.choice(list(candidates))
                
                for s in block_subs[b]:
                    counts[s] -= 1
                for s in block_subs[b_prime]:
                    counts[s] += 1
                selected[i] = b_prime
                improved = True
                break
                
        return set(selected)

    best_solution = None
    best_size = float('inf')
    
    current_solution = greedy_cover(42)
    current_solution = local_search(current_solution)
    best_solution = current_solution
    best_size = len(current_solution)
    
    iter_count = 0
    while time_ok():
        iter_count += 1
        
        # Perturbation
        perturbed = list(current_solution)
        num_remove = random.randint(1, 3)
        if len(perturbed) > num_remove:
            to_remove = random.sample(perturbed, num_remove)
            for b in to_remove:
                perturbed.remove(b)
            
            # Repair
            counts = [0] * n_t
            for b in perturbed:
                for s in block_subs[b]:
                    counts[s] += 1
            
            covered_mask = 0
            for b in perturbed:
                covered_mask |= block_masks[b]
                
            uncovered = (1 << n_t) - 1
            uncovered &= ~covered_mask
            
            while uncovered:
                s = (uncovered & -uncovered).bit_length() - 1
                candidates = sub_to_blocks_list[s]
                
                best_b = -1
                max_gain = -1
                for b in candidates:
                    if b in perturbed:
                        continue
                    gain = bin(block_masks[b] & uncovered).count('1')
                    if gain > max_gain:
                        max_gain = gain
                        best_b = b
                        
                if best_b == -1:
                    for b in candidates:
                        if b not in perturbed:
                            best_b = b
                            break
                if best_b == -1:
                    break
                    
                perturbed.append(best_b)
                uncovered &= ~block_masks[best_b]
                
            if uncovered == 0:
                repaired_set = set(perturbed)
                local_sol = local_search(repaired_set, max_iter=50)
                
                if len(local_sol) < best_size:
                    best_size = len(local_sol)
                    best_solution = local_sol
                    current_solution = local_sol
                elif len(local_sol) == best_size and random.random() < 0.5:
                    current_solution = local_sol

    if best_solution is None:
        best_solution = greedy_cover(0)
        
    return [list(all_blocks[b]) for b in best_solution]
