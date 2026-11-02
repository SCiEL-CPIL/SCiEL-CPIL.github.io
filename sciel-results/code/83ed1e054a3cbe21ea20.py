import time
import random
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # Map t-subset to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    # For each block, store the bitset of t-subsets it covers
    block_masks = [0] * num_k_blocks
    # For each t-subset, store the list of blocks that cover it
    ts_to_blocks = [[] for _ in range(num_t_subsets)]

    for j, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            mask |= (1 << idx)
            ts_to_blocks[idx].append(j)
        block_masks[j] = mask

    def run_greedy(seed):
        random.seed(seed)
        uncovered_mask = (1 << num_t_subsets) - 1
        is_selected = [False] * num_k_blocks
        sol = []
        
        while uncovered_mask != 0:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = -1
            best_block = -1
            
            # To speed up, we can skip blocks that are already selected.
            # We can also try to find a good block faster.
            # For now, full scan.
            for b in range(num_k_blocks):
                if is_selected[b]:
                    continue
                gain = bin(block_masks[b] & uncovered_mask).count('1')
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
            
            if best_block == -1:
                break
            
            sol.append(all_k_blocks[best_block])
            is_selected[best_block] = True
            uncovered_mask &= ~block_masks[best_block]
            
        return sol

    def local_search(sol):
        block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
        sol_indices = [block_tuple_to_idx[b] for b in sol]
        sol_set = set(sol_indices)
        
        sol_mask = 0
        for idx in sol_indices:
            sol_mask |= block_masks[idx]
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            
            # Phase 1: Remove redundant blocks
            changed = True
            while changed:
                changed = False
                new_sol_indices = []
                for idx in sol_indices:
                    if time.time() > deadline - 5.0:
                        return [all_k_blocks[i] for i in sol_indices]
                    others_mask = sol_mask ^ block_masks[idx]
                    unique_bits = block_masks[idx] & ~others_mask
                    if unique_bits == 0:
                        sol_mask = others_mask
                    else:
                        new_sol_indices.append(idx)
                
                if len(new_sol_indices) < len(sol_indices):
                    sol_indices = new_sol_indices
                    sol_set = set(sol_indices)
                    improved = True
                    changed = True
                    sol_mask = 0
                    for idx in sol_indices:
                        sol_mask |= block_masks[idx]
                else:
                    break
            
            # Phase 2: 1-Exchange
            for i in range(len(sol_indices)):
                if time.time() > deadline - 5.0:
                    return [all_k_blocks[j] for j in sol_indices]
                idx = sol_indices[i]
                others_mask = sol_mask ^ block_masks[idx]
                unique_bits = block_masks[idx] & ~others_mask
                
                if unique_bits == 0:
                    sol_indices.pop(i)
                    sol_set.remove(idx)
                    sol_mask ^= block_masks[idx]
                    improved = True
                    break
                
                temp = unique_bits
                if temp == 0:
                    continue
                lsb = temp & -temp
                bit_idx = lsb.bit_length() - 1
                
                candidates = ts_to_blocks[bit_idx]
                random.shuffle(candidates)
                
                swapped = False
                for k in candidates:
                    if k in sol_set:
                        continue
                    if (unique_bits & ~block_masks[k]) == 0:
                        sol_indices[i] = k
                        sol_set.remove(idx)
                        sol_set.add(k)
                        sol_mask = (sol_mask ^ block_masks[idx]) | block_masks[k]
                        improved = True
                        swapped = True
                        break
                
                if swapped:
                    break
            
            # Phase 3: 2-for-1 Exchange
            if len(sol_indices) > 100 and time.time() < deadline - 10.0:
                num_pairs_to_check = 200
                checked = 0
                while checked < num_pairs_to_check:
                    if time.time() > deadline - 5.0:
                        break
                    i = random.randint(0, len(sol_indices) - 1)
                    j = random.randint(0, len(sol_indices) - 1)
                    if i == j:
                        continue
                    checked += 1
                    
                    idx_i = sol_indices[i]
                    idx_j = sol_indices[j]
                    
                    mask_i = block_masks[idx_i]
                    mask_j = block_masks[idx_j]
                    
                    others_mask_ij = sol_mask ^ mask_i ^ mask_j
                    
                    unique_i = mask_i & ~others_mask_ij
                    unique_j = mask_j & ~others_mask_ij
                    U = unique_i | unique_j
                    
                    if U == 0:
                        continue
                    
                    temp = U
                    lsb = temp & -temp
                    bit_idx = lsb.bit_length() - 1
                    
                    candidates = ts_to_blocks[bit_idx]
                    random.shuffle(candidates)
                    
                    found = False
                    for k in candidates:
                        if k in sol_set:
                            continue
                        if (U & ~block_masks[k]) == 0:
                            if i > j:
                                i, j = j, i
                            sol_indices.pop(j)
                            sol_set.remove(idx_j)
                            i = i - 1
                            sol_indices.pop(i)
                            sol_set.remove(idx_i)
                            
                            sol_indices.append(k)
                            sol_set.add(k)
                            
                            sol_mask = (sol_mask ^ mask_i ^ mask_j) | block_masks[k]
                            
                            improved = True
                            found = True
                            break
                    
                    if found:
                        break

        return [all_k_blocks[idx] for idx in sol_indices]

    best_solution = None
    best_len = float('inf')
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        sol = run_greedy(i)
        if sol is None:
            continue
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    if best_solution is not None:
        while time.time() < deadline - 5.0:
            new_sol = local_search(best_solution)
            if len(new_sol) < len(best_solution):
                best_solution = new_sol
                best_len = len(new_sol)
            else:
                break

    if best_solution is None:
        best_solution = run_greedy(0)
        if best_solution is None:
            best_solution = [all_k_blocks[0]]

    return best_solution
