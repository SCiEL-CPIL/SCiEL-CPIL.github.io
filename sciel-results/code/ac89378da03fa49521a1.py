import itertools
import time
import random
import heapq
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation using NumPy
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build incidence matrix: blocks x tsets (0 or 1)
    # For each block, which t-sets it covers
    block_tsets = np.zeros((num_blocks, num_tsets), dtype=np.uint8)
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = tset_index[ts]
            block_tsets[i, idx] = 1
            tset_to_blocks[idx].append(i)
    
    # Convert tset_to_blocks to numpy arrays for faster access
    tset_to_blocks_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) 
                         for lst in tset_to_blocks]
    
    full_count = num_tsets

    # Helper: Get cover counts (vectorized)
    def get_cover_counts(block_indices):
        if len(block_indices) == 0:
            return np.zeros(num_tsets, dtype=np.int32)
        idx_arr = np.array(block_indices, dtype=np.int32)
        return np.sum(block_tsets[idx_arr], axis=0, dtype=np.int32)

    # Helper: Check validity
    def is_valid_solution(blocks):
        if len(blocks) == 0:
            return num_tsets == 0
        idx_arr = np.array(blocks, dtype=np.int32)
        covered = np.sum(block_tsets[idx_arr], axis=0)
        return np.all(covered >= 1)

    # 2. Greedy (vectorized)
    def greedy_cover(seed):
        rng = random.Random(seed)
        covered = np.zeros(num_tsets, dtype=np.int32)
        selected = []
        
        while True:
            if elapsed() > deadline - 5:
                break
            
            # Vectorized: find best block
            uncovered = (covered == 0).astype(np.int32)
            gains = block_tsets @ uncovered  # vector of gains for each block
            
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
                
            selected.append(best_idx)
            covered += block_tsets[best_idx]
            
        return selected

    # 3. Local Search (vectorized)
    def local_search(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True
            
        cover_counts = get_cover_counts(sol)
        
        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 80
        
        while elapsed() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Remove redundant
            order = list(range(len(sol)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(sol):
                    continue
                b = sol[pos]
                # Check if any t-set is only covered by this block
                only_covered = (cover_counts == 1) & (block_tsets[b] == 1)
                if not np.any(only_covered):
                    # Redundant
                    cover_counts -= block_tsets[b]
                    in_sol[b] = False
                    sol.pop(pos)
                    improved = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    break
                    
            if improved:
                no_improve = 0
                continue
                
            # Phase 2: Swap
            if sol:
                found_swap = False
                attempts = 30
                if elapsed() < deadline - 30:
                    attempts = 60
                    
                for _ in range(attempts):
                    if not sol:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]
                    
                    # Find t-sets that are only covered by b_out
                    only_covered_mask = (cover_counts == 1) & (block_tsets[b_out] == 1)
                    if not np.any(only_covered_mask):
                        # Redundant
                        cover_counts -= block_tsets[b_out]
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break
                    
                    only_covered_idx = np.where(only_covered_mask)[0]
                    
                    # Find candidate blocks that cover all only-covered t-sets
                    # and are not already in solution
                    candidates = []
                    for ts_idx in only_covered_idx:
                        cand_list = tset_to_blocks_np[ts_idx]
                        if len(cand_list) == 0:
                            candidates = []
                            break
                        if not candidates:
                            candidates = cand_list.tolist()
                        else:
                            candidates = [c for c in candidates if c in set(cand_list.tolist())]
                    
                    # Filter: must not be in solution, and must cover all only-covered t-sets
                    valid_candidates = []
                    for b_in in candidates:
                        if in_sol[b_in]:
                            continue
                        # Check if b_in covers all only-covered t-sets
                        if np.all(block_tsets[b_in, only_covered_idx] == 1):
                            valid_candidates.append(b_in)
                    
                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        # Update counts
                        cover_counts -= block_tsets[b_out]
                        cover_counts += block_tsets[b_in]
                        in_sol[b_out] = False
                        in_sol[b_in] = True
                        sol[pos] = b_in
                        found_swap = True
                        improved = True
                        break
                        
                if found_swap:
                    no_improve = 0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue
            else:
                break
                
            no_improve += 1
            
        return best_sol

    # 4. Iterated Local Search
    best_solution = None
    best_size = float('inf')
    
    # Initial run
    sol = greedy_cover(0)
    ls_sol = local_search(sol, 0)
    best_solution = ls_sol
    best_size = len(ls_sol)
    
    # ILS Loop
    current_sol = best_solution
    while elapsed() < deadline - 15:
        # Perturb
        perturbed = list(current_sol)
        rng = random.Random(int(elapsed() * 1000))
        
        # Remove random blocks
        num_remove = min(3, len(perturbed))
        for _ in range(num_remove):
            if not perturbed:
                break
            idx = rng.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
            
        # Fix coverage by adding blocks greedily
        covered = np.zeros(num_tsets, dtype=np.int32)
        for b in perturbed:
            covered += block_tsets[b]
        
        while True:
            if elapsed() > deadline - 10:
                break
            uncovered = (covered == 0).astype(np.int32)
            if not np.any(uncovered):
                break
            gains = block_tsets @ uncovered
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            perturbed.append(best_idx)
            covered += block_tsets[best_idx]
        
        # Local Search on Perturbed
        ls_sol = local_search(perturbed, int(elapsed() * 1000))
        
        if len(ls_sol) < best_size:
            best_size = len(ls_sol)
            best_solution = ls_sol
            current_sol = best_solution

    return [list(all_blocks[i]) for i in best_solution]
