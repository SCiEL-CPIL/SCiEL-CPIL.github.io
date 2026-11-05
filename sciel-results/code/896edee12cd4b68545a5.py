import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # 1. Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks and their coverage as lists of t-subset indices
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Build coverage: for each block, list of t-subset indices it covers
    block_cover = []
    for block in all_k_blocks:
        cover = []
        for t_sub in itertools.combinations(block, t):
            cover.append(sub_to_idx[t_sub])
        block_cover.append(cover)
    
    # Build inverse: for each t-subset, which blocks cover it
    # Use a numpy array for fast vectorized operations
    # coverage_matrix[i, j] = 1 if block i covers t-subset j
    coverage_matrix = np.zeros((n_k_blocks, n_t_subs), dtype=np.int8)
    for i, cover in enumerate(block_cover):
        for idx in cover:
            coverage_matrix[i, idx] = 1
    
    # 3. Fast greedy with numpy vectorization
    def greedy_cover(randomize=False):
        uncovered = np.ones(n_t_subs, dtype=bool)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline - 60:
                break
            
            # Compute gain for each block: number of uncovered t-subsets it covers
            gains = coverage_matrix[:, uncovered].sum(axis=1)
            
            if randomize:
                # Among top gainers, pick randomly
                max_gain = gains.max()
                if max_gain == 0:
                    break
                top_candidates = np.where(gains == max_gain)[0]
                if len(top_candidates) > 1:
                    # Pick randomly from top candidates
                    pick = random.choice(top_candidates.tolist())
                else:
                    pick = top_candidates[0]
            else:
                pick = gains.argmax()
            
            if gains[pick] == 0:
                break
            
            selected.append(pick)
            uncovered[coverage_matrix[pick] == 1] = False
        
        return selected
    
    # 4. Redundancy removal
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Compute coverage counts
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b in blocks:
            cover_count[coverage_matrix[b] == 1] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, b in enumerate(blocks):
                # Check if any t-subset is only covered by this block
                block_cover_mask = coverage_matrix[b] == 1
                if not (block_cover_mask & (cover_count == 1)).any():
                    # Remove this block
                    cover_count[block_cover_mask] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        return blocks
    
    # 5. Local search: try to reduce count by finding 1-for-2 or better swaps
    def local_search(blocks, time_limit=30):
        blocks = blocks[:]
        start_ls = time.time()
        
        improved = True
        while improved and time.time() < start_ls + time_limit:
            improved = False
            cover_count = np.zeros(n_t_subs, dtype=np.int32)
            for b in blocks:
                cover_count[coverage_matrix[b] == 1] += 1
            
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    return blocks
                
                b = blocks[i]
                block_mask = coverage_matrix[b] == 1
                # Find t-subsets uniquely covered by this block
                unique_to_block = block_mask & (cover_count == 1)
                
                if not unique_to_block.any():
                    # Can remove
                    blocks.pop(i)
                    cover_count[block_mask] -= 1
                    improved = True
                    break
                
                # Try to find a block that covers all unique_to_block
                # and doesn't add too much overhead
                target = unique_to_block
                # Find candidate blocks that cover all target subsets
                # A candidate must have 1s in all positions where target has 1s
                candidate_mask = np.all(coverage_matrix[:, target] == 1, axis=1)
                candidates = np.where(candidate_mask)[0]
                
                for c in candidates:
                    if c in blocks:
                        continue
                    # Check: replacing b with c should not leave anything uncovered
                    # c must cover everything b uniquely covered (guaranteed by candidate_mask)
                    # Also, after removal of b, any subset that b covered and c doesn't
                    # must still be covered by other blocks
                    new_mask = coverage_matrix[c] == 1
                    # Subsets covered by b but not by c
                    lost = block_mask & ~new_mask
                    # These must be covered by other blocks (cover_count[b] - 1 >= 1 for lost)
                    if lost.any():
                        # Check if other blocks cover them
                        other_cover = cover_count[lost] - 1  # subtract b's contribution
                        if (other_cover < 1).any():
                            continue
                    
                    # Perform swap
                    blocks[i] = c
                    cover_count[block_mask] -= 1
                    cover_count[new_mask] += 1
                    improved = True
                    break
            
            if improved:
                continue
        
        return remove_redundant(blocks)
    
    # 6. Run multiple randomized greedy restarts
    best_solution = None
    best_size = float('inf')
    
    # First: deterministic greedy
    sol = greedy_cover(randomize=False)
    sol = remove_redundant(sol)
    if len(sol) < best_size:
        best_size = len(sol)
        best_solution = sol[:]
    
    # Then: randomized restarts
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 40:
            break
        
        sol = greedy_cover(randomize=True)
        sol = remove_redundant(sol)
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]
    
    # 7. Apply local search to best solution
    if time.time() < deadline - 20 and best_solution is not None:
        best_solution = local_search(best_solution, time_limit=min(30, deadline - time.time() - 15))
        best_size = len(best_solution)
    
    # 8. LP-based rounding if time permits
    if time.time() < deadline - 25 and best_solution is not None:
        # Build pool from best solution + random blocks
        pool_set = set(best_solution)
        remaining = [i for i in range(n_k_blocks) if i not in pool_set]
        random.shuffle(remaining)
        pool_size = min(n_k_blocks, 3000)
        pool = list(pool_set) + remaining[:max(0, pool_size - len(pool_set))]
        n_pool = len(pool)
        pool_to_idx = {b: i for i, b in enumerate(pool)}
        
        # Build A matrix
        A = np.zeros((n_t_subs, n_pool), dtype=np.float64)
        for pi, bi in enumerate(pool):
            A[coverage_matrix[bi] == 1, pi] = 1.0
        
        c = np.ones(n_pool, dtype=np.float64)
        b_vec = np.ones(n_t_subs, dtype=np.float64)
        bounds = [(0, 1)] * n_pool
        
        try:
            result = linprog(c, A_ub=-A.T, b_ub=-b_vec, bounds=bounds, method='highs', maxiter=3000)
            if result.success and result.x is not None:
                x_lp = result.x
                
                for round_num in range(5):
                    if time.time() > deadline - 10:
                        break
                    
                    sampled = []
                    for i in range(n_pool):
                        if x_lp[i] > 0.05 and random.random() < x_lp[i]:
                            sampled.append(pool[i])
                    
                    # Complete with greedy
                    uncovered = np.ones(n_t_subs, dtype=bool)
                    for bi in sampled:
                        uncovered[coverage_matrix[bi] == 1] = False
                    
                    if uncovered.any():
                        while uncovered.any():
                            if time.time() > deadline - 8:
                                break
                            gains = coverage_matrix[:, uncovered].sum(axis=1)
                            pick = gains.argmax()
                            if gains[pick] == 0:
                                break
                            sampled.append(pick)
                            uncovered[coverage_matrix[pick] == 1] = False
                    
                    sampled = remove_redundant(sampled)
                    if len(sampled) < best_size:
                        best_size = len(sampled)
                        best_solution = sampled[:]
                        
        except Exception:
            pass
    
    # Final redundancy removal
    if best_solution is not None:
        best_solution = remove_redundant(best_solution)
    
    # Convert to list of lists
    blocks = [list(all_k_blocks[i]) for i in best_solution]
    
    return blocks
