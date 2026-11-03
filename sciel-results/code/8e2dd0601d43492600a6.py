import itertools
import random
import sys
import time

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an optimized greedy set-cover
    with bitset acceleration, randomized restarts, and local search improvements.
    
    Returns a list of blocks, each a sorted list of k distinct integers in 0..v-1.
    """
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    full_mask = (1 << n_t) - 1
    
    # Precompute all blocks and their coverage masks
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    def popcount(x):
        return x.bit_count()

    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
            
        covered = 0
        uncovered = full_mask
        solution = []
        
        while uncovered:
            best_gain = 0
            candidates = []
            max_possible = popcount(uncovered)
            
            # Find the block that covers the most uncovered t-subsets
            for i, mask in enumerate(block_masks):
                gain = popcount(mask & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                    if gain == max_possible:
                        break # Can't get better
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            if len(candidates) == 1:
                best_block_idx = candidates[0]
            else:
                best_block_idx = random.choice(candidates)
                
            solution.append(all_blocks[best_block_idx])
            covered |= block_masks[best_block_idx]
            uncovered = full_mask ^ covered
            
        return solution

    def optimize_solution(sol):
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        sol_indices = [block_to_idx[b] for b in sol]
        
        # Remove redundant blocks
        improved = True
        while improved:
            improved = False
            n_sol = len(sol_indices)
            for i in range(n_sol):
                cov_without_i = 0
                for j in range(n_sol):
                    if j != i:
                        cov_without_i |= block_masks[sol_indices[j]]
                        
                if cov_without_i == full_mask:
                    sol_indices.pop(i)
                    improved = True
                    break
                    
        return [all_blocks[i] for i in sol_indices]

    # Initial deterministic run
    best_sol = optimize_solution(greedy_construct(42))
    best_len = len(best_sol)
    
    # Randomized restarts to find better local optima
    start_time = time.time()
    seed = 1
    while time.time() - start_time < 90: # Leave 30s buffer for safety
        seed += 1
        sol = greedy_construct(seed)
        sol_opt = optimize_solution(sol)
        if len(sol_opt) < best_len:
            best_len = len(sol_opt)
            best_sol = sol_opt
            
    return best_sol
