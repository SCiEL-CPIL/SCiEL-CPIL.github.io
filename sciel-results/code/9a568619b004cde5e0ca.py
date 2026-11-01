import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Map t-subsets to IDs
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_map = {sub: i for i, sub in enumerate(t_subs)}
    num_t = len(t_subs)
    
    # 2. Pre-compute block -> t-subset IDs
    k_subs = list(itertools.combinations(range(v), k))
    num_k = len(k_subs)
    
    t_in_k = list(itertools.combinations(range(k), t))
    num_t_in_k = len(t_in_k)
    
    block_t_ids = np.zeros((num_k, num_t_in_k), dtype=np.int32)
    
    for i, block in enumerate(k_subs):
        for j, t_indices in enumerate(t_in_k):
            global_t_sub = tuple(sorted(block[x] for x in t_indices))
            block_t_ids[i, j] = t_sub_map[global_t_sub]
            
    # 3. Greedy Cover to get initial solution
    covered = np.zeros(num_t, dtype=bool)
    used_blocks = set()
    solution = []
    
    while not covered.all():
        if time.time() > deadline - 30.0:
            break
            
        G = np.sum(~covered[block_t_ids], axis=1)
        G[list(used_blocks)] = -1
        best_idx = np.argmax(G)
        
        if G[best_idx] <= 0:
            break
            
        used_blocks.add(best_idx)
        solution.append(list(k_subs[best_idx]))
        covered[block_t_ids[best_idx]] = True
        
    # Build index mapping
    block_to_idx = {block: i for i, block in enumerate(k_subs)}
    
    # 4. Local Search: Remove redundant blocks
    def remove_redundant(sol):
        if not sol:
            return sol, False
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in sol:
            idx = block_to_idx[tuple(b)]
            cover_count[block_t_ids[idx]] += 1
            
        new_sol = []
        changed = False
        for b in sol:
            idx = block_to_idx[tuple(b)]
            min_count = np.min(cover_count[block_t_ids[idx]])
            if min_count > 1:
                changed = True
                cover_count[block_t_ids[idx]] -= 1
            else:
                new_sol.append(b)
        return new_sol, changed
    
    for _ in range(10):
        if time.time() > deadline - 30.0:
            break
        solution, changed = remove_redundant(solution)
        if not changed:
            break
    
    # 5. WalkSAT-style local search
    # State: which blocks are included
    included = np.zeros(num_k, dtype=bool)
    for b in solution:
        included[block_to_idx[tuple(b)]] = True
    
    # Coverage count
    cover_count = np.zeros(num_t, dtype=np.int32)
    for i in range(num_k):
        if included[i]:
            cover_count[block_t_ids[i]] += 1
    
    num_uncovered = np.sum(cover_count == 0)
    best_num_uncovered = num_uncovered
    best_included = included.copy()
    
    # WalkSAT parameters
    noise = 0.3  # probability of random flip
    max_steps = 5000
    
    step = 0
    while step < max_steps and time.time() < deadline - 10.0:
        step += 1
        
        # Pick a block to flip
        if random.random() < noise:
            # Random flip
            i = random.randint(0, num_k - 1)
        else:
            # Pick block that maximizes improvement
            # For each block, compute delta in uncovered count if flipped
            # If included: removing it may increase uncovered
            # If not included: adding it may decrease uncovered
            
            # Vectorized approach
            # For included blocks: delta = number of t-subsets that would become uncovered
            #   = number of t-subsets in block where cover_count == 1
            # For not included: delta = -number of t-subsets that would become covered
            #   = -number of t-subsets in block where cover_count == 0
            
            delta = np.zeros(num_k, dtype=np.int32)
            
            # For included blocks
            inc_mask = included
            if np.any(inc_mask):
                # For each included block, count t-subsets with cover_count == 1
                inc_block_tids = block_t_ids[inc_mask]
                # cover_count == 1 for these t-subsets
                delta[inc_mask] = np.sum(cover_count[inc_block_tids] == 1, axis=1)
            
            # For not included blocks
            not_inc_mask = ~included
            if np.any(not_inc_mask):
                not_inc_block_tids = block_t_ids[not_inc_mask]
                # cover_count == 0 for these t-subsets (would become covered)
                delta[not_inc_mask] = -np.sum(cover_count[not_inc_block_tids] == 0, axis=1)
            
            # Pick best delta (most negative = best improvement)
            best_i = np.argmin(delta)
            i = int(best_i)
        
        # Flip block i
        if included[i]:
            # Remove
            cover_count[block_t_ids[i]] -= 1
            included[i] = False
        else:
            # Add
            cover_count[block_t_ids[i]] += 1
            included[i] = True
        
        # Update uncovered count
        num_uncovered = np.sum(cover_count == 0)
        
        if num_uncovered < best_num_uncovered:
            best_num_uncovered = num_uncovered
            best_included = included.copy()
            if best_num_uncovered == 0:
                break
        
        # Accept if improvement or with some probability (simulated annealing style)
        # For simplicity, always accept (this is more like WalkSAT without restarts)
        # But we should sometimes backtrack. Let's use a simple strategy:
        # If no improvement for many steps, reset to best and try different noise
    
    # Build solution from best_included
    best_solution = []
    for i in range(num_k):
        if best_included[i]:
            best_solution.append(list(k_subs[i]))
    
    # 6. Final redundancy removal
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        best_solution, changed = remove_redundant(best_solution)
        if not changed:
            break
    
    # Use the better of greedy and walksat solutions
    if len(best_solution) < len(solution):
        solution = best_solution
    
    return solution
