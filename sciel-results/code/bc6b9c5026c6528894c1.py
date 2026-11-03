import itertools
import random
import time

# Global cache for precomputation
_PRECOMP_CACHE = {}

def _precompute(v, k, t):
    """
    Precomputes the incidence structure.
    """
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    block_combinations = list(itertools.combinations(range(v), k))
    num_blocks = len(block_combinations)
    
    block_to_tsubs = [[] for _ in range(num_blocks)]
    tsub_to_blocks = [[] for _ in range(num_t)]
    
    for i, block in enumerate(block_combinations):
        for ts in itertools.combinations(block, t):
            j = t_to_idx[ts]
            block_to_tsubs[i].append(j)
            tsub_to_blocks[j].append(i)
            
    # Precompute set of tsub indices for each block for fast intersection?
    # block_to_tsubs_sets = [set(lst) for lst in block_to_tsubs]
    
    return t_subsets, block_combinations, block_to_tsubs, tsub_to_blocks, num_t, num_blocks

def construct(v, k, t):
    key = (v, k, t)
    if key not in _PRECOMP_CACHE:
        _PRECOMP_CACHE[key] = _precompute(v, k, t)
    
    t_subsets, block_combinations, block_to_tsubs, tsub_to_blocks, num_t, num_blocks = _PRECOMP_CACHE[key]
    
    # Fallback for large problems
    if num_blocks > 100000:
        return _simple_greedy(v, k, t)

    best_sol = None
    best_len = float('inf')
    
    num_trials = 100 # Increase trials for better local search
    
    for trial in range(num_trials):
        random.seed(1000 + trial)
        
        # 1. Randomized Greedy Construction
        uncovered = [True] * num_t
        num_uncovered = num_t
        selected_blocks = []
        
        while num_uncovered > 0:
            # Find max coverage
            max_cover = -1
            best_blocks = []
            
            for i in range(num_blocks):
                cover = 0
                for j in block_to_tsubs[i]:
                    if uncovered[j]:
                        cover += 1
                if cover > max_cover:
                    max_cover = cover
                    best_blocks = [i]
                elif cover == max_cover:
                    best_blocks.append(i)
            
            if max_cover == 0:
                break # Should not happen if valid input
                
            # Pick randomly from best_blocks
            chosen_block = random.choice(best_blocks)
            selected_blocks.append(chosen_block)
            
            # Update uncovered
            for j in block_to_tsubs[chosen_block]:
                if uncovered[j]:
                    uncovered[j] = False
                    num_uncovered -= 1
        
        # 2. Local Search: Redundancy Removal
        # Calculate frequency
        freq = [0] * num_t
        for i in selected_blocks:
            for j in block_to_tsubs[i]:
                freq[j] += 1
                
        new_sol = []
        for i in selected_blocks:
            is_redundant = True
            for j in block_to_tsubs[i]:
                if freq[j] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                new_sol.append(i)
                
        # 3. Local Search: 1-Swap
        # Try to replace a block with another to reduce size or maintain size but allow further reductions?
        # Actually, just try to remove blocks if possible after swaps.
        # Let's do a simple 1-swap: 
        # For each block A in new_sol:
        #   Find exclusive tsubs of A (freq==1)
        #   If empty, remove A (already done in redundancy step, but freq changes)
        #   Else, find block B not in new_sol that covers all exclusive tsubs of A.
        #   If found, swap A with B.
        
        # To make it effective, we iterate until no change.
        improved = True
        while improved:
            improved = False
            # Recalculate freq for current new_sol
            freq = [0] * num_t
            for i in new_sol:
                for j in block_to_tsubs[i]:
                    freq[j] += 1
            
            for idx, A in enumerate(new_sol):
                exclusive_A = [j for j in block_to_tsubs[A] if freq[j] == 1]
                
                if not exclusive_A:
                    # Can remove A
                    new_sol.pop(idx)
                    improved = True
                    break # Restart loop
                    
                # Try to find B to replace A
                # B must cover all j in exclusive_A
                # To speed up, check if B covers exclusive_A[0]
                candidates = tsub_to_blocks[exclusive_A[0]]
                
                for B in candidates:
                    if B in new_sol:
                        continue
                    
                    # Check if B covers all j in exclusive_A
                    ok = True
                    for j in exclusive_A:
                        # Check if j is in B
                        # We can check if j is in block_to_tsubs[B]
                        # Since exclusive_A is small, this is fast
                        if j not in block_to_tsubs[B]:
                            ok = False
                            break
                            
                    if ok:
                        # Perform swap
                        # Remove A, Add B
                        new_sol[idx] = B
                        improved = True
                        break # Restart loop after swap
                
                if improved:
                    break
                    
        if len(new_sol) < best_len:
            best_len = len(new_sol)
            best_sol = new_sol.copy()
            
    if best_sol is None:
        return []
        
    result_blocks = []
    for b_idx in best_sol:
        result_blocks.append(list(block_combinations[b_idx]))
        
    return result_blocks

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        best_block = None
        max_cover = -1
        for block in itertools.combinations(range(v), k):
            cover = 0
            for ts in itertools.combinations(block, t):
                if ts in uncovered:
                    cover += 1
            if cover > max_cover:
                max_cover = cover
                best_block = block
        if best_block is None:
            break
        blocks.append(list(best_block))
        for ts in itertools.combinations(best_block, t):
            uncovered.discard(ts)
    return blocks
