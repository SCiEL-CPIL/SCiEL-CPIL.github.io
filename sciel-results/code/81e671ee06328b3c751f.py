import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to its index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(all_k_blocks)
    
    # Precompute for each k-block, the list of t-subset indices it covers
    block_t_indices = []
    for b in all_k_blocks:
        covered = []
        for s in itertools.combinations(b, t):
            covered.append(t_subset_to_idx[s])
        block_t_indices.append(covered)
        
    # Precompute index: for each t-subset, which k-blocks cover it?
    t_to_blocks = [[] for _ in range(n_t_subsets)]
    for i, covered in enumerate(block_t_indices):
        for j in covered:
            t_to_blocks[j].append(i)

    best_solution = None
    best_size = float('inf')
    
    # Run multiple random greedy constructions
    while True:
        if time.time() - start_time > timeout:
            break
            
        # Randomized Greedy Construction
        # counts[i] is the number of blocks covering t-subset i
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        solution = []
        
        # We need to know how many are uncovered.
        # Initially n_t_subsets.
        # We can track uncovered_count.
        uncovered_count = n_t_subsets
        
        while uncovered_count > 0:
            # Pick a random uncovered t-subset by probing
            # This is efficient when many are uncovered.
            # When few are uncovered, it might take a few tries.
            while True:
                target_t_idx = random.randint(0, n_t_subsets - 1)
                if counts[target_t_idx] == 0:
                    break
            
            # Get candidate blocks covering this t-subset
            candidates = t_to_blocks[target_t_idx]
            
            if not candidates:
                # Should not happen
                break
            
            # Evaluate candidates based on how many other uncovered subsets they cover
            # Gain = number of t_idx in block where counts[t_idx] == 0
            best_block_idx = -1
            max_gain = -1
            top_candidates = []
            
            for b_idx in candidates:
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if counts[t_idx] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    top_candidates = [b_idx]
                elif gain == max_gain:
                    top_candidates.append(b_idx)
            
            # Pick randomly from top candidates
            chosen_block_idx = random.choice(top_candidates)
            
            solution.append(chosen_block_idx)
            
            # Update counts and uncovered_count
            for t_idx in block_t_indices[chosen_block_idx]:
                if counts[t_idx] == 0:
                    uncovered_count -= 1
                counts[t_idx] += 1
        
        # Cleanup: Remove redundant blocks
        # A block is redundant if all its t-subsets are covered by at least one other block
        # i.e., for all s in block, counts[s] > 1
        changed = True
        while changed:
            changed = False
            sol_indices = list(range(len(solution)))
            random.shuffle(sol_indices)
            
            for idx in sol_indices:
                b_idx = solution[idx]
                # Check if redundant
                redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if counts[t_idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    for t_idx in block_t_indices[b_idx]:
                        counts[t_idx] -= 1
                    solution.pop(idx)
                    changed = True
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution[:]
            
    if best_solution is None:
        return []
        
    # Convert indices back to actual blocks
    final_blocks = [list(all_k_blocks[i]) for i in best_solution]
    return final_blocks
