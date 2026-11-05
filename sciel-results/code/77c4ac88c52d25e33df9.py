import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    # Use 115s as hard limit to ensure return
    deadline = start_time + 115.0
    
    # 1. Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # block_subsets[i] = list of t-subset indices covered by block i
    block_subsets = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(subset_to_idx[s])
        block_subsets.append(subs)
        
    # subset_to_blocks[s_idx] = list of block indices covering subset s_idx
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, subs in enumerate(block_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)
            
    best_solution = None
    best_len = float('inf')
    
    # Randomized Greedy with Local Search
    # We run multiple instances.
    
    for run in range(200):
        if time.time() > deadline - 20.0:
            break
            
        # Random seed for tie-breaking
        random.seed(run)
        
        coverage_count = [0] * num_t_subsets
        gain = [len(subs) for subs in block_subsets]
        used_blocks = [False] * num_blocks
        
        solution = []
        
        while True:
            # Find max gain
            max_gain = -1
            candidates = []
            
            # Optimization: Skip if time is tight
            if time.time() > deadline - 5.0:
                break
                
            for b_idx in range(num_blocks):
                if not used_blocks[b_idx]:
                    g = gain[b_idx]
                    if g > max_gain:
                        max_gain = g
                        candidates = [b_idx]
                    elif g == max_gain:
                        candidates.append(b_idx)
                        
            if max_gain <= 0:
                break
                
            # Pick one from candidates
            chosen = random.choice(candidates)
            
            used_blocks[chosen] = True
            solution.append(all_blocks[chosen])
            
            # Update coverage and gains
            for s_idx in block_subsets[chosen]:
                if coverage_count[s_idx] == 0:
                    for b_idx in subset_to_blocks[s_idx]:
                        if not used_blocks[b_idx]:
                            gain[b_idx] -= 1
                coverage_count[s_idx] += 1
                
        # Check if complete
        if all(c > 0 for c in coverage_count):
            if len(solution) < best_len:
                best_len = len(solution)
                best_solution = solution
                
    if best_solution is None:
        # Fallback: Return a valid covering if we found one, else error?
        # The loop should always find one.
        # If not, we might have timed out in the middle.
        # We need to return *something* valid.
        # If we didn't complete, we can't return an invalid list.
        # But the greedy algorithm is very fast, so it should complete at least once.
        pass

    # Local Search Pruning on the best solution
    if best_solution is not None:
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        final_block_indices = [block_to_idx[b] for b in best_solution]
        
        # Recalculate coverage
        final_coverage = [0] * num_t_subsets
        for idx in final_block_indices:
            for s_idx in block_subsets[idx]:
                final_coverage[s_idx] += 1
                
        improved = True
        while improved:
            improved = False
            for i in range(len(final_block_indices)):
                if time.time() > deadline - 5.0:
                    break
                idx = final_block_indices[i]
                # Temporarily remove
                for s_idx in block_subsets[idx]:
                    final_coverage[s_idx] -= 1
                    
                if all(c > 0 for c in final_coverage):
                    final_block_indices.pop(i)
                    improved = True
                    break
                else:
                    for s_idx in block_subsets[idx]:
                        final_coverage[s_idx] += 1

        final_blocks = [list(all_blocks[idx]) for idx in final_block_indices]
        if len(final_blocks) < len(best_solution):
            best_solution = final_blocks

    return best_solution
