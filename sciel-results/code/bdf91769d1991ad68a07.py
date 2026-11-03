import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    masks = [0] * n_blocks
    # Also store the list of t-subset indices for each block for fast count updates
    block_t_indices = [None] * n_blocks
    for i, block in enumerate(k_subsets):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        masks[i] = mask
        block_t_indices[i] = indices
        
    # 4. Greedy Set Cover
    uncovered = all_mask
    selected_greedy = []
    used_blocks = [False] * n_blocks
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(n_blocks):
            if used_blocks[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        if best_idx == -1:
            break
        selected_greedy.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~masks[best_idx]
        
    current_best = len(selected_greedy)
    best_solution = selected_greedy[:]

    # Fast Redundancy Removal using Counts
    def remove_redundants(blocks_list):
        # Make a copy to modify
        blocks_list = blocks_list[:]
        if not blocks_list:
            return blocks_list
            
        # Initialize coverage counts
        cover_counts = [0] * n_t_subsets
        for b in blocks_list:
            for idx in block_t_indices[b]:
                cover_counts[idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks_list) - 1, -1, -1):
                b = blocks_list[i]
                redundant = True
                for idx in block_t_indices[b]:
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block b
                    for idx in block_t_indices[b]:
                        cover_counts[idx] -= 1
                    blocks_list.pop(i)
                    changed = True
                    break # Restart scan from beginning? Or continue? 
                    # Restarting is safer but slower. Continuing is fine if we just removed one.
                    # Actually, removing one might make another redundant.
                    # Breaking and restarting the while loop is O(N^2) worst case.
                    # Let's just break the inner for and let the while loop handle it.
        return blocks_list

    best_solution = remove_redundants(best_solution)
    current_best = len(best_solution)

    # Local Search: 1-Swap (Replace one block with another)
    # Try to find a block B not in solution that can replace A in solution
    # such that the set remains valid, and potentially allows further reduction.
    # Simpler: Try to ADD a block, then REMOVE redundants. If size decreases, keep.
    # This is "1-Add".
    
    start_time = time.time()
    used_in_solution = [False] * n_blocks
    for b in best_solution:
        used_in_solution[b] = True
        
    # Iterate a few times
    for iteration in range(20):
        if time.time() - start_time > 60:
            break
            
        improved = False
        # Try adding each unused block
        # To speed up, we can sample or iterate all.
        # 75k blocks. Adding one and checking redundancy is O(C(k,t) + N_solution * C(k,t)).
        # This is feasible.
        
        for b in range(n_blocks):
            if used_in_solution[b]:
                continue
                
            # Add b
            best_solution.append(b)
            used_in_solution[b] = True
            
            # Remove redundants
            new_solution = remove_redundants(best_solution)
            
            if len(new_solution) < len(best_solution):
                # Improvement
                best_solution = new_solution
                # Update used_in_solution
                used_in_solution = [False] * n_blocks
                for nb in best_solution:
                    used_in_solution[nb] = True
                improved = True
                # Restart search? Or continue?
                # Restarting is better to find more improvements.
                break
            else:
                # Rollback
                best_solution.pop()
                used_in_solution[b] = False
                
        if not improved:
            break

    return [list(k_subsets[i]) for i in best_solution]
