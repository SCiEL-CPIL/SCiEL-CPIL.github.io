import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute blocks and their subsets
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # For each block, list of subset indices it covers
    block_to_subsets = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(subset_to_idx[s])
        block_to_subsets.append(subs)
        
    # For each subset, list of block indices that cover it
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)
    
    # Convert to numpy arrays for faster operations
    block_to_subsets_np = np.array(block_to_subsets, dtype=np.int32)
    # For each block, we need to know which subsets it covers
    # We'll create a sparse-like structure or use the list form for updates
    # Actually, for the greedy step, we need to count how many uncovered subsets each block covers.
    # Let's precompute for each block the list of subsets it covers as a numpy array.
    # block_to_subsets is already a list of lists.
            
    best_solution = None
    best_len = float('inf')
    
    def greedy_randomized():
        # Returns a list of block indices
        coverage = np.zeros(num_subsets, dtype=np.int32)
        solution = []
        uncovered_count = num_subsets
        
        while uncovered_count > 0:
            if time.time() > deadline:
                return solution if len(solution) < best_len else None
                
            # Calculate score for each block: number of currently uncovered subsets it covers
            scores = np.zeros(num_blocks, dtype=np.int32)
            
            # For each uncovered subset, increment score of its blocks.
            # Vectorized approach: 
            # uncovered_mask = (coverage == 0)
            # This is hard to vectorize directly with the sparse structure unless we use a matrix.
            # Let's stick to the loop but optimize it slightly.
            # Or, we can precompute a block-subset incidence matrix? Too big (108157 x 3876).
            
            # Alternative: 
            # For each block, count how many of its subsets are uncovered.
            # This requires iterating over all blocks and their subsets.
            # num_blocks * C(7,4) = 108157 * 35 = 3.7M operations.
            # This is faster than 1.7M if we do it in C via numpy?
            # Let's try:
            # scores = np.sum(coverage[block_to_subsets_np] == 0, axis=1)
            # This creates a temporary array of size num_blocks * 35.
            # 108157 * 35 = 3.7M integers. This is feasible.
            
            # Let's test this numpy approach.
            # block_to_subsets_np is shape (num_blocks, 35)
            # coverage is shape (num_subsets,)
            # coverage[block_to_subsets_np] is shape (num_blocks, 35)
            # (coverage[block_to_subsets_np] == 0).sum(axis=1) gives scores.
            
            scores = np.sum(coverage[block_to_subsets_np] == 0, axis=1)
            
            # Find max score
            max_score = np.max(scores)
            if max_score == 0:
                break # Should not happen if uncovered > 0, but just in case
            
            # Randomize tie-breaking
            candidates = np.where(scores == max_score)[0]
            best_b = random.choice(candidates)
            
            solution.append(best_b)
            
            # Update coverage
            # coverage[block_to_subsets_np[best_b]] += 1
            # But we need to update uncovered_count.
            subs_covered = block_to_subsets_np[best_b]
            newly_covered = np.sum(coverage[subs_covered] == 0)
            uncovered_count -= newly_covered
            coverage[subs_covered] += 1
            
        return solution

    def local_search(solution):
        # solution is a list of block indices
        # Returns improved solution
        coverage = np.zeros(num_subsets, dtype=np.int32)
        for b in solution:
            coverage[block_to_subsets_np[b]] += 1
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            n = len(solution)
            for i in range(n):
                if time.time() > deadline:
                    break
                    
                b = solution[i]
                # Check if redundant
                # critical_subs are those with coverage == 1 that are covered by b
                subs_b = block_to_subsets_np[b]
                critical_mask = (coverage[subs_b] == 1)
                num_critical = np.sum(critical_mask)
                
                if num_critical == 0:
                    # Redundant, remove
                    coverage[subs_b] -= 1
                    solution.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                    
                else:
                    # Critical, try to find a replacement
                    # Find blocks that cover ALL critical subs
                    critical_subs = subs_b[critical_mask]
                    
                    # Intersection of subset_to_blocks[s] for s in critical_subs
                    # We can do this by counting how many critical subs each block covers
                    # and checking if it covers all of them.
                    # This is O(num_blocks * len(critical_subs)) which is 108157 * num_critical.
                    # num_critical is at most 35.
                    
                    # For each block, count how many of the critical_subs it covers.
                    # We can use numpy for this too.
                    # But subset_to_blocks is a list of lists.
                    # Let's stick to the set intersection for now, but maybe optimize if needed.
                    
                    candidates = set(subset_to_blocks[critical_subs[0]])
                    for s in critical_subs[1:]:
                        candidates.intersection_update(subset_to_blocks[s])
                        if not candidates:
                            break
                            
                    if candidates:
                        sol_set = set(solution)
                        valid_candidates = [c for c in candidates if c not in sol_set]
                        
                        if valid_candidates:
                            # Pick one that maximizes additional coverage?
                            # Or just random.
                            new_b = random.choice(valid_candidates)
                            
                            # Perform swap
                            coverage[subs_b] -= 1
                            coverage[block_to_subsets_np[new_b]] += 1
                            
                            solution[i] = new_b
                            improved = True
                            break # Restart loop to catch new redundancies
                            
        return solution

    # Run multiple times
    # Increase iterations if time permits
    for run in range(50):
        if time.time() > deadline - 5: # Leave 5s for finalization
            break
            
        sol = greedy_randomized()
        if sol is None:
            continue
            
        sol = local_search(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol.copy()
            
    if best_solution is None:
        # Fallback: just return a valid covering, even if large
        # Greedy should always produce one.
        sol = greedy_randomized()
        return [list(all_blocks[b]) for b in sol]
        
    return [list(all_blocks[b]) for b in best_solution]

if __name__ == "__main__":
    # Test
    blocks = construct(19, 7, 4)
    print(len(blocks))
