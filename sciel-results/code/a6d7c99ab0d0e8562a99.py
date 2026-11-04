import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets and map to IDs
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the list of triplet IDs it covers
    # This is expensive memory-wise if stored as lists, but 346k * 35 ints is ~12M ints, which is ~48MB. Acceptable.
    # We can store them as numpy arrays for faster intersection?
    # Actually, for greedy set cover, we just need to know which triplets are covered.
    
    block_triplets = []
    for block in all_blocks:
        t_ids = [triplet_to_id[tup] for tup in itertools.combinations(block, t)]
        block_triplets.append(np.array(t_ids, dtype=np.int32))
        
    # Greedy Set Cover
    # uncovered: boolean array of size n_triplets, True if not yet covered
    uncovered = np.ones(n_triplets, dtype=bool)
    solution_blocks = []
    
    # To speed up greedy, we can maintain counts of how many times each triplet is covered by *candidate* blocks?
    # No, standard greedy: pick block with max (block_triplets & uncovered).sum()
    
    # Optimization: Instead of checking all 346k blocks every time (which is 346k * 35 ops * ~70 steps = ~800M ops, might be slow in Python),
    # we can use a smarter approach or just hope numpy vectorization helps.
    # 346k blocks * 35 triplets = 12M entries.
    # If we stack them into a 2D array (346k x 35), we can do matrix operations?
    # But the lengths are fixed (35).
    
    # Let's try a slightly different greedy: 
    # Pick a random uncovered triplet.
    # Find all blocks that contain it.
    # Among those, pick the one that covers the most *other* uncovered triplets.
    # This is "Greedy by Element" which is often faster.
    
    # Precompute: for each triplet, which blocks contain it?
    # triplet_to_blocks: list of lists of block indices
    triplet_to_blocks = [[] for _ in range(n_triplets)]
    for b_idx, t_ids in enumerate(block_triplets):
        for tid in t_ids:
            triplet_to_blocks[tid].append(b_idx)
            
    # Now Greedy:
    while uncovered.any():
        if time.time() > deadline - 10:
            break
            
        # Pick a random uncovered triplet
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        target_tid = unc_idx[random.randint(0, len(unc_idx) - 1)]
        
        # Candidates are blocks containing target_tid
        candidates = triplet_to_blocks[target_tid]
        
        if not candidates:
            # This shouldn't happen if v,k,t are valid
            break
            
        # Score each candidate: number of uncovered triplets it covers
        best_block_idx = -1
        best_score = -1
        
        for b_idx in candidates:
            # Count how many triplets in this block are still uncovered
            # block_triplets[b_idx] is an array of tids
            # We need sum(uncovered[block_triplets[b_idx]])
            score = np.sum(uncovered[block_triplets[b_idx]])
            if score > best_score:
                best_score = score
                best_block_idx = b_idx
                if best_score == len(block_triplets[b_idx]):
                    break # Perfect score for this block size
            
        if best_block_idx != -1:
            solution_blocks.append(all_blocks[best_block_idx])
            # Update uncovered
            uncovered[block_triplets[best_block_idx]] = False
            
    # Post-processing: Remove redundant blocks
    # A block is redundant if all its triplets are covered by other blocks in the solution.
    # We need to count coverage from the *other* blocks.
    
    # Count how many times each triplet is covered by the current solution
    coverage_count = np.zeros(n_triplets, dtype=np.int32)
    for block in solution_blocks:
        for tid in block_triplets[all_blocks.index(block)]: # This is slow, O(N^2)
            coverage_count[tid] += 1
            
    # Better: Re-iterate using indices
    # Let's map solution blocks to their indices in all_blocks
    sol_block_indices = []
    for block in solution_blocks:
        # Find index in all_blocks
        # Since all_blocks is a list of tuples, we can use a dict for lookup
        pass
        
    # Let's just do a simple iterative removal
    # We need a fast way to get triplet ids for a block in the solution.
    # Let's keep the indices.
    
    sol_indices = []
    for block in solution_blocks:
        # We need to find the index. 
        # Instead of searching, let's store indices during construction.
        pass
        
    # Let's refactor the greedy loop to store indices.
    
    # RESTART with cleaner implementation
    
    # Clear previous attempts
    solution_indices = []
    uncovered = np.ones(n_triplets, dtype=bool)
    
    while uncovered.any():
        if time.time() > deadline - 10:
            break
            
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        target_tid = unc_idx[random.randint(0, len(unc_idx) - 1)]
        
        candidates = triplet_to_blocks[target_tid]
        if not candidates:
            break
            
        best_block_idx = -1
        best_score = -1
        
        for b_idx in candidates:
            score = np.sum(uncovered[block_triplets[b_idx]])
            if score > best_score:
                best_score = score
                best_block_idx = b_idx
                if best_score == len(block_triplets[b_idx]):
                    break
            
        if best_block_idx != -1:
            solution_indices.append(best_block_idx)
            uncovered[block_triplets[best_block_idx]] = False
            
    # Convert indices to blocks
    final_blocks = [all_blocks[i] for i in solution_indices]
    
    # Redundancy Removal
    # Count coverage
    coverage_count = np.zeros(n_triplets, dtype=np.int32)
    for idx in solution_indices:
        coverage_count[block_triplets[idx]] += 1
        
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        new_indices = []
        for idx in solution_indices:
            # Check if any triplet in this block has coverage_count == 1
            # If yes, this block is essential.
            # If no, it's redundant.
            t_ids = block_triplets[idx]
            if np.any(coverage_count[t_ids] == 1):
                new_indices.append(idx)
            else:
                # Remove it
                coverage_count[t_ids] -= 1
                changed = True
        solution_indices = new_indices
        
    final_blocks = [all_blocks[i] for i in solution_indices]
    
    # Verify coverage just in case
    covered = set()
    for block in final_blocks:
        for triplet in itertools.combinations(block, t):
            covered.add(triplet)
            
    if len(covered) < n_triplets:
        # Fallback: If something went wrong, try a simple greedy again without the fancy selection?
        # Or just return what we have, hoping the evaluator is lenient? No, it must be valid.
        # Let's add a simple repair: add blocks for missing triplets.
        missing = set(all_triplets) - covered
        for triplet in missing:
            # Find a block containing this triplet
            # We can just pick the first one from triplet_to_blocks
            b_idx = triplet_to_blocks[triplet_to_id[triplet]][0]
            final_blocks.append(all_blocks[b_idx])
            
    return final_blocks

if __name__ == "__main__":
    pass
