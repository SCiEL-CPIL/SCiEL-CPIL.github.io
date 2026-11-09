import itertools
import random

def construct(v, k, t):
    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # Generate all k-subsets
    k_subs = list(itertools.combinations(range(v), k))
    num_k_subs = len(k_subs)
    
    # For each block, store the list of t-subset indices it covers
    block_t_subs = []
    # For each t-subset, store the list of block indices that cover it
    blocks_covering = [[] for _ in range(num_t_subs)]
    
    for b_idx, block in enumerate(k_subs):
        t_indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            t_indices.append(idx)
            blocks_covering[idx].append(b_idx)
        block_t_subs.append(t_indices)
        
    # 2. Greedy Set Cover
    uncovered = set(range(num_t_subs))
    solution_blocks = []
    solution_indices = []
    
    # To speed up, we can shuffle the t-subsets or just pick randomly
    # Picking a random uncovered t-subset each time is a valid heuristic
    
    while uncovered:
        # Pick a random uncovered t-subset
        s = random.choice(list(uncovered))
        
        # Get all blocks covering s
        candidates = blocks_covering[s]
        
        best_block_idx = -1
        best_gain = -1
        
        # Evaluate gain for each candidate
        # Optimization: If a candidate is already in solution, skip? 
        # No, re-adding is not allowed, but we track solution_indices.
        # Actually, we don't need to skip if we just track which are in solution?
        # No, we just pick the best one. If it's already there, gain is 0 (or low).
        # But we should avoid picking a block already in solution if possible, 
        # although its gain for *newly* uncovered sets is 0.
        
        for b_idx in candidates:
            # Calculate gain: number of t-subs in this block that are in uncovered
            gain = 0
            for t_idx in block_t_subs[b_idx]:
                if t_idx in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            break # Should not happen
            
        # Add to solution
        solution_indices.append(best_block_idx)
        solution_blocks.append(list(k_subs[best_block_idx]))
        
        # Update uncovered
        for t_idx in block_t_subs[best_block_idx]:
            uncovered.discard(t_idx)
            
    # 3. Pruning: Remove redundant blocks
    # Calculate cover counts
    cover_count = [0] * num_t_subs
    for b_idx in solution_indices:
        for t_idx in block_t_subs[b_idx]:
            cover_count[t_idx] += 1
            
    changed = True
    while changed:
        changed = False
        for i in range(len(solution_indices)):
            b_idx = solution_indices[i]
            # Check if any t-subset in this block is uniquely covered by this block
            is_redundant = True
            for t_idx in block_t_subs[b_idx]:
                if cover_count[t_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block i
                for t_idx in block_t_subs[b_idx]:
                    cover_count[t_idx] -= 1
                solution_indices.pop(i)
                solution_blocks.pop(i)
                changed = True
                break # Restart loop
                
    return solution_blocks
