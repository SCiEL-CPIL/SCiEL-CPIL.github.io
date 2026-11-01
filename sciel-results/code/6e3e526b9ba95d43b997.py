import itertools
import random
def construct(v, k, t):
    import time
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    t_combos = list(itertools.combinations(range(v), t))
    t_index = {c: i for i, c in enumerate(t_combos)}
    n_t = len(t_combos)
    
    coverage = [0] * n_t
    blocks = []
    
    # 1. Greedy Construction
    # We need to know which t-subsets are uncovered.
    # uncovered_count = n_t
    # We pick the block that maximizes the number of newly covered t-subsets.
    
    # To speed up greedy, we can precompute the list of t-subset indices for every possible k-subset?
    # 75582 blocks * 56 subsets = 4.2M entries.
    # This is feasible in memory and time (a few seconds in Python with NumPy or optimized lists).
    # Let's try to precompute.
    
    k_combos = list(itertools.combinations(range(v), k))
    # Map each k-combo to its t-subset indices
    # This might be slow in pure python.
    # Alternative: Greedy step-by-step.
    
    # Step-by-step Greedy is O(Iterations * k * v * C(k, t-1))
    # Iterations ~ 300.
    # 300 * 8 * 19 * 56 ~ 2.5M operations. Very fast.
    
    while True:
        if time.time() > deadline: break
        # Find an uncovered t-subset
        # To make it faster, we can just check if coverage[i] == 0
        # But finding the "best" block to cover the most uncovered is harder.
        # Simple Greedy: Pick first uncovered, build block around it.
        
        # Find first uncovered
        first_uncov = -1
        for i in range(n_t):
            if coverage[i] == 0:
                first_uncov = i
                break
        if first_uncov == -1:
            break # Covered all
            
        seed = t_combos[first_uncov]
        block = list(seed)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            # Points to try
            # We can filter points that are not in block
            for p in range(v):
                if p in block: continue
                gain = 0
                # Subsets involving p and t-1 from block
                for comb in itertools.combinations(block, t-1):
                    s = tuple(sorted(comb + (p,)))
                    idx = t_index[s]
                    if coverage[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            block.append(best_point)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update coverage
        for s in itertools.combinations(block, t):
            coverage[t_index[s]] += 1
            
    # 2. Local Search
    # Try to remove redundant blocks or swap to enable removals.
    
    # Helper to get t-subset indices for a block
    def get_block_indices(block):
        return [t_index[s] for s in itertools.combinations(block, t)]
        
    # Initial indices
    block_indices = [get_block_indices(b) for b in blocks]
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to remove redundant blocks
        for i in range(len(blocks)):
            if time.time() > deadline: break
            # Check if block i is redundant
            redundant = True
            for idx in block_indices[i]:
                if coverage[idx] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for idx in block_indices[i]:
                    coverage[idx] -= 1
                blocks.pop(i)
                block_indices.pop(i)
                improved = True
                break # Restart from beginning as indices changed
                
        if improved:
            continue
            
        # Try swaps
        # For each block, try to find a replacement that covers its unique parts
        # and is different, hoping to create redundancy elsewhere.
        
        # To avoid getting stuck in a loop, we can limit swaps or use randomization.
        # Let's try to swap block i with a random valid replacement.
        
        # We need to identify unique parts of block i.
        # Unique parts are indices where coverage == 1.
        
        # We will iterate through blocks again.
        for i in range(len(blocks)):
            if time.time() > deadline: break
            unique_indices = [idx for idx in block_indices[i] if coverage[idx] == 1]
            
            if not unique_indices:
                # Should have been removed in previous step, but maybe coverage changed?
                # Actually, if not redundant, unique_indices is not empty.
                pass
                
            # We need a new block B' such that unique_indices are covered by B'.
            # This means B' must contain the union of the t-subsets corresponding to unique_indices?
            # No, B' must contain each t-subset that is uniquely covered by B.
            # Let U be the set of points in these unique t-subsets.
            # B' must contain U.
            
            # Calculate U
            U = set()
            for idx in unique_indices:
                U.update(t_combos[idx])
                
            # If |U| > k, no replacement possible (shouldn't happen if unique t-subsets are in B)
            # Since unique t-subsets are subsets of B, U is subset of B.
            
            # Find a random k-subset B' containing U.
            # We can sample.
            # Number of such subsets: C(v - |U|, k - |U|)
            
            # Try a few random swaps
            for _ in range(10): # Limit attempts per block
                if time.time() > deadline: break
                
                # Construct B'
                # Pick k - |U| random points from v \ U
                remaining_points = [p for p in range(v) if p not in U]
                if len(remaining_points) < k - len(U):
                    continue
                extra = random.sample(remaining_points, k - len(U))
                new_block = tuple(sorted(list(U) + extra))
                
                # Check if new_block is already in blocks
                # To speed up, keep a set of blocks
                # We don't have a set, let's check linearly or maintain one.
                # For now, linear check is O(N). N~300. 300*10*300 = 90k. OK.
                
                if new_block in blocks:
                    continue
                    
                # Check if new_block covers unique_indices
                # It must, by construction (U is in new_block, and unique_indices are subsets of U? No.)
                # Wait. U is the union of points in unique t-subsets.
                # If B' contains U, does it contain the t-subsets?
                # Yes, if a t-subset S is subset of U, and B' contains U, then S is subset of B'.
                
                # Perform Swap
                # Remove i
                for idx in block_indices[i]:
                    coverage[idx] -= 1
                    
                # Add new_block
                new_indices = get_block_indices(new_block)
                for idx in new_indices:
                    coverage[idx] += 1
                    
                # Replace
                blocks[i] = new_block
                block_indices[i] = new_indices
                
                improved = True
                break # Restart loop to check for new redundancies
                
            if improved:
                break
                
    return [list(b) for b in blocks]
