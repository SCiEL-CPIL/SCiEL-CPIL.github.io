import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit
    
    # Precompute
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Precompute block indices
    # Use a faster way to fill the array?
    # We can precompute the t-subsets of a generic k-tuple of indices?
    # No, the values matter for the lookup.
    
    # Let's stick to the loop, it's only 1.7M iterations.
    block_indices = np.empty((num_blocks, len(list(itertools.combinations(range(k), t)))), dtype=np.int32)
    
    for i, block in enumerate(k_subsets):
        count = 0
        for ts in itertools.combinations(block, t):
            block_indices[i, count] = t_subset_to_idx[ts]
            count += 1

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy attempts
    # We can estimate how many attempts we can afford.
    # One attempt takes ~0.5s.
    # We can do ~20 attempts.
    
    for run in range(20):
        if time.time() > deadline - 5.0:
            break
            
        uncovered = np.ones(num_t_subsets, dtype=bool)
        blocks = []
        
        while True:
            if time.time() > deadline - 2.0:
                break
                
            counts = uncovered[block_indices].sum(axis=1)
            
            if counts.max() == 0:
                break
                
            # Randomized selection: pick from top K
            # Sort indices by count descending
            # np.argsort is O(N log N). N=50k. This is fast.
            # But we can just find the max value and then pick randomly from those with that value?
            # Or pick randomly from top 50.
            
            # To be efficient, let's just pick the argmax for the first few runs,
            # and randomized for others?
            # Or always randomized from top 10.
            
            # Let's try picking randomly from the top 10 blocks.
            # Get top 10 indices
            # np.argpartition is O(N).
            top_k = 10
            if num_blocks > top_k:
                # Get indices of the top_k largest elements
                # partition returns indices such that the element at index top_k-1 is the top_k-th largest
                top_indices = np.argpartition(counts, -top_k)[-top_k:]
                # Pick one randomly
                best_idx = np.random.choice(top_indices)
            else:
                best_idx = np.argmax(counts)
                
            blocks.append(list(k_subsets[best_idx]))
            uncovered[block_indices[best_idx]] = False
            
            if not uncovered.any():
                break
                
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
            
        # If we found a very good solution (e.g., close to lower bound 131), we might stop?
        # But we don't know the exact optimum. 151 is known best.
        # If we get <= 151, we can stop?
        # Actually, if we get 151, we can't do better than known best?
        # Wait, the task says "best known covering has 151".
        # If we find 151, it's a tie with best known.
        # If we find < 151, it's a new record!
        # But finding < 151 is extremely unlikely with simple heuristics.
        # So we just minimize.

    if best_blocks is None:
        # Fallback to pure greedy if randomization failed or something
        # But the loop above always produces something.
        # Just in case, if best_blocks is None, run one pure greedy.
        uncovered = np.ones(num_t_subsets, dtype=bool)
        blocks = []
        while True:
            if time.time() > deadline:
                break
            counts = uncovered[block_indices].sum(axis=1)
            if counts.max() == 0: break
            best_idx = np.argmax(counts)
            blocks.append(list(k_subsets[best_idx]))
            uncovered[block_indices[best_idx]] = False
            if not uncovered.any(): break
        best_blocks = blocks
        
    return best_blocks
