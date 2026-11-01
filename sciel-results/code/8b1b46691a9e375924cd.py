import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # Build incidence matrix
    # For v=19, k=8, t=5:
    # n_k = 75582, n_t = 11628
    # Dense matrix is ~800MB. We assume this is acceptable.
    # If not, we would need a sparse approach, but let's try dense first for speed.
    A = np.zeros((n_k, n_t), dtype=np.uint8)
    for i, block in enumerate(k_blocks):
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            A[i, idx] = 1

    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with random noise to break ties and explore different greedy paths
    num_restarts = 500  # Can be high because each run is fast
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered = np.ones(n_t, dtype=np.uint8)
        selected_blocks = []
        
        while True:
            if time.time() > deadline:
                break
                
            # Calculate scores
            scores = A @ uncovered
            
            # Add small random noise to break ties and encourage diversity
            # The noise should be small enough not to override significant differences
            # but large enough to change the choice in case of ties or near-ties.
            # Max score is binom(k, t) = 56.
            # Noise scale: 0.1
            if restart > 0:
                noise = np.random.uniform(0, 0.1, size=n_k)
                scores = scores + noise
            else:
                noise = 0
                
            best_idx = np.argmax(scores)
            
            if scores[best_idx] == 0:
                break
                
            selected_blocks.append(k_blocks[best_idx])
            
            # Update uncovered mask
            # Get indices of 1s in row best_idx
            # A[best_idx] is a boolean-like array (uint8)
            mask = A[best_idx] == 1
            uncovered[mask] = 0
            
        if len(selected_blocks) < best_count:
            best_count = len(selected_blocks)
            best_blocks = [list(b) for b in selected_blocks]
            
            # If we hit the theoretical lower bound or a known good bound, we can stop early?
            # Lower bound for C(19,8,5) is 264.
            # If we get close to 335 (best known), it's good.
            # We don't know the optimal, so we keep searching for better.
            
    if best_blocks is None:
        # Fallback if something went wrong (should not happen)
        # Return a trivial cover (might be huge) or empty
        return []
        
    return best_blocks
