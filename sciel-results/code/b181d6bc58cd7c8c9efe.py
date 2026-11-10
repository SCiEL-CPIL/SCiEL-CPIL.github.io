# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Calculate forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Precompute adjacency masks
    # adj_mask[i] is a bitmask where bit j is 1 if j is a forbidden difference from i
    # i.e., (i - j) % m in forbidden OR (j - i) % m in forbidden
    # Since forbidden is symmetric (if d in F, -d in F), we just check one way.
    adj_masks = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i - d) % m
            mask |= (1 << j)
            # Since d is in F, -d is also in F (square of -x), so (i + d) % m is also a neighbor
            j2 = (i + d) % m
            mask |= (1 << j2)
        adj_masks[i] = mask

    def popcount(x):
        return bin(x).count('1')

    def random_greedy():
        """Generate a random maximal independent set using greedy algorithm."""
        mask = 0
        # Shuffle vertices
        vertices = list(range(m))
        random.shuffle(vertices)
        for v in vertices:
            # Check if v can be added
            # v can be added if no neighbor of v is in mask
            if (mask & adj_masks[v]) == 0:
                mask |= (1 << v)
        return mask

    def local_search(mask):
        """Try to improve the mask by swapping one vertex for a non-neighbor set."""
        current_size = popcount(mask)
        improved = True
        while improved:
            improved = False
            # Try adding each vertex not in mask
            # We iterate over all possible v
            for v in range(m):
                if mask & (1 << v):
                    continue
                # If we add v, we must remove all neighbors of v
                # new_mask = (mask & ~adj_masks[v]) | (1 << v)
                # But we must ensure v itself is not removed by adj_masks[v]
                # adj_masks[v] does not contain v because 0 is not in forbidden.
                new_mask = (mask & ~adj_masks[v]) | (1 << v)
                new_size = popcount(new_mask)
                if new_size > current_size:
                    mask = new_mask
                    current_size = new_size
                    improved = True
                    break # Restart the loop with the new mask
        return mask, current_size

    best_mask = 0
    best_size = 0
    
    start_time = time.time()
    time_limit = 590.0 # Safety margin
    
    # Initial random restarts
    # We can tune the number of restarts, but the loop condition handles it.
    while time.time() - start_time < time_limit:
        mask = random_greedy()
        mask, size = local_search(mask)
        
        if size > best_size:
            best_size = size
            best_mask = mask
            # Optional: If we reach a theoretical upper bound, we could stop.
            # Upper bound: n / (avg_degree + 1) is a weak bound.
            # Just keep searching.

    # Convert best_mask to array
    result = []
    for i in range(m):
        if best_mask & (1 << i):
            result.append(i)
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
