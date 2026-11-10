# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations_with_replacement


def squares_in_square10() -> tuple:
    """
    Packs 10 squares into the unit square to maximize the sum of their side lengths.
    
    Strategy:
    Models the packing as a shelf-based constraint satisfaction problem.
    1. Enumerate integer partitions of 10 into K shelves (K from 2 to 5).
    2. For each partition, determine the optimal height allocation to each shelf
       to maximize the sum of square side lengths, subject to:
       - Width constraint: k_i * s_i <= 1  => s_i <= 1/k_i
       - Height constraint: sum(H_i) <= 1, where H_i = s_i (optimal uniform size)
       - Objective: sum(k_i * s_i)
    3. Select the best partition and heights.
    4. Construct the square centers and sides.
    """
    n = 10
    best_value = 0.0
    best_partition = None
    best_heights = None
    
    # Possible numbers of shelves
    # K=1 is trivial (10 in a row, sum=1)
    # K=10 is trivial (1 in each row, sum=1)
    # We check K=2 to 5
    
    for K in range(2, 6):
        # Generate unique partitions of 10 into K parts
        # We use combinations_with_replacement to get sorted tuples, then permute?
        # Actually, the order of shelves doesn't affect the value, only the placement.
        # So we can just generate sorted partitions.
        
        # Generate all non-decreasing sequences k_1 <= ... <= k_K summing to 10
        # k_i >= 1
        
        def generate_partitions(n, k, min_val=1):
            if k == 1:
                if n >= min_val:
                    yield (n,)
                return
            for i in range(min_val, n - k + 2):
                for rest in generate_partitions(n - i, k - 1, i):
                    yield (i,) + rest
        
        for part in generate_partitions(n, K):
            # Calculate optimal heights for this partition
            # Sort by efficiency (k_i) descending to prioritize high slope shelves
            # Slope is k_i. Kink is 1/k_i.
            indices = sorted(range(K), key=lambda i: part[i], reverse=True)
            
            H = [0.0] * K
            H_rem = 1.0
            
            for idx in indices:
                k_i = part[idx]
                H_max = 1.0 / k_i
                H_alloc = min(H_rem, H_max)
                H[idx] = H_alloc
                H_rem -= H_alloc
                if H_rem <= 1e-9:
                    break
            
            # Calculate value
            val = 0.0
            for i in range(K):
                k_i = part[i]
                s_i = H[i]
                # Value is sum of side lengths = k_i * s_i
                # Note: if H[i] < 1/k_i, then k_i * s_i < 1, which is valid.
                val += k_i * s_i
            
            if val > best_value:
                best_value = val
                best_partition = part
                best_heights = H
    
    # Fallback if no partition found (should not happen)
    if best_partition is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    # Construct the packing
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    y_current = 0.0
    square_idx = 0
    
    # We place shelves from bottom to top.
    # The order of shelves in best_partition doesn't matter for the value,
    # but we should place them in a stable order.
    # Let's just iterate through the partition as is.
    
    for i in range(len(best_partition)):
        k_i = best_partition[i]
        H_i = best_heights[i]
        s_i = H_i
        
        # Calculate x positions for k_i squares of size s_i
        # Center the group horizontally
        total_width = k_i * s_i
        start_x = (1.0 - total_width) / 2.0
        
        for j in range(k_i):
            x_center = start_x + (j + 0.5) * s_i
            y_center = y_current + H_i / 2.0
            
            centers[square_idx, 0] = x_center
            centers[square_idx, 1] = y_center
            sides[square_idx] = s_i
            
            square_idx += 1
        
        y_current += H_i
    
    return centers, sides

# EVOLVE-BLOCK-END
