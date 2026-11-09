# EVOLVE-BLOCK-START
import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.
    
    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    start_time = time.time()
    deadline = 600.0
    
    d = 11
    
    # Generate a binary code of length 11 with minimum Hamming distance 3
    # This corresponds to points with coordinates +/-1 such that 
    # min pairwise squared distance >= 4 * 3 = 12 > 11 (max norm squared)
    
    # Parity check matrix H is 4 x 11
    # We pick 11 distinct non-zero columns from F_2^4
    # There are 15 non-zero vectors in F_2^4.
    
    cols = []
    for i in range(1, 16):
        vec = [(i >> j) & 1 for j in range(4)]
        cols.append(vec)
        
    # Take first 11 columns
    H_cols = cols[:11]
    
    # Construct the code: all x in {0,1}^11 such that H x = 0 (mod 2)
    # We can do this by finding the null space over GF(2)
    # Since we just need a valid set, and we know the size should be 2^(11-4)=128 if rank is 4,
    # we can iterate through all 2^11 possibilities? No, that's 2048, which is fine.
    # But we can be smarter.
    
    # Let's just brute force the 2^11 possibilities to be safe and simple, 
    # as 2048 is very small.
    
    points_bin = []
    for mask in range(2**11):
        x = [(mask >> i) & 1 for i in range(11)]
        
        # Check H x = 0 mod 2
        valid = True
        for j in range(4):
            dot = 0
            for i in range(11):
                dot += H_cols[i][j] * x[i]
            if dot % 2 != 0:
                valid = False
                break
        
        if valid:
            points_bin.append(x)
            
    # Convert to +/-1
    # 0 -> -1, 1 -> 1 (or 0->1, 1->-1, doesn't matter)
    points = np.array([[1 if b == 1 else -1 for b in p] for p in points_bin], dtype=np.int64)
    
    # Verify condition
    # Max norm squared
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Min pairwise distance squared
    min_dist_sq = float('inf')
    n_points = len(points)
    for i in range(n_points):
        for j in range(i+1, n_points):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                
    if min_dist_sq >= max_norm_sq and n_points >= 2:
        return points
    else:
        # Fallback to seed
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)

# EVOLVE-BLOCK-END
