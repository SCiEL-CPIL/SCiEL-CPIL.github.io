# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    such that their maximum squared norm is strictly less than their minimum
    squared pairwise distance. Aims to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    d = 11
    
    # Strategy: Use the hypercube {0, 1}^d or {±1}^d?
    # The condition is: min_{i != j} ||x_i - x_j||^2 > max_i ||x_i||^2
    # Let M = max_i ||x_i||^2. We need ||x_i - x_j||^2 > M for all i != j.
    
    # Consider points in {0, 1}^d.
    # ||x_i||^2 is the number of 1s (weight w_i). So M = max w_i.
    # ||x_i - x_j||^2 is the Hamming distance between x_i and x_j.
    # So we need a binary code of length 11 such that the minimum Hamming distance
    # is strictly greater than the maximum weight of any codeword.
    
    # Let's try to maximize the size of such a code.
    # Let W be the max weight. We need d_min >= W + 1.
    # Also, the code size is limited by the volume/plotkin bound etc.
    
    # Alternatively, consider {±1}^d.
    # ||x_i||^2 = d = 11 for all points. So M = 11.
    # We need ||x_i - x_j||^2 > 11.
    # For x, y in {±1}^d, ||x - y||^2 = sum (x_k - y_k)^2.
    # x_k - y_k is 0 if x_k=y_k, or ±2 if x_k != y_k.
    # So ||x - y||^2 = 4 * HammingDistance(x, y).
    # We need 4 * HD(x, y) > 11 => HD(x, y) >= 3.
    # So we need a binary code of length 11 with minimum Hamming distance 3.
    # The maximum size of such a code is A(11, 3).
    # Known values: A(11, 3) = 512? No, that's A(11, 1).
    # A(n, 3) is related to Hamming codes.
    # For n=11, the max size of a code with d=3 is 2^(11-3) = 256?
    # Wait, the Hamming bound says sum_{i=0}^{(d-1)/2} C(n, i) * A(n, d) <= 2^n.
    # For d=3, sum is 1 + n = 12. So 12 * A(11, 3) <= 2048 => A(11, 3) <= 170.
    # Actually, A(11, 3) is known to be 176? Or 512/3?
    # Let's check standard tables. A(11, 3) = 176.
    # If we can find 176 points in {±1}^11 with min Hamming distance 3,
    # then min squared distance is 4*3=12, and max squared norm is 11.
    # 12 > 11, so this is valid.
    
    # Can we do better?
    # If we use {0, 1}^d, M = max weight.
    # If we restrict to weight <= W, we need min Hamming distance > W.
    # If W=1, min dist > 1 => min dist >= 2.
    # Code of length 11, min dist 2, max weight 1.
    # Points are 000...000, 100...000, ..., 000...001.
    # Size 12. Min dist between 1s is 2 (e.g. 100 and 010 dist is 2).
    # Max weight is 1. 2 > 1. Valid. Size 12.
    
    # If W=2, min dist > 2 => min dist >= 3.
    # Code of length 11, min dist 3, max weight 2.
    # This is a constant weight code A(11, 3, 2).
    # A(11, 3, 2) is the number of pairs in a 2-(11, 2, 1) design? No.
    # It's just the max size of a set of 2-subsets of {1..11} such that any two intersect in at most 0 elements?
    # Wait, Hamming distance between two 2-subsets {a,b} and {c,d} is 4 if disjoint, 2 if share 1, 0 if same.
    # We need dist >= 3, so they must be disjoint.
    # So we need a packing of 2-subsets of {1..11}. Max size is floor(11/2) = 5.
    # Plus maybe the zero vector?
    # If we include 0, dist(0, {a,b}) = 2. We need dist > 2. So 2 is not > 2.
    # So we cannot include 0 if W=2 and min dist must be > 2.
    # So size is 5.
    
    # The {±1}^d approach seems much better (size ~176).
    
    # Let's implement a greedy search for a binary code of length 11 with min Hamming distance 3.
    # We will represent points as bitmasks (integers 0 to 2047).
    # Dist(x, y) = popcount(x XOR y).
    
    # Greedy algorithm:
    # 1. Start with an empty set.
    # 2. Iterate through all 2^11 vectors in some order (e.g., Gray code or random).
    # 3. Add a vector if it is at Hamming distance >= 3 from all existing vectors.
    
    # To maximize the number, randomization or multiple restarts might help,
    # but greedy on a good order often works well for small n.
    # Let's try a few random shuffles and keep the best result.
    
    best_points = []
    
    # Precompute popcount for 0..2047 for speed
    popcount = np.zeros(2048, dtype=np.int32)
    for i in range(1, 2048):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Function to check if a new point can be added
    def can_add(candidate, current_points, popcount):
        # current_points is a list of ints
        # candidate is an int
        for p in current_points:
            if popcount[candidate ^ p] < 3:
                return False
        return True
    
    # Try multiple random restarts
    num_restarts = 100
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Generate a random permutation of all 2^11 vectors
        # 2048 is small, so permutation is fast
        perm = rng.permutation(2048)
        
        current_points = []
        # Add 0 first? Or just start empty.
        # Let's start empty and let the greedy pick.
        
        for val in perm:
            # Check time
            if time.time() > deadline - 5:
                break
            
            if can_add(val, current_points, popcount):
                current_points.append(val)
                
        if len(current_points) > len(best_points):
            best_points = current_points[:]
            
        # Early exit if we hit a known high bound or time is up
        if len(best_points) > 170:
            break

    # Convert best_points (list of ints) to array of shape (k, 11) with values in {±1}
    # Map bit 0 to coordinate 0, etc.
    # If bit is 1, value is 1. If bit is 0, value is -1.
    # Or vice versa, doesn't matter as long as consistent.
    # Let's map 1 -> 1, 0 -> -1.
    
    k = len(best_points)
    if k == 0:
        # Fallback to seed
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        
    points = np.zeros((k, d), dtype=np.int64)
    for i, val in enumerate(best_points):
        for j in range(d):
            if (val >> j) & 1:
                points[i, j] = 1
            else:
                points[i, j] = -1
                
    # Verify validity
    # 1. Shape (k, 11)
    # 2. k >= 2
    # 3. No zero vector (impossible in {±1}^d)
    # 4. Min sq dist > Max sq norm
    
    # Max sq norm is 11 for all points.
    # Min sq dist is 4 * min_hamming_dist.
    # We enforced min_hamming_dist >= 3, so min_sq_dist >= 12.
    # 12 > 11, so valid.
    
    return points

# EVOLVE-BLOCK-END
