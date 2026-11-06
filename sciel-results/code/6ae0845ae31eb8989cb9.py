import numpy as np
import time
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Generate all grid points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.float64)
    n_points = len(points)
    
    best_size = 0
    best_set = []
    
    # Number of restarts, adjust based on time
    # Each restart is O(n_points * |S|^2)
    # If |S| ~ 20, cost ~ 4096 * 400 = 1.6M ops.
    # We can do ~100 restarts in 300s if optimized.
    
    def can_add(p, S):
        """
        Check if point p can be added to set S without forming an isosceles triangle.
        S is a list of np.array points.
        """
        if len(S) == 0:
            return True
        if len(S) == 1:
            return True # Any 2 points form no triangle
        
        # Calculate distances from p to all points in S
        # S is list of (x,y)
        S_arr = np.array(S)
        diffs = S_arr - p
        dists_sq = np.sum(diffs**2, axis=1)
        
        # Check 1: Are any distances from p to S equal? (p is apex)
        # If dists_sq has duplicates, it's isosceles.
        # We can check by sorting or using set.
        # Since we are in Python loop, set is fast for small N.
        if len(set(dists_sq)) != len(dists_sq):
            return False
            
        # Check 2: For each pair (i, j) in S, check if dist(p, Si) == dist(Si, Sj) 
        #        or dist(p, Sj) == dist(Si, Sj)
        # We need dist(Si, Sj). We can compute on the fly or precompute for S.
        # Precomputing for S every time can_add is called is expensive if S is large.
        # But S is small.
        
        n_s = len(S)
        # Precompute pairwise distances in S? 
        # Actually, we can just compute dist(Si, Sj) inside the loop.
        # To optimize, we can precompute a matrix of squared distances for S.
        # But S changes.
        
        # Let's just loop.
        for i in range(n_s):
            si = S[i]
            dpi_sq = dists_sq[i]
            for j in range(i + 1, n_s):
                sj = S[j]
                dpj_sq = dists_sq[j]
                
                # Calculate dist(si, sj) squared
                diff_ij = si - sj
                dij_sq = np.sum(diff_ij**2)
                
                # Check isosceles conditions for triangle (p, si, sj)
                # Sides: dpi, dpj, dij
                if dpi_sq == dij_sq:
                    return False
                if dpj_sq == dij_sq:
                    return False
                # dpi_sq == dpj_sq already checked globally
                
        return True

    # Run multiple greedy restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Random permutation
        indices = np.random.permutation(n_points)
        
        S = []
        for idx in indices:
            p = points[idx]
            if can_add(p, S):
                S.append(p)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # If we found a very good set, we might try to improve it with local search?
        # For now, just restart.
        
    # Convert best_set to list of tuples
    return [(int(p[0]), int(p[1])) for p in best_set]
# EVOLVE-BLOCK-END
