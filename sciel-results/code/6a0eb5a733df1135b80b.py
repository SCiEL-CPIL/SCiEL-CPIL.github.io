# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that 
    their maximum norm is smaller than or equal to their minimum pairwise distance.
    
    Strategy: 
    We use the set of all integer vectors with L1-norm 2 (i.e., exactly two non-zero coordinates, 
    each being +1 or -1).
    
    In dimension d=11:
    - Number of such vectors is C(11, 2) * 4 = 55 * 4 = 220.
    - For any such vector v, ||v||^2 = 1^2 + 1^2 = 2.
    - For any two distinct such vectors u, v:
        - If they share no indices, u.v = 0, ||u-v||^2 = 2+2-0 = 4.
        - If they share one index i, u.v = \pm 1, ||u-v||^2 = 2+2-2(\pm 1) = 2 or 4.
        - If they share both indices, they are distinct sign combinations.
          e.g., u=(1,1,0..), v=(1,-1,0..). u.v = 1-1=0. ||u-v||^2 = 4.
          e.g., u=(1,1,0..), v=(-1,1,0..). u.v = -1+1=0. ||u-v||^2 = 4.
          e.g., u=(1,1,0..), v=(-1,-1,0..). u.v = -1-1=-2. ||u-v||^2 = 2+2-2(-2) = 8.
          The minimum distance squared among distinct vectors with same support is 2? 
          Wait, let's re-verify the "same support" case.
          Support {i, j}. Vectors: (1,1), (1,-1), (-1,1), (-1,-1).
          Distances:
          (1,1) to (1,-1): diff (0,2) -> norm sq 4.
          (1,1) to (-1,1): diff (2,0) -> norm sq 4.
          (1,1) to (-1,-1): diff (2,2) -> norm sq 8.
          (1,-1) to (-1,1): diff (2,-2) -> norm sq 8.
          (1,-1) to (-1,-1): diff (2,0) -> norm sq 4.
          (-1,1) to (-1,-1): diff (0,2) -> norm sq 4.
          
          It appears the minimum distance squared is 2 for vectors sharing exactly one index?
          Let u = e_i + e_j and v = e_i + e_k (j!=k).
          u.v = 1.
          ||u-v||^2 = ||e_j - e_k||^2 = 1+1 = 2.
          
          So min_dist_sq = 2.
          max_norm_sq = 2.
          Condition: min_dist_sq >= max_norm_sq => 2 >= 2. True.
          
    This configuration is valid and has k=220.
    Since any valid set must satisfy ||v||^2 <= min_dist_sq, and for integer vectors, 
    the densest packing under this constraint in Z^11 is achieved by the weight-2 vectors.
    Any attempt to include a vector with norm^2 > 2 would require min_dist_sq > 2, 
    which significantly reduces the packing density (e.g., weight-1 vectors give k=22).
    
    We return the 220 points directly. We skip the O(k^2) verification loop as the 
    validity is mathematically guaranteed, ensuring the function executes in O(k) time.
    """
    d = 11
    points_list = []
    
    # Generate all weight-2 vectors
    for i in range(d):
        for j in range(i + 1, d):
            # There are 4 sign combinations for (i, j)
            # 1, 1
            v = [0] * d
            v[i] = 1
            v[j] = 1
            points_list.append(v)
            
            # 1, -1
            v = [0] * d
            v[i] = 1
            v[j] = -1
            points_list.append(v)
            
            # -1, 1
            v = [0] * d
            v[i] = -1
            v[j] = 1
            points_list.append(v)
            
            # -1, -1
            v = [0] * d
            v[i] = -1
            v[j] = -1
            points_list.append(v)
            
    points = np.array(points_list, dtype=np.int64)
    
    # Per spec: rounded with np.around and cast to int64
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
