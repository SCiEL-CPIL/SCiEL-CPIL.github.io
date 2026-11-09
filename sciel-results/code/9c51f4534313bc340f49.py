# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
import random

def kissing_number11() -> np.ndarray:
    """
    Find a large set of integer vectors in Z^11 satisfying:
    1. min pairwise squared distance >= max squared norm.
    2. Points are distinct and non-zero.
    
    Strategy:
    1. Start with all 220 weight-2 vectors (norm 2).
    2. Greedily add weight-4 vectors (norm 4) that are compatible.
    3. Try multiple random orderings to maximize additions.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    d = 11
    
    # 1. Generate base set: All weight-2 vectors
    base_points = []
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                base_points.append(p)
    
    # 2. Generate candidate weight-4 vectors
    candidates = []
    for idxs in combinations(range(d), 4):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    for s4 in (1, -1):
                        p = np.zeros(d, dtype=np.int64)
                        p[idxs[0]] = s1
                        p[idxs[1]] = s2
                        p[idxs[2]] = s3
                        p[idxs[3]] = s4
                        candidates.append(p)
    
    # Precompute base norms
    base_norm_sq = [2] * len(base_points)
    
    # Try multiple random orderings
    best_points = list(base_points)
    best_norm_sq = list(base_norm_sq)
    best_count = len(base_points)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        random.seed(trial * 12345 + 678)
        shuffled = list(candidates)
        random.shuffle(shuffled)
        
        current_points = list(base_points)
        current_norm_sq = list(base_norm_sq)
        
        for q in shuffled:
            if time.time() > deadline:
                break
            
            q_norm_sq = 4
            
            # Check compatibility with all current points
            compatible = True
            for i, p in enumerate(current_points):
                p_norm_sq = current_norm_sq[i]
                dot = int(np.dot(p, q))
                # Condition: ||p-q||^2 >= max(p_norm_sq, q_norm_sq)
                max_norm = max(p_norm_sq, q_norm_sq)
                dist_sq = p_norm_sq + q_norm_sq - 2 * dot
                if dist_sq < max_norm:
                    compatible = False
                    break
            
            if compatible:
                current_points.append(q)
                current_norm_sq.append(q_norm_sq)
        
        if len(current_points) > best_count:
            best_count = len(current_points)
            best_points = current_points
            best_norm_sq = current_norm_sq
    
    if best_count > 220:
        points = np.array(best_points, dtype=np.int64)
    else:
        points = np.array(base_points, dtype=np.int64)
        
    return points

# EVOLVE-BLOCK-END
