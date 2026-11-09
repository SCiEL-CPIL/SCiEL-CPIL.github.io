# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
import random

def kissing_number11() -> np.ndarray:
    """
    Attempts to find a large set of integer vectors in Z^11 satisfying:
    1. min pairwise squared distance >= max squared norm.
    2. Points are distinct and non-zero.
    
    Strategy:
    1. Start with the known valid set of 220 weight-2 vectors (norm 2).
    2. Try to add weight-4 vectors (norm 4) if they satisfy the condition
       with all existing points and each other.
       Condition for p (norm 2) and q (norm 4):
       ||p-q||^2 >= 4 => 2 + 4 - 2 p.q >= 4 => p.q <= 1.
       Condition for q1, q2 (norm 4):
       ||q1-q2||^2 >= 4 => 4 + 4 - 2 q1.q2 >= 4 => q1.q2 <= 2.
       
    If the greedy addition of weight-4 vectors fails to beat 220, return the weight-2 set.
    """
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
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
    
    current_points = list(base_points)
    
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
    
    # Shuffle candidates to avoid bias
    random.seed(42)
    random.shuffle(candidates)
    
    # 3. Greedy addition
    # Precompute norms
    norm_sq = [np.dot(p, p) for p in current_points]
    
    for q in candidates:
        if time.time() > deadline:
            break
            
        q_norm_sq = np.dot(q, q)
        
        # Check compatibility with all current points
        compatible = True
        for i, p in enumerate(current_points):
            p_norm_sq = norm_sq[i]
            dot = np.dot(p, q)
            # Condition: ||p-q||^2 >= max(p_norm_sq, q_norm_sq)
            # ||p-q||^2 = p_norm_sq + q_norm_sq - 2 dot
            # p_norm_sq + q_norm_sq - 2 dot >= max(p_norm_sq, q_norm_sq)
            # If p_norm_sq == q_norm_sq == N: 2N - 2dot >= N => dot <= N/2
            # Here p_norm_sq=2, q_norm_sq=4. Max is 4.
            # 2 + 4 - 2dot >= 4 => 2 - 2dot >= 0 => dot <= 1
            
            if p_norm_sq == 2 and q_norm_sq == 4:
                if dot > 1:
                    compatible = False
                    break
            elif p_norm_sq == 4 and q_norm_sq == 4:
                if dot > 2:
                    compatible = False
                    break
            elif p_norm_sq == 4 and q_norm_sq == 2:
                if dot > 1:
                    compatible = False
                    break
            else:
                # Generic check
                max_norm = max(p_norm_sq, q_norm_sq)
                dist_sq = p_norm_sq + q_norm_sq - 2 * dot
                if dist_sq < max_norm:
                    compatible = False
                    break
        
        if compatible:
            current_points.append(q)
            norm_sq.append(q_norm_sq)
            
        if len(current_points) > 220:
            # We have beaten the base set. Continue to see if we can get more?
            # Or stop early if we are confident?
            # Let's continue to maximize.
            pass

    if len(current_points) > 220:
        points = np.array(current_points, dtype=np.int64)
    else:
        points = np.array(base_points, dtype=np.int64)
        
    return points

# EVOLVE-BLOCK-END
