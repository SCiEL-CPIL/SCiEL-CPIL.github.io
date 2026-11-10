# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    
    Strategy: Use the 2E8 ⊕ 2D3 lattice construction to get 252 points with norm^2 = 8.
    Then, attempt to extend the set by greedily adding additional integer vectors
    with norm^2 >= 8 (and potentially higher) that satisfy the distance constraint:
    min_dist^2 >= max_norm^2.
    
    The base 252 points all have norm^2 = 8.
    If we add a vector v with norm^2 = N > 8, the constraint becomes:
    For all existing u (norm^2 = 8): ||v-u||^2 >= N
    N + 8 - 2*v.u >= N  =>  8 - 2*v.u >= 0  =>  v.u <= 4.
    
    Also, for any two new vectors v, w with norm^2 = N:
    2N - 2*v.w >= N  =>  N >= 2*v.w.
    
    This greedy extension allows us to potentially exceed the 252 limit of the pure lattice construction
    by filling in "gaps" with higher-norm vectors that are sufficiently orthogonal to the base set.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    d = 11
    points_list = []
    
    # --- Generate 240 Scaled E8 Vectors (8 dims) ---
    # Type 1: (±2, ±2, 0, 0, 0, 0, 0, 0)
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(8, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                p = np.zeros(d, dtype=np.int64)
                p[:8] = v
                points_list.append(p)
                
    # Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of minus signs
    # 128 such vectors.
    for signs in product([1, -1], repeat=8):
        if signs.count(-1) % 2 == 0:
            v = np.array(signs, dtype=np.int64)
            p = np.zeros(d, dtype=np.int64)
            p[:8] = v
            points_list.append(p)
            
    # --- Generate 12 Scaled D3 Vectors (3 dims) ---
    # (±2, ±2, 0) permutations in last 3 dims (indices 8, 9, 10)
    for i, j in combinations([8, 9, 10], 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    base_k = len(points)
    
    # --- Greedy Extension Phase ---
    # Try to add more points with higher norms.
    # We'll look for integer vectors with norm^2 in {9, 10, 12, ...}
    # Constraint for adding v (norm^2 = N) to a set S:
    # 1. For all u in S: ||v-u||^2 >= max(N, ||u||^2)
    #    Since ||u||^2 = 8 for base points, and N >= 8, max is N.
    #    N + 8 - 2*v.u >= N  =>  v.u <= 4.
    # 2. For all existing added points w (norm^2 = M):
    #    ||v-w||^2 >= max(N, M)
    #    N + M - 2*v.w >= max(N, M)
    #    If N >= M: N + M - 2*v.w >= N => M >= 2*v.w => v.w <= M/2
    #    If M > N: N + M - 2*v.w >= M => N >= 2*v.w => v.w <= N/2
    
    # To keep it simple and fast, we will primarily look for vectors with norm^2 = 9 (weight 3 with one 2? No, weight 3 with 1,1,1 is norm 3. Wait.)
    # Integer vectors with norm^2 = 9:
    # (±3, 0, ...) -> 22 vectors.
    # (±2, ±2, ±1, ...) -> 2^2 * 1 * C(11,3) * 3! ? No.
    # Let's just generate candidates with norm^2 up to 12.
    
    candidates = []
    # Norm^2 = 9
    # Type A: (±3, 0, ...)
    for i in range(d):
        for s in [1, -1]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = 3 * s
            candidates.append(p)
    # Type B: (±2, ±2, ±1, 0, ...)
    for i, j, k in combinations(range(d), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 2 * si
                    p[j] = 2 * sj
                    p[k] = sk
                    candidates.append(p)
    
    # Sort candidates by norm^2 (ascending) to prefer lower norms first?
    # Actually, higher norms are harder to fit. Lower norms (9) are easier than 12.
    # But all have norm^2=9 here.
    
    # Shuffle candidates to break symmetry
    rng = np.random.default_rng(42)
    cand_indices = np.arange(len(candidates))
    rng.shuffle(cand_indices)
    
    added_count = 0
    for idx in cand_indices:
        if time.time() > deadline - 5:
            break
            
        v = candidates[idx]
        norm_v_sq = np.dot(v, v)
        
        # Check against all existing points
        valid = True
        for u in points:
            norm_u_sq = np.dot(u, u)
            dot = np.dot(v, u)
            dist_sq = norm_v_sq + norm_u_sq - 2 * dot
            max_norm_sq = max(norm_v_sq, norm_u_sq)
            if dist_sq < max_norm_sq:
                valid = False
                break
        
        if valid:
            points = np.vstack([points, v])
            added_count += 1
            # Optional: stop if we've added a lot? No, keep going until time.
            # But checking against growing list is O(k^2).
            # 252 + added. If we add 100, k=352. Pairs ~ 60k. Dot products are fast.
            # 590s is plenty.
            
    if time.time() > deadline:
         return np.array(points[:2], dtype=np.int64)

    return points.astype(np.int64)

# EVOLVE-BLOCK-END
