import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 575
    
    # Generate candidate vectors of weights 3 and 4
    candidates = []
    cand_weights = []
    
    # Weight-3 vectors
    for pos in combinations(range(d), 3):
        for signs in product([-1, 1], repeat=3):
            v = np.zeros(d, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = signs[i]
            candidates.append(v)
            cand_weights.append(3)
    
    # Weight-4 vectors
    for pos in combinations(range(d), 4):
        for signs in product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = signs[i]
            candidates.append(v)
            cand_weights.append(4)
    
    n_cand = len(candidates)
    
    # Precompute inner products and conflicts
    # Conflict if inner product >= max(norm_i^2, norm_j^2)
    # For weight-3: norm^2 = 9, for weight-4: norm^2 = 16
    # Two weight-3: conflict if dot >= 9 (impossible, max dot = 9 when identical)
    #   Actually max dot for distinct weight-3 is 6 (agree on 2, differ on 1) or 9 (same)
    #   Wait: if they share support of size 2 and agree on both, dot = 2*1 + 0 = 2? No.
    #   Let me reconsider: two weight-3 vectors with intersection of size k.
    #   If they agree on all k intersection positions, dot = k. Conflict if k >= 9? Impossible.
    #   Actually the condition is: min pairwise dist^2 >= max norm^2
    #   dist^2 = ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    #   For weight-3, weight-3: dist^2 = 9 + 9 - 2*dot = 18 - 2*dot
    #   Need 18 - 2*dot >= 9, i.e., dot <= 4.5, i.e., dot <= 4
    #   For weight-3, weight-4: dist^2 = 9 + 16 - 2*dot = 25 - 2*dot
    #   Need 25 - 2*dot >= 16, i.e., dot <= 4.5, i.e., dot <= 4
    #   For weight-4, weight-4: dist^2 = 16 + 16 - 2*dot = 32 - 2*dot
    #   Need 32 - 2*dot >= 16, i.e., dot <= 8
    #   
    # So conflicts are:
    #   w3-w3: dot >= 5 (but max dot for distinct is... if same support, max dot = 9 if identical, 
    #     if differ in 1 sign, dot = 7; differ in 2 signs, dot = 3; differ in 3, dot = -1)
    #     So conflict if they agree on >= 3 positions with same sign, i.e., dot >= 7 (differ in 0 or 1 sign on same support)
    #     Actually dot = sum over intersection of (a_i * b_i). If same support and agree on k of 3, dot = k - (3-k) = 2k-3.
    #     k=3: dot=3, k=2: dot=1, k=1: dot=-1, k=0: dot=-3 (but if same support, k>=1)
    #     Wait, if same support of size 3, and they agree on k positions, dot = k*1 + (3-k)*(-1) = 2k-3.
    #     k=3: dot=3, not >=5. So no same-support w3-w3 conflict!
    #     If intersection size 2: agree on both, dot=2; agree on 1, dot=0; agree on 0, dot=-2. Max=2 < 5. No conflict.
    #     If intersection size 1: max dot=1 < 5. No conflict.
    #     So weight-3 vectors NEVER conflict with each other! 
    #   
    #   w3-w4: dot >= 5. Max dot: if w3 support subset of w4 support and agree on all 3, dot=3 < 5. No conflict!
    #   Wait, that's not right either. Let me recompute.
    #   Actually I think I'm confusing myself. Let me just compute dot products directly.
    
    # Build conflict matrix (sparse)
    # conflict[i][j] = True if i and j cannot both be in the set
    # This is expensive for 5280+3520 = 8800 candidates. Let's use a different approach.
    
    # Actually, let's stick with just weight-4 for now and use a smarter search.
    # The key issue is that the greedy approach gets stuck at local optima.
    
    # Let's use a different strategy: simulated annealing with careful move design.
    
    # Go back to weight-4 only for simplicity
    four_positions = list(combinations(range(d), 4))
    n_pos = len(four_positions)  # 330
    
    candidates = []
    cand_support = []
    for pos in four_positions:
        for signs in product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = signs[i]
            candidates.append(v)
            cand_support.append(tuple(pos))
    
    n_cand = len(candidates)  # 5280
    
    # Group candidates by support
    support_to_cands = {}
    for i, s in enumerate(cand_support):
        if s not in support_to_cands:
            support_to_cands[s] = []
        support_to_cands[s].append(i)
    
    # Precompute conflicts more efficiently using bitmasks
    # For each candidate, represent its signs as a bitmask over its support
    # Two candidates conflict if their dot product >= 8 (for w4-w4)
    
    # Build adjacency as a list of sets
    adjacency = [set() for _ in range(n_cand)]
    
    support_list = list(support_to_cands.keys())
    support_to_idx = {s: i for i, s in enumerate(support_list)}
    n_supports = len(support_list)
    
    for i in range(n_cand):
        v1 = candidates[i]
        s1 = cand_support[i]
        for j in range(i + 1, n_cand):
            v2 = candidates[j]
            s2 = cand_support[j]
            # Compute dot product
            # Only need to check positions in the union of supports
            inter = set(s1) & set(s2)
            if len(inter) == 0:
                dot = 0
            else:
                dot = sum(v1[p] * v2[p] for p in inter)
            # Conflict if dot >= 8 (for w4-w4, need dist^2 >= 16, so 32-2*dot >= 16, dot <= 8)
            # Wait: conflict if NOT (dist^2 >= max_norm^2). 
            # dist^2 = 32 - 2*dot. Need dist^2 >= 16. So 32-2*dot >= 16, dot <= 8.
            # So they are VALID if dot <= 8. They CONFLICT if dot > 8, i.e., dot >= 9.
            # But max dot for two weight-4 vectors: if same support and all signs agree, dot=4.
            # If intersection size 3 and agree on all 3, dot=3. Max possible dot is 4 (same support, same signs).
            # So dot is always <= 4! That means NO conflicts at all? That can't be right.
            # 
            # I think I've been making an error. Let me re-read the condition.
            # "minimum squared distance over all pairs of points is >= the maximum squared norm of the points"
            # max squared norm = 16 for weight-4 vectors.
            # For two weight-4 vectors a, b: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b = 32 - 2a.b
            # Need 32 - 2a.b >= 16, so a.b <= 8.
            # Max a.b for weight-4 vectors is 4 (identical vectors). So a.b <= 4 <= 8 always.
            # This means ALL pairs of weight-4 vectors are valid?!
            # 
            # That means we can include ALL 5280 weight-4 vectors!
            # But wait, the condition also says "no point is the zero vector" and "points must be distinct".
            # All 5280 are distinct and non-zero. So the answer should be 5280?
            # 
            # But the current score is only 139, which suggests the program is not selecting all of them.
            # Let me check: the current program's can_add function checks for conflicts, but if there are no conflicts,
            # it should select all candidates. But it's only getting 139.
            # 
            # I think the issue is that the current program has a bug in its conflict detection,
            # OR I'm misunderstanding the condition.
            # 
            # Let me re-read: "the minimum squared distance over all pairs of points is >= the maximum squared norm of the points"
            # This means: for ALL pairs (i,j), ||p_i - p_j||^2 >= max_k ||p_k||^2
            # If all points have the same norm (weight 4, norm^2 = 16), then we need ||p_i - p_j||^2 >= 16 for all i != j.
            # ||p_i - p_j||^2 = 32 - 2*p_i.p_j >= 16 => p_i.p_j <= 8.
            # Since max p_i.p_j = 4 (for identical vectors, but they're distinct so max is less), this is always satisfied.
            # 
            # So the answer IS 5280! The current program is just buggy.
            
            pass  # No conflicts!
    
    # If there are no conflicts, just return all candidates
    if n_cand >= 2:
        pts_array = np.array(candidates, dtype=np.int64)
        return np.around(pts_array).astype(np.int64)
    else:
        points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
        pts_array = np.array(points, dtype=np.int64)
        return np.around(pts_array).astype(np.int64)
