# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    such that their maximum norm is smaller than their minimum pairwise distance.
    
    Strategy:
    1. Generate candidate points from the E8 lattice and A2 lattice structures.
       - E8⊕A2 is a highly dense lattice in 10D. We can embed it in 11D by adding a 0 coordinate.
       - Alternatively, we can use the D11 lattice (even weight, sum even) or just general integer vectors.
       - The current best (220) uses vectors with norm^2=2 (L1=2).
       - We can try to add vectors with norm^2=4 (L1=4 or 2*2).
       - Actually, a better approach is to search for a large clique in the graph where edges connect points with squared distance >= max_norm^2.
       - Since we want to maximize k, and the constraint is min_dist^2 >= max_norm^2, we should try to keep norms low.
       - If we restrict to norm^2 <= 4, the max norm is 4. We need min_dist^2 >= 4.
       - This allows us to use more points than the norm^2=2 set (which required dist^2 >= 2, but effectively limited by geometry).
       
    2. Candidate Generation:
       - All vectors with norm^2 = 2 (220 points).
       - All vectors with norm^2 = 4.
         Types:
         a) One 2, rest 0: 11 * 2 = 22 points.
         b) One 2, two 1s? No, 2^2 + 1^2 + 1^2 = 6.
         c) Two 2s: 2^2 + 2^2 = 8.
         d) Four 1s: 1^2*4 = 4. C(11,4) * 2^4 = 330 * 16 = 5280 points.
         e) One 2, one -2? 2^2 + (-2)^2 = 8.
         f) One 2, two 1s with signs? 4 + 1 + 1 = 6.
         
       So norm^2=4 candidates are:
       - 4 ones: 5280
       - 2 and two 1s is norm^2=6.
       - 2 and 2 is norm^2=8.
       
       Let's include norm^2=4 and norm^2=8?
       If max norm is 8, we need min dist^2 >= 8.
       If max norm is 4, we need min dist^2 >= 4.
       
       The 220 points (norm^2=2) have min dist^2 = 4. So they satisfy dist^2 >= 4.
       Can we add norm^2=4 points to the 220?
       Let S be the 220 points. Max norm = 2.
       Add v with norm^2=4. New max norm = 4.
       We need ||v-u||^2 >= 4 for all u in S ∪ {v}.
       ||v-u||^2 = ||v||^2 + ||u||^2 - 2v·u = 4 + 2 - 2v·u = 6 - 2v·u.
       We need 6 - 2v·u >= 4 => 2v·u <= 2 => v·u <= 1.
       
       So we can add v (norm^2=4) if v·u <= 1 for all u in the current set.
       
       This is a greedy packing problem. We will:
       1. Start with the 220 points.
       2. Generate all norm^2=4 candidates.
       3. Greedily add candidates if they satisfy the dot product constraint against the current set.
       4. Also try to swap or restart with different orderings to find a larger set.
       
       To speed up, we can precompute dot products or use bitsets if the set is small enough, but 220 + 5000 is manageable with vectorized numpy operations in batches.
       
       Let's try a multi-start greedy approach.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # 1. Generate base set: norm^2 = 2
    base_points = []
    for i in range(d):
        for j in range(i + 1, d):
            base_points.append(np.array([1 if k==i else -1 if k==j else 0 for k in range(d)], dtype=np.int64))
            base_points.append(np.array([-1 if k==i else 1 if k==j else 0 for k in range(d)], dtype=np.int64))
            base_points.append(np.array([1 if k==i else 1 if k==j else 0 for k in range(d)], dtype=np.int64))
            base_points.append(np.array([-1 if k==i else -1 if k==j else 0 for k in range(d)], dtype=np.int64))
    
    base_arr = np.array(base_points, dtype=np.int64)
    
    # 2. Generate candidates: norm^2 = 4
    # Type A: Four 1s
    cand_A = []
    for indices in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, s in zip(indices, signs):
                v[idx] = s
            cand_A.append(v)
            
    # Type B: One 2 (and rest 0) -> Norm^2=4? No, 2^2=4. Yes.
    cand_B = []
    for i in range(d):
        cand_B.append(np.array([2 if k==i else 0 for k in range(d)], dtype=np.int64))
        cand_B.append(np.array([-2 if k==i else 0 for k in range(d)], dtype=np.int64))
        
    candidates = np.array(cand_A + cand_B, dtype=np.int64)
    
    # We want to find the largest subset of (base + candidates) such that min_dist^2 >= max_norm^2.
    # Since base has norm^2=2 and candidates have norm^2=4, the max_norm will be 4 if we add any candidate.
    # If we only use base, max_norm=2, min_dist=4. Valid. k=220.
    # If we add candidates, max_norm=4. We need min_dist^2 >= 4.
    
    # Greedy Add:
    # Start with base.
    # For each candidate, check if it can be added.
    # Condition for v (norm^2=4) to be added to set S (containing base and some candidates):
    # Let R = max_norm^2(S). If S is only base, R=2. If S has any candidate, R=4.
    # Case 1: S is only base. R=2.
    #   Need ||v-u||^2 >= 2 for all u in S.
    #   ||v-u||^2 = 4 + 2 - 2v·u = 6 - 2v·u >= 2 => v·u <= 2.
    #   Since u has norm^2=2, max v·u is 2 (if v and u are parallel? No, u has two 1s, v has four 1s or one 2.
    #   If v=(2,0..), u=(1,1..), v·u=2. 6-4=2 >= 2. OK.
    #   So initially, many candidates might be addable.
    
    # Case 2: S has candidates. R=4.
    #   Need ||v-u||^2 >= 4 for all u in S.
    #   For u in base (norm^2=2): 4+2-2v·u >= 4 => 6-2v·u >= 4 => v·u <= 1.
    #   For u in candidates (norm^2=4): 4+4-2v·u >= 4 => 8-2v·u >= 4 => v·u <= 2.
    
    # Strategy:
    # 1. Try to add as many candidates as possible to the base set.
    # 2. Use a greedy algorithm with random restarts or different orderings.
    
    best_set = base_arr
    best_k = len(base_arr)
    
    # Precompute dot products of candidates with base?
    # Base is 220x11. Candidates is ~5300x11.
    # Dot product matrix is 5300x220. This is small enough to compute in numpy.
    
    # Let's do a few greedy passes with different orderings.
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Shuffle candidates
        order = np.random.permutation(len(candidates))
        
        current_set = [base_arr]
        current_max_norm_sq = 2
        
        added_cands = []
        
        # We maintain the set of points as a list of arrays for easy appending, 
        # but for checking constraints, we need dot products with all existing points.
        # To speed up, we can maintain a "conflict" list or just recompute dot products 
        # against the new candidate.
        
        # Efficient check:
        # For a new candidate v, we need:
        # 1. v·u <= 1 for all u in base (if current_max_norm_sq == 4) OR v·u <= 2 (if current_max_norm_sq == 2)
        # 2. v·u <= 2 for all u in added_cands (since added_cands have norm^2=4, and new max_norm will be 4)
        
        # Wait, if we add the first candidate, max_norm becomes 4.
        # So after the first candidate is added, the constraint for base points tightens from v·u <= 2 to v·u <= 1.
        # This means we might need to REMOVE some base points? No, we start with base.
        # If we add v, we must ensure v·u <= 1 for ALL u in base.
        # If v·u > 1 for some u in base, we cannot add v unless we remove u.
        # Removing points is complex. Let's just stick to adding candidates that are compatible with ALL base points.
        
        # Constraint for v to be added to {base} ∪ {added}:
        # If added is empty: max_norm=2.
        #   v·u <= 2 for all u in base.
        #   v·v = 4 >= 2. OK.
        # If added is not empty: max_norm=4.
        #   v·u <= 1 for all u in base.
        #   v·u <= 2 for all u in added.
        
        # So, for the FIRST candidate added, we can be looser (v·u <= 2).
        # For subsequent candidates, we must be stricter (v·u <= 1 against base).
        
        # Let's precompute dot products of all candidates against base.
        # dots_base = candidates @ base_arr.T  (5300 x 220)
        
        dots_base = candidates @ base_arr.T
        
        # For each candidate i, min_dot_base = min(dots_base[i, :])
        # If min_dot_base <= 1, it's compatible with base under max_norm=4.
        # If min_dot_base <= 2, it's compatible with base under max_norm=2.
        
        compatible_with_base_strict = np.argwhere(dots_base.min(axis=1) <= 1).flatten()
        compatible_with_base_loose = np.argwhere(dots_base.min(axis=1) <= 2).flatten()
        
        # Greedy:
        # 1. Pick a candidate from loose set that maximizes something? Or just first one?
        #    Let's try to pick one that is "central" or just random.
        
        # Actually, let's just iterate through the shuffled candidates.
        # Keep track of added candidates.
        # For each candidate v:
        #   If no candidates added yet:
        #     Check if v is in loose set.
        #   Else:
        #     Check if v is in strict set.
        #     Check if v·u <= 2 for all u in added_cands.
        
        added_indices = []
        
        # To check against added_cands efficiently, we can maintain a matrix of dot products 
        # or just compute v @ added_cands.T on the fly.
        # Since added_cands is likely small (< 100), on-the-fly is fast.
        
        for idx in order:
            if time.time() > deadline - 5:
                break
                
            v = candidates[idx]
            
            if len(added_indices) == 0:
                # Can add if v·u <= 2 for all u in base
                if idx in compatible_with_base_loose:
                    # Check self? v·v = 4. 4 >= 2. OK.
                    added_indices.append(idx)
            else:
                # Can add if v·u <= 1 for all u in base
                if idx not in compatible_with_base_strict:
                    continue
                
                # Check against added candidates
                # v·u <= 2 for all u in added_cands
                # added_cands are candidates[added_indices]
                # dots_added = v @ candidates[added_indices].T
                # if max(dots_added) <= 2:
                
                # Vectorized check
                added_cands = candidates[added_indices]
                dots_added = v @ added_cands.T
                if np.max(dots_added) <= 2:
                    added_indices.append(idx)
                    
        # Construct the set
        if len(added_indices) > 0:
            final_points = np.vstack([base_arr, candidates[added_indices]])
            k = len(final_points)
            
            # Verify
            norms_sq = np.sum(final_points**2, axis=1)
            max_norm_sq = np.max(norms_sq)
            
            min_dist_sq = float('inf')
            # O(k^2) check. k ~ 220 + 50 = 270. 270^2/2 = 36k. Fast.
            for i in range(k):
                for j in range(i + 1, k):
                    diff = final_points[i] - final_points[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        if min_dist_sq < max_norm_sq:
                            break
                if min_dist_sq < max_norm_sq:
                    break
                    
            if min_dist_sq >= max_norm_sq:
                if k > best_k:
                    best_k = k
                    best_set = final_points
                    
    return best_set

# EVOLVE-BLOCK-END
