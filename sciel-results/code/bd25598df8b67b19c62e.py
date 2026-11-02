"""
Strategy: Deterministic symmetry-preserving greedy construction.
Focuses on D11 lattice roots (weight 2) and extends with weight 4 vectors
using a 'maximize minimum distance' greedy rule.
"""
import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pool
    # Prioritize Weight 2 (D11 roots) as they form the densest, most symmetric core.
    # Then Weight 4 (which allows for more distinct directions).
    # Weight 3 is skipped initially as it often creates odd-distance issues 
    # and is less "regular" in the lattice sense for kissing numbers.
    
    candidates = []
    
    # Weight 2 vectors: (±1, ±1, 0, ...)
    # Count: C(11,2) * 4 = 55 * 4 = 220
    # These are the roots of D11.
    for comb in itertools.combinations(range(D), 2):
        i, j = comb
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = s1
                v[j] = s2
                candidates.append(v)
                
    # Weight 4 vectors: (±1, ±1, ±1, ±1, 0, ...)
    # Count: C(11,4) * 16 = 330 * 16 = 5280
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Precompute Gram matrix
    # 5500 x 5500 is ~ 242 MB for int64. Acceptable.
    G = candidates @ candidates.T
    
    # Conflict Condition:
    # We need pairwise distance >= max_norm.
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2<u,v> >= max(||u||^2, ||v||^2)
    # Let M = max(||u||^2, ||v||^2).
    # ||u||^2 + ||v||^2 - 2<u,v> >= M
    # 2<u,v> <= ||u||^2 + ||v||^2 - M
    # Since M is the max, let m be the min. ||u||^2 + ||v||^2 = M + m.
    # 2<u,v> <= m
    # <u,v> <= m/2
    
    min_norms_sq = np.minimum(norms_sq[:, None], norms_sq[None, :])
    limits = min_norms_sq / 2.0
    np.fill_diagonal(limits, np.inf)
    
    conflict_mask = G > limits
    np.fill_diagonal(conflict_mask, False)
    
    # Precompute neighbor lists for fast updates
    neighbor_indices = []
    for i in range(n_cand):
        neighbor_indices.append(np.nonzero(conflict_mask[i])[0])
        
    # 2. Greedy Selection with "Max Angular Separation" Heuristic
    # Instead of random order, we order candidates to maximize spread.
    # We start with a "seed" vector that is likely central or representative.
    # Then, at each step, we pick the candidate that is farthest (in terms of 
    # minimum distance to already selected vectors) from the current set.
    # This is a "Greedy Farthest Point" or "Max-Min" heuristic.
    
    # To make it deterministic and fast, we can pre-sort candidates by a heuristic.
    # However, dynamic max-min is O(N^2 * K). N=5500, K~100-200.
    # 5500^2 * 150 = 4.5 * 10^9 ops. Too slow in Python.
    
    # Alternative: 
    # 1. Start with a small set of highly symmetric vectors (e.g., all weight-2 vectors 
    #    that are pairwise compatible? No, weight-2 vectors conflict if they share 
    #    an index with same sign or different sign? 
    #    u=(1,1,0..), v=(1,-1,0..) -> dot=0. dist^2=4. max_norm=2. 4>=2 OK.
    #    u=(1,1,0..), v=(1,1,0..) -> same.
    #    u=(1,1,0..), v=(-1,-1,0..) -> dot=-2. dist^2=8. OK.
    #    u=(1,1,0..), v=(1,0,1..) -> dot=1. dist^2=2+2-2=2. max_norm=2. 2>=2 OK.
    #    u=(1,1,0..), v=(1,0,-1..) -> dot=1. dist^2=2. OK.
    #    Actually, ALL weight-2 vectors are mutually compatible!
    #    Let's verify: u, v weight 2. 
    #    If disjoint support: dot=0. dist^2=4. max=2. 4>=2. OK.
    #    If share 1 index: dot = ±1. dist^2 = 2+2∓2 = 0 or 4. 
    #       If dot=1 (same sign), dist^2=0. CONFLICT (distance 0 < norm 2).
    #       If dot=-1 (diff sign), dist^2=4. OK.
    #    If share 2 indices (identical support):
    #       If same signs: dot=2. dist^2=0. CONFLICT.
    #       If 1 diff sign: dot=0. dist^2=4. OK.
    #       If 2 diff signs: dot=-2. dist^2=8. OK.
    
    # So, we can select a large independent set from the weight-2 vectors first.
    # The weight-2 vectors form a graph where edges exist if they conflict.
    # Conflicts in weight-2:
    # 1. Identical vectors (dot=2).
    # 2. Vectors sharing exactly one index with the same sign (dot=1).
    #    Note: (1,1) and (1,0,1) share index 0 with sign 1. Conflict.
    #    (1,1) and (-1,0,1) share index 0 with sign -1 vs 1? 
    #    u=(1,1,0), v=(-1,0,1). dot = -1. dist^2=4. OK.
    #    So conflict only if dot > 1? No, limit is min_norm/2 = 1.
    #    So dot <= 1 is OK.
    #    Wait, limit is m/2. m=2. Limit=1.
    #    So dot <= 1 is compatible.
    #    dot > 1 is conflict.
    #    For weight 2, max dot is 2 (identical).
    #    So only identical vectors conflict?
    #    Let's re-evaluate dot=1 case.
    #    u=(1,1,0), v=(1,0,1). dot=1. 1 <= 1. Compatible.
    #    So, actually, NO two distinct weight-2 vectors conflict?
    #    Let's check u=(1,1,0), v=(1,1,0) -> dot=2. Conflict.
    #    So the only conflicts among weight-2 vectors are between identical vectors (which are the same candidate) or... 
    #    Wait, are there other conflicts?
    #    If u and v are distinct weight-2 vectors, max dot is 1 (if they share 1 index with same sign) or 0 or -1 or -2.
    #    Since limit is 1, dot=1 is allowed.
    #    Therefore, ALL distinct weight-2 vectors are mutually compatible!
    
    # This is a huge insight. We can take ALL 220 weight-2 vectors.
    # Let's verify this logic carefully.
    # Condition: ||u-v|| >= max(||u||, ||v||).
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v>.
    # For weight 2, ||u||^2 = 2, ||v||^2 = 2.
    # ||u-v||^2 = 4 - 2<u,v>.
    # We need 4 - 2<u,v> >= 2  => 2 >= 2<u,v> => 1 >= <u,v>.
    # Max <u,v> for distinct weight-2 vectors:
    # If they share 2 indices (same support):
    #   Same signs: <u,v>=2. (Identical vectors, not distinct)
    #   1 sign diff: <u,v>=0.
    #   2 sign diff: <u,v>=-2.
    # If they share 1 index:
    #   Same sign: <u,v>=1.
    #   Diff sign: <u,v>=-1.
    # If they share 0 indices:
    #   <u,v>=0.
    # So, for any two DISTINCT weight-2 vectors, <u,v> <= 1.
    # Thus, they are ALWAYS compatible.
    
    # So, we start with all 220 weight-2 vectors.
    
    selected_indices = []
    in_set = np.zeros(n_cand, dtype=bool)
    
    # Indices of weight-2 vectors are the first 220 in our candidate list.
    w2_indices = np.arange(220)
    for idx in w2_indices:
        in_set[idx] = True
        selected_indices.append(idx)
        
    current_score = len(selected_indices)
    best_score = current_score
    best_in_set = in_set.copy()
    
    # Update conflict counts for the rest of the candidates
    current_conflicts = np.zeros(n_cand, dtype=np.int32)
    for idx in w2_indices:
        current_conflicts[neighbor_indices[idx]] += 1
        # Note: neighbor_indices[idx] includes other w2 vectors, but we don't care 
        # about conflicts among selected w2 vectors because we already know they are compatible.
        # Wait, if they are compatible, conflict_mask[idx, other_w2] should be False.
        # Let's check conflict_mask logic.
        # G[idx, other_w2] <= 1. limits[idx, other_w2] = 1.
        # So conflict_mask is False.
        # So current_conflicts for other w2 vectors will NOT be incremented.
        # This is correct.
        
    # Now, greedily add weight-4 vectors.
    # Heuristic: Pick the weight-4 vector that has the fewest conflicts with the current set?
    # Or the one that is "farthest" in some sense?
    # Since we want to maximize k, we just want to add as many as possible.
    # Greedy: At each step, pick the candidate with 0 conflicts.
    # If multiple, pick the one that minimizes the "average dot product" with the set?
    # Or simply pick the first one?
    # To ensure symmetry and avoid bias, we can iterate through weight-4 vectors 
    # in a specific order (e.g., lexicographic) and add if compatible.
    # This is a standard greedy Independent Set approach.
    
    # However, a simple greedy might get stuck. 
    # Let's try a "Weighted Greedy" where we prioritize vectors that allow more future additions.
    # This is hard to estimate.
    
    # Let's stick to a simple deterministic greedy first:
    # Iterate through all weight-4 candidates. If compatible, add.
    # Then, do a second pass? Or use local search?
    
    # Given the time limit, let's do:
    # 1. Add all compatible weight-4 vectors in lexicographic order.
    # 2. If time remains, perform local swaps (remove 1, add 2 if possible).
    
    w4_indices = np.arange(220, n_cand)
    
    # Pass 1: Greedy Add
    for idx in w4_indices:
        if current_conflicts[idx] == 0:
            in_set[idx] = True
            selected_indices.append(idx)
            current_score += 1
            current_conflicts[neighbor_indices[idx]] += 1
            
            if current_score > best_score:
                best_score = current_score
                best_in_set = in_set.copy()
                
    # Pass 2: Local Search / Swaps
    # Try to improve by removing one vector and adding two?
    # Or removing one and adding one? (Net 0, might unlock others)
    # Let's try removing a vector and seeing if we can add more.
    
    # To keep it simple and fast, let's just try adding vectors that were previously 
    # incompatible but might become compatible if we remove a blocker.
    # This requires tracking blockers.
    
    # Simpler: Randomized Restart with Deterministic Seed?
    # No, let's try a "Greedy with Backtracking" on the weight-4 part?
    # Too complex.
    
    # Let's try a different heuristic for the weight-4 selection:
    # Sort weight-4 vectors by their "degree" in the conflict graph (number of conflicts with W2 set).
    # Vectors with fewer conflicts are "easier" to place.
    
    # Recalculate conflicts with just W2 set?
    # We already have current_conflicts which includes W2 and previously added W4.
    
    # Let's restart the W4 selection with a smarter order.
    # Reset state to just W2.
    in_set = np.zeros(n_cand, dtype=bool)
    current_conflicts = np.zeros(n_cand, dtype=np.int32)
    for idx in w2_indices:
        in_set[idx] = True
        current_conflicts[neighbor_indices[idx]] += 1
        
    current_score = len(w2_indices)
    best_score = current_score
    best_in_set = in_set.copy()
    
    # Calculate initial conflict count for each W4 vector against the W2 set.
    # This is current_conflicts[w4_indices].
    
    # Sort W4 indices by conflict count (ascending).
    # Ties broken by index (deterministic).
    sorted_w4 = w4_indices[np.argsort(current_conflicts[w4_indices], kind='mergesort')]
    
    for idx in sorted_w4:
        if current_conflicts[idx] == 0:
            in_set[idx] = True
            current_score += 1
            current_conflicts[neighbor_indices[idx]] += 1
            
            if current_score > best_score:
                best_score = current_score
                best_in_set = in_set.copy()
                
    # Check time
    if time.time() < deadline:
        # Try to improve with local search
        # Simple hill climbing: Try to add any remaining compatible vector.
        # Then try to swap.
        
        # Add any remaining
        for idx in w4_indices:
            if not in_set[idx] and current_conflicts[idx] == 0:
                in_set[idx] = True
                current_score += 1
                current_conflicts[neighbor_indices[idx]] += 1
                if current_score > best_score:
                    best_score = current_score
                    best_in_set = in_set.copy()
                    
    final_indices = np.nonzero(best_in_set)[0]
    final_vecs = candidates[final_indices]
    
    return final_vecs

# EVOLVE-BLOCK-END
