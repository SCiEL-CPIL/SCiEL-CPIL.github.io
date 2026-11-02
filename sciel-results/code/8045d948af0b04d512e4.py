"""Kissing number in dimension 11: construct nonzero integer vectors with min pairwise distance >= max norm."""
import time
import numpy as np

D = 11
DEADLINE_MARGIN = 5.0  # seconds before the 150s wall-clock kill


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    # --- Step 1: Generate candidate vectors ---
    # The D11 lattice has shortest vectors of norm^2 = 2 (pairs ±e_i ± e_j) and norm^2 = 4 (all ±1).
    # We'll use a mix. The kissing number in 11D is known to be at least 552 (from D11 lattice).
    # D11 lattice: vectors with coordinates all ±1, sum even, OR pairs ±e_i ± e_j with sum 0.
    # Actually, D_n lattice: integer vectors with even sum of coordinates.
    # Shortest vectors in D_11: 
    #   Type A: ±e_i ± e_j (norm^2 = 2), count = 2 * C(11,2) = 110... wait, ±e_i+e_j and ±e_i-e_j
    #   Actually for D_n: shortest vectors are ±e_i ± e_j (all four sign combos for each pair), giving 4*C(n,2) = 2n(n-1)
    #   For n=11: 2*11*10 = 220 vectors of norm^2=2
    #   Type B: vectors with all coordinates ±1 and even number of -1's (sum is even). 
    #   Number of such vectors: 2^(n-1) - ... let me think. All 2^11 = 2048 sign vectors.
    #   Those with even sum: exactly half, so 1024. But we need them to be in D_11 (even sum of coords).
    #   A vector with all ±1 has sum = (# of +1) - (# of -1) = 2*(# of +1) - 11.
    #   This is even iff 11 is even... no, 2*k - 11 is always odd. So NO all-±1 vector is in D_11!
    #   
    # Wait, D_n = {x in Z^n : sum(x_i) is even}. 
    # For x with all coords ±1: sum = (#plus) - (#minus) = 2*#plus - n. 
    # For n=11 (odd): 2*#plus - 11 is always odd. So indeed no all-±1 vector is in D_11.
    #
    # So the shortest vectors in D_11 are just the ±e_i ± e_j type, norm^2 = 2, count = 4*C(11,2) = 220.
    # But wait, we also have vectors like (1,1,1,1,0,...,0) type? No, those have norm^2=4 and even sum.
    # Actually let me reconsider. D_n contains all integer vectors with even coordinate sum.
    # Shortest: norm^2=1 would be ±e_i, but sum=±1 (odd), so NOT in D_n.
    # norm^2=2: ±e_i±e_j, sum=±2,0 (even), IN D_n. Count: for each pair {i,j}, 4 sign combos = 4. Total 4*C(11,2)=220.
    # norm^2=4: e.g., ±e_i±e_j±e_k±e_l (4 nonzero coords, each ±1, even count of -1s among the 4... 
    #   actually sum must be even. Sum of 4 terms each ±1: possible sums -4,-2,0,2,4 all even. So all 16 sign combos work.
    #   Count: C(11,4)*16 = 330*16 = 5280.
    #   Also 2 coords with value ±2: ±2e_i, sum=±2 (even), IN D_n. Count: 11*2 = 22.
    #   Also 1 coord ±2 and 2 coords ±1: sum = ±2±1±1, can be even or odd. E.g., 2+1-1=2 (even). 2-1-1=0 (even). 2+1+1=4 (even). 2-1+1=2. So all 8 sign combos for the two ±1's work with ±2e_i. Count: 11*2*10*... let me not go down this path.
    #
    # The kissing number of D_11 is 552. This comes from: 220 (norm^2=2) + 332 more? 
    # Actually, the kissing number of D_n for n≥10 is known. For D_11, it's 552.
    # The 552 vectors are: all 220 of type ±e_i±e_j, plus 332 of norm^2=4.
    # Hmm, let me just use a computational approach.

    # Generate all candidate vectors in D_11 with small norm
    candidates = []
    
    # Type 1: ±e_i ± e_j, norm^2 = 2
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Type 2: vectors with exactly 4 nonzero entries, each ±1, in D_11 (sum even - always true for 4 ±1's)
    # Count: C(11,4) * 16 = 330 * 16 = 5280
    from itertools import combinations
    for quad in combinations(range(D), 4):
        for signs in range(16):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(quad):
                v[idx] = 1 if (signs >> k) & 1 else -1
            candidates.append(v)
    
    # Type 3: ±2e_i, norm^2 = 4
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # We want max norm to be as small as possible to allow more vectors.
    # Use only norm^2 = 2 vectors first (220 of them), then add norm^2 = 4 vectors.
    
    # Filter: keep vectors with norm^2 <= 4
    mask = norms_sq <= 4
    candidates = candidates[mask]
    norms_sq = norms_sq[mask]
    n_cand = len(candidates)
    
    # Compute pairwise dot products and distances
    # dist^2(u,v) = |u|^2 + |v|^2 - 2*u.v
    # We need dist^2(u,v) >= max_norm_sq for all pairs
    # If we use only norm^2=2 vectors: max_norm_sq = 2, need dist^2 >= 2
    # dist^2 = 2 + 2 - 2*u.v = 4 - 2*u.v >= 2 => u.v <= 1
    # For norm^2=4 vectors: max_norm_sq = 4, need dist^2 >= 4
    # dist^2 = 2+4-2*u.v = 6-2*u.v >= 4 => u.v <= 1 (if u is norm^2=2, v is norm^2=4)
    # dist^2 = 4+4-2*u.v = 8-2*u.v >= 4 => u.v <= 2 (if both norm^2=4)
    
    # Strategy: Start with all norm^2=2 vectors (they form a valid set since |u.v|<=1 for distinct ±e_i±e_j)
    # Then greedily add norm^2=4 vectors that don't conflict.
    
    # Actually, let's just do a greedy maximum independent set on the conflict graph.
    # Conflict: two vectors conflict if dist^2 < max_norm_sq among all selected vectors.
    # This is tricky because max_norm_sq depends on which vectors we select.
    
    # Simpler approach: Fix max_norm_sq = 2 (use only norm^2=2 vectors). Then we can have at most... 
    # all 220? Let's check: for two distinct ±e_i±e_j vectors, what's the max |u.v|?
    # u = e_i + e_j, v = e_i + e_k (i!=j!=k): u.v = 1. dist^2 = 4-2=2. OK.
    # u = e_i + e_j, v = -e_i - e_j: u.v = -2. dist^2 = 4-(-4)=8. OK.
    # u = e_i + e_j, v = e_i - e_j: u.v = 1-1=0. dist^2=4. OK.
    # u = e_i + e_j, v = e_j + e_k: u.v = 1. dist^2 = 2. OK.
    # So all 220 norm^2=2 vectors are mutually at distance >= 2? Let's verify the worst case.
    # u = e_i+e_j, v = e_i+e_j (same vector, not distinct). For distinct vectors:
    # The minimum distance between distinct ±e_a±e_b vectors: 
    # If they share one index: u=e_i+e_j, v=e_i+e_k. u.v=1, dist^2=4-2=2. >= 2. OK.
    # If they share no indices: u=e_i+e_j, v=e_k+e_l. u.v=0, dist^2=4. OK.
    # If they're negatives: u=e_i+e_j, v=-e_i-e_j. u.v=-2, dist^2=4+4=8. OK.
    # So yes, all 220 form a valid kissing configuration with max_norm=2, min_dist=2.
    
    # Now can we add norm^2=4 vectors? If we add a vector w with |w|^2=4, then max_norm_sq becomes 4.
    # Then we need ALL pairs to have dist^2 >= 4.
    # For two norm^2=2 vectors: dist^2 >= 2, but we need >= 4. 
    # u=e_i+e_j, v=e_i+e_k: dist^2=2 < 4. CONFLICT!
    # So we can't just mix. If we include any norm^2=4 vector, we need to drop many norm^2=2 vectors.
    
    # Better: Try to maximize count. Let's try different max_norm_sq values.
    
    # Option A: max_norm_sq = 2, use only norm^2=2 vectors. Count = 220.
    # Option B: max_norm_sq = 4, use norm^2=4 vectors (and possibly some norm^2=2 if compatible).
    
    # For Option B with only norm^2=4 vectors: need dist^2 >= 4 for all pairs.
    # dist^2 = 8 - 2*u.v >= 4 => u.v <= 2.
    # For two distinct all-±1 (4 nonzero) vectors: u.v can be up to 4 (if same) or less.
    # Max u.v for distinct vectors: if they share 3 nonzero positions with same signs, u.v=3+(-1) or similar.
    # Let's just do greedy.
    
    # Let's try: start with norm^2=4 vectors, greedily add those with u.v <= 2 for all already selected.
    
    # Separate candidates by norm
    mask2 = norms_sq == 2
    mask4 = norms_sq == 4
    
    cands2 = candidates[mask2]
    cands4 = candidates[mask4]
    
    n2 = len(cands2)
    n4 = len(cands4)
    
    # Option A: just return all norm^2=2 vectors
    result_A = cands2.copy()
    
    # Option B: greedy on norm^2=4 vectors
    # For efficiency, precompute dot products
    if n4 > 0 and time.time() < deadline:
        # Compute dot product matrix for cands4
        # This could be large: n4 up to ~5300, so 5300x5300 = 28M entries, fine.
        dots4 = cands4 @ cands4.T
        
        # Greedy: repeatedly pick the vector with minimum number of conflicts
        selected4 = []
        remaining4 = set(range(n4))
        
        # Precompute conflict counts
        # Two vectors conflict if |u.v| > 2 (since u.v <= 2 needed, and by symmetry u.v >= -2 also fine since dist^2=8-2*u.v, if u.v=-2, dist^2=12>=4)
        # Actually we need u.v <= 2 (upper bound only, since dist^2 = 8-2*u.v, larger u.v means smaller dist)
        # u.v <= 2 is the condition.
        
        conflict_count4 = np.zeros(n4, dtype=np.int32)
        for i in range(n4):
            # Count how many j have dots4[i,j] > 2
            conflict_count4[i] = np.sum(dots4[i] > 2)
        
        while remaining4 and time.time() < deadline:
            # Pick vector with minimum conflict count among remaining
            min_conf = min(conflict_count4[list(remaining4)])
            # Among those with min conflict, pick first
            candidates_min = [j for j in remaining4 if conflict_count4[j] == min_conf]
            pick = candidates_min[0]
            selected4.append(pick)
            remaining4.discard(pick)
            
            # Update conflict counts for remaining: remove pick from their conflicts
            for j in remaining4:
                if dots4[j, pick] > 2:
                    conflict_count4[j] -= 1
        
        result_B = cands4[selected4]
    
    else:
        result_B = np.empty((0, D), dtype=np.int64)
    
    # Option C: Mix - start with norm^2=2, see if we can add norm^2=4
    # If we have norm^2=2 vectors, max_norm_sq=2 initially. Adding a norm^2=4 vector raises max to 4.
    # Then we need all norm^2=2 pairs to have dist^2>=4, which fails for adjacent ones.
    # So this is complex. Let's skip for now.
    
    # Choose the better option
    if len(result_B) > len(result_A):
        best = result_B
    else:
        best = result_A
    
    # Local search: try to add more vectors from the other norm class
    if time.time() < deadline:
        # Try to improve by adding vectors from the other class
        # If best is from norm^2=2 (size 220), try adding norm^2=4 vectors
        # But this changes max_norm_sq to 4, requiring re-validation.
        # Let's try a fresh greedy on ALL candidates with max_norm_sq=4.
        
        # For max_norm_sq=4: need dist^2 >= 4 for all pairs.
        # dist^2(u,v) = |u|^2 + |v|^2 - 2*u.v >= 4
        # For two norm^2=2: 4 - 2*u.v >= 4 => u.v <= 0
        # For norm^2=2 and norm^2=4: 6 - 2*u.v >= 4 => u.v <= 1
        # For two norm^2=4: 8 - 2*u.v >= 4 => u.v <= 2
        
        # Greedy on all candidates with these constraints
        n_all = n_cand
        if n_all <= 10000:
            dots_all = candidates @ candidates.T
            
            # Conflict matrix: conflict[i,j] = True if they violate the constraint
            # Constraint depends on norms of i and j
            conflict = np.zeros((n_all, n_all), dtype=bool)
            for i in range(n_all):
                for j in range(i+1, n_all):
                    ni = norms_sq[i]
                    nj = norms_sq[j]
                    uv = dots_all[i, j]
                    if ni == 2 and nj == 2:
                        if uv > 0:
                            conflict[i, j] = conflict[j, i] = True
                    elif (ni == 2 and nj == 4) or (ni == 4 and nj == 2):
                        if uv > 1:
                            conflict[i, j] = conflict[j, i] = True
                    elif ni == 4 and nj == 4:
                        if uv > 2:
                            conflict[i, j] = conflict[j, i] = True
            
            # Greedy minimum degree
            deg = np.sum(conflict, axis=1).astype(np.int32)
            remaining = set(range(n_all))
            selected = []
            
            while remaining and time.time() < deadline:
                min_deg = min(deg[list(remaining)])
                cands_min = [j for j in remaining if deg[j] == min_deg]
                pick = cands_min[0]
                selected.append(pick)
                remaining.discard(pick)
                for j in remaining:
                    if conflict[j, pick]:
                        deg[j] -= 1
            
            result_C = candidates[selected]
            
            if len(result_C) > len(best):
                best = result_C
    
    # Verify the result
    if len(best) > 0:
        best_norms_sq = np.sum(best * best, axis=1)
        max_norm_sq = np.max(best_norms_sq)
        dots = best @ best.T
        dists_sq = best_norms_sq[:, None] + best_norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = np.min(dists_sq)
        if min_dist_sq < max_norm_sq - 1e-9:
            # Invalid, fall back to norm^2=2 only
            best = cands2
    
    return best
