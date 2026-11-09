"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import heapq

def construct():
    # Strategy: Try to find a large independent set in the conflict graph of weight 4 vectors.
    # If it beats the baseline (220), return it. Otherwise, return the baseline weight 2 set.
    
    # Baseline: Weight 2 vectors
    baseline_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                baseline_vectors.append(v)
    baseline_count = len(baseline_vectors)
    
    # Candidate: Weight 4 vectors
    # Generate all vectors with exactly 4 non-zero entries of value 1 or -1
    # Norm squared = 4. Max norm = 2.
    # Constraint: min pairwise distance >= 2.
    # d^2 >= 4.
    
    candidates = []
    indices = range(D)
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(v)
            
    # Convert to numpy for faster distance calculation if needed, 
    # but for 5280 vectors, O(N^2) is ~27M ops, which is fine in Python/NumPy.
    candidates_np = np.array(candidates, dtype=np.int8)
    N = len(candidates_np)
    
    # Greedy Independent Set
    # We want to select as many as possible such that pairwise distance squared >= 4.
    # Sort candidates by "popularity" or just pick arbitrarily?
    # A common heuristic for MIS is to pick the node with the fewest conflicts (or most, depending on strategy).
    # Let's try a simple greedy: 
    # 1. Calculate all pairwise distances? No, too memory heavy for 5280^2 floats? 27M floats = 200MB. OK.
    # But we can do it on the fly.
    
    # Better heuristic: 
    # Sort vectors by some metric? 
    # Let's just use a random greedy or a degree-based greedy.
    # Since we don't have time to build the full graph explicitly in a complex way, 
    # let's try a simple iterative selection.
    
    # To improve performance, we can group by support.
    # Vectors with disjoint supports are at distance sqrt(8) > 2. 
    # Vectors with overlapping supports need checking.
    
    # Let's try a simple greedy:
    # 1. Start with all candidates.
    # 2. Pick one, add to set.
    # 3. Remove all candidates that conflict with it.
    # 4. Repeat.
    
    # To make it fast, we can use a list of active candidates.
    # N=5280.
    
    # Optimization: 
    # Two vectors conflict if their Hamming distance is small?
    # d^2 = sum (v_i - w_i)^2.
    # If v, w are weight 4.
    # If they are identical, d=0.
    # If they differ in 1 sign, d^2 = (2)^2 = 4. (OK, >=4)
    # If they differ in 2 signs, d^2 = 4+4=8. (OK)
    # If they share 3 indices:
    #   e.g., 1110 and 1101. Diff: 0011. d^2=2. (Conflict)
    #   e.g., 1110 and -1110. Diff: 2220. d^2=12. (OK)
    #   e.g., 1110 and 11-10. Diff: 0020. d^2=4. (OK)
    #   e.g., 1110 and -1-110. Diff: 2200. d^2=8. (OK)
    #   e.g., 1110 and -11-10. Diff: 2020. d^2=8. (OK)
    #   e.g., 1110 and 1-1-10. Diff: 0220. d^2=8. (OK)
    #   e.g., 1110 and -1-1-10. Diff: 2220. d^2=12. (OK)
    #   Wait, let's re-evaluate "share 3 indices".
    #   Support A = {1,2,3,4}, Support B = {1,2,3,5}.
    #   Indices 1,2,3 common.
    #   v[4]=s4, w[4]=0. v[5]=0, w[5]=s5.
    #   d^2 = (v1-w1)^2 + ... + (v4-0)^2 + (0-w5)^2.
    #   If v1=w1, v2=w2, v3=w3: d^2 = 1 + 1 = 2. CONFLICT.
    #   If v1=-w1, v2=w2, v3=w3: d^2 = 4+1+1=6. OK.
    
    # So, conflicts happen primarily when supports overlap significantly and signs are similar.
    
    # Let's implement a greedy that prefers vectors with "unique" signs or supports?
    # Actually, just random greedy might work well enough.
    
    # To ensure determinism and speed:
    # We will use a simple greedy: 
    # 1. Shuffle candidates.
    # 2. For each candidate, if it doesn't conflict with any already selected, select it.
    
    # This is O(N^2) in worst case. 5280^2 = 27M. 
    # In Python, 27M iterations might take ~5-10 seconds. Acceptable.
    
    import random
    random.seed(42)
    
    # We can optimize by checking conflicts only against selected ones.
    selected = []
    # To speed up, we can maintain a set of "forbidden" patterns?
    # Or just loop.
    
    # Let's try to be smarter.
    # Group by support.
    # There are 330 supports.
    # For each support, there are 16 vectors.
    # Vectors with same support:
    #   d^2 between any two distinct is at least 4 (if 1 sign flip) or more.
    #   So we can select ALL 16 vectors for a given support?
    #   Check: v=1111, w=-1111. d^2=16. OK.
    #   v=1111, w=111-1. d^2=4. OK.
    #   So for a fixed support, all 16 are mutually valid!
    
    # Now check between different supports.
    # Let S1, S2 be supports.
    # Intersection size k.
    # If k=0 (disjoint): d^2 = 4+4=8. OK.
    # If k=1: 
    #   Common index i.
    #   v_i, w_i in {1,-1}.
    #   Other 3 indices in v, 3 in w are 0.
    #   d^2 = (v_i-w_i)^2 + 3*1^2 + 3*1^2 = (v_i-w_i)^2 + 6.
    #   Min d^2 = 0 + 6 = 6 (if v_i=w_i). OK.
    #   So supports intersecting in 1 are always OK.
    
    # If k=2:
    #   Common indices i, j.
    #   d^2 = (v_i-w_i)^2 + (v_j-w_j)^2 + 2*1^2 + 2*1^2 = (v_i-w_i)^2 + (v_j-w_j)^2 + 4.
    #   Min d^2 = 4 (if v_i=w_i, v_j=w_j). OK.
    
    # If k=3:
    #   Common indices i, j, k.
    #   d^2 = (v_i-w_i)^2 + (v_j-w_j)^2 + (v_k-w_k)^2 + 1^2 + 1^2.
    #   Min d^2 = 2 (if all signs match). CONFLICT.
    
    # If k=4 (same support):
    #   Min d^2 = 4. OK.
    
    # Conclusion:
    # We can select ALL 16 vectors for any support.
    # We must ensure that for any two selected supports S1, S2 with |S1 \cap S2| = 3,
    # the signs on the intersection are NOT all identical.
    # i.e., there must be at least one sign flip on the intersection.
    
    # This is a constraint satisfaction problem on the 330 supports.
    # We want to select a subset of supports such that for any pair with intersection 3, the sign vectors are not identical on the intersection.
    # Actually, we can select ALL supports, and just pick the "best" sign combination for each?
    # No, if we select all 330 supports, we have 330*16 = 5280 vectors.
    # But we can't select all 16 for a support if it conflicts with neighbors.
    # Wait, if we select support S1 and S2 (intersect 3), we just need to ensure we don't pick the "bad" pairs.
    # Actually, we can pick a subset of the 16 vectors for each support.
    
    # Heuristic:
    # For each support, pick one representative vector?
    # If we pick 1 vector per support, total 330.
    # 330 > 220.
    # Let's check if we can pick 1 vector per support such that all pairwise distances >= 2.
    # This is equivalent to finding a code.
    
    # Let's try to construct such a set.
    # Randomly assign signs to each support.
    # Check conflicts.
    # If conflict, flip a sign.
    
    # This is much faster than checking 5280 vectors.
    # We only have 330 supports.
    
    supports = list(itertools.combinations(indices, 4))
    num_supports = len(supports)
    
    # Assign random signs
    signs = [np.random.choice([1, -1], size=4) for _ in range(num_supports)]
    
    # Build the set of vectors
    # vector[i] corresponds to supports[i]
    
    # Check for conflicts between all pairs of supports with intersection >= 3
    # There are ~330*330/2 = 54k pairs.
    # For each pair, check intersection size.
    # If size 3, check if signs are identical on the intersection.
    
    # Let's refine:
    # We can select MULTIPLE vectors per support.
    # But let's start with 1 per support to see if we can beat 220.
    # If 1 per support works, we have 330.
    # Can we do 2 per support?
    # If we pick v and w from same support, d(v,w)>=4.
    # But v might conflict with u from neighbor support.
    
    # Let's try to maximize the count.
    # Start with 1 vector per support (330 vectors).
    # Verify validity.
    
    selected_vectors = []
    for i, supp in enumerate(supports):
        v = [0]*D
        for idx, sign in zip(supp, signs[i]):
            v[idx] = sign
        selected_vectors.append(v)
        
    # Verify
    # This is O(330^2) = 100k checks. Fast.
    valid = True
    sv = np.array(selected_vectors, dtype=np.int8)
    # d^2 matrix
    # d^2 = sum(sv^2) + sum(sv^2) - 2 sv@sv.T
    # sum(sv^2) = 4 for all.
    # d^2 = 8 - 2 * dot
    # Need d^2 >= 4 => 8 - 2 dot >= 4 => 2 dot <= 4 => dot <= 2.
    
    dots = sv @ sv.T
    # Diagonal is 4.
    # Check off-diagonal.
    mask = ~np.eye(num_supports, dtype=bool)
    if np.any(dots[mask] > 2):
        valid = False
        
    if valid:
        return np.array(selected_vectors, dtype=np.int64)
        
    # If invalid, fallback to baseline
    return np.array(baseline_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
