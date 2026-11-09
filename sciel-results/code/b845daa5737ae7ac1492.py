import time
import numpy as np
from itertools import combinations

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def time_remaining():
        return deadline - time.time()

    # Strategy: Construct a kissing configuration by combining known optimal sub-configurations.
    # Dimension 11 can be decomposed as 8 + 3.
    # The E8 lattice has a kissing number of 240.
    # The A3 lattice (in 4D, but we can embed in 3D? No, A3 is 3D roots of G2? No, G2 is 2D. A3 is 3D? No, A_n is n+1 dim. A3 is 4D roots? No, A3 roots are in 4D hyperplane sum=0. 
    # Let's look at the dimensions.
    # E8 is 8D. Kissing number 240.
    # Remaining dimension is 11 - 8 = 3.
    # In 3D, the kissing number is 12 (octahedron + cube? No, 12 vectors: 6 axis, 6 face diagonals? No. 
    # 3D kissing number is 12. Vectors: (±1,±1,0), (±1,0,±1), (0,±1,±1). Norm^2=2.
    # If we take E8 vectors (padded with 0s in last 3 dims) and A3 vectors (padded with 0s in first 8 dims),
    # do they satisfy the distance constraint?
    # Let S = S_E8 ∪ S_A3.
    # Vectors in S_E8 have norm^2 = 4 (in 8D part) or 2 (for the 112 type? No, E8 roots have norm^2=2).
    # E8 roots:
    # 112 vectors of type (±1, ±1, 0, ..., 0). Norm^2 = 2.
    # 128 vectors of type (±1/2, ..., ±1/2) with even number of minus signs. These are not integer.
    # Wait, the task requires INTEGER vectors.
    # The standard E8 lattice contains integer vectors and half-integer vectors.
    # If we restrict to integer vectors in E8, we only have the 112 vectors of type (±1, ±1, 0...0).
    # Let's check their properties.
    # Norm^2 = 2.
    # Distance between two such vectors:
    # If they share no indices: diff is (±1, ±1, ±1, ±1, 0...) -> Norm^2 = 4.
    # If they share 1 index: e.g., (1,1,0...) and (1,-1,0...). Diff (0,2,0...) -> Norm^2 = 4.
    # If they share 2 indices: e.g., (1,1,0...) and (1,1,0...). Same vector (not allowed in set).
    # So for the 112 integer E8 roots, min dist^2 is 4. Max norm^2 is 2.
    # Condition: min dist >= max norm -> sqrt(4) >= sqrt(2) -> 2 >= 1.414. Valid.
    # So we have 112 vectors in 8D.
    
    # Now for the remaining 3 dimensions.
    # We want integer vectors in R^3 (padded to R^11) with norm^2 <= 2 (to match E8 max norm? 
    # If we use vectors with norm^2 > 2, the global max norm increases, tightening the distance constraint for E8 vectors.
    # If max norm^2 becomes 4, E8 vectors (norm^2=2) must be dist^2 >= 4 from each other. They are.
    # But they must also be dist^2 >= 4 from the new 3D vectors.
    # An E8 vector v_E (non-zero in first 8, 0 in last 3) and a 3D vector v_3 (0 in first 8, non-zero in last 3).
    # dist^2 = ||v_E||^2 + ||v_3||^2 = 2 + ||v_3||^2.
    # If max norm^2 is R^2, we need 2 + ||v_3||^2 >= R^2.
    # If we pick 3D vectors with norm^2 = 2 (e.g., (1,1,0)), then R^2 = max(2, 2) = 2.
    # Then we need min dist^2 >= 2.
    # dist^2 between E8 and 3D is 2+2=4 >= 2. OK.
    # dist^2 between E8 and E8 is 4 >= 2. OK.
    # dist^2 between 3D and 3D must be >= 2.
    # In 3D, what is the max number of integer vectors with norm^2=2 and min dist^2 >= 2?
    # Vectors with norm^2=2 in Z^3: (±1, ±1, 0), (±1, 0, ±1), (0, ±1, ±1). Total 12.
    # Let's check min dist^2 among these 12.
    # Take (1,1,0) and (1,-1,0). Diff (0,2,0). Norm^2=4.
    # Take (1,1,0) and (-1,1,0). Diff (2,0,0). Norm^2=4.
    # Take (1,1,0) and (1,0,1). Diff (0,1,-1). Norm^2=2.
    # So min dist^2 is 2.
    # Condition: min dist^2 >= R^2 = 2. So 2 >= 2 is valid.
    # Thus, we can take all 12 vectors of norm^2=2 in 3D.
    
    # Total count: 112 (from E8 part) + 12 (from 3D part) = 124.
    # Current score is 1.5773. Seed is likely 1.0. 124 vectors?
    # Wait, the current program score 1.5773 implies it found ~1.5773 * seed_size vectors?
    # If seed size is 1, score is k. So current program finds ~1.5 vectors? No, that's impossible for a "kissing number" task.
    # Let's re-read: "score is k". "search score is value/seed".
    # If seed program returns 1 vector, score is 1/1 = 1.0.
    # If current program returns 157 vectors, score is 157/100? No.
    # "value/seed". If seed program returns a fixed number, say 100, and current returns 157, score is 1.57.
    # Or maybe the seed program is the E8+A3 construction mentioned in inspirations?
    # Inspiration 2 says: "deterministic construction of the E8 ⊕ A3 lattice kissing configuration (252 vectors)".
    # If the seed program returns 252 vectors, and score is 1.5773, then 252 * 1.5773 ≈ 397?
    # Or maybe the seed program is poor?
    # Let's assume the "seed" is a baseline. The current program scores 1.5773.
    # If I can construct 252 vectors, and the seed is 252, score is 1.0.
    # If the seed is smaller, say 100, and I find 252, score is 2.52.
    # Let's look at the inspirations.
    # Inspiration 2 (score 1.2682) uses E8+A3 (252 vectors).
    # Current program (score 1.5773) uses greedy on integer vectors.
    # This implies the current program finds MORE than 252 vectors? Or the seed is smaller.
    # If current finds 1.5773 * seed_count, and inspo 2 finds 1.2682 * seed_count.
    # If inspo 2 returns 252, then 252 = 1.2682 * seed_count -> seed_count ≈ 198.
    # Then current program returns 1.5773 * 198 ≈ 312 vectors.
    # Can we find > 312 vectors?
    # The kissing number in 11D is known to be 252? No, that's for 8+3?
    # Actually, the kissing number for dimension 11 is not exactly known, but bounds are known.
    # However, we are restricted to INTEGER vectors.
    # The set of integer vectors is a sublattice of the full space.
    # The "kissing number" for the integer lattice Z^11 might be different from the general kissing number.
    # Let's check the constraints again.
    # "Construct nonzero integer vectors... min pairwise distance >= max vector norm."
    # If we use the 112 E8 integer roots (norm^2=2) and 12 A3 integer roots (norm^2=2), we get 124.
    # This is likely less than 312.
    # Why does the current program get 1.5773?
    # Maybe it uses vectors with larger norms?
    # If we allow norm^2=4 vectors, we can have more candidates.
    # My previous analysis in the current program code suggested trying norm^2=4.
    # Let's try to construct a larger set using a mix or higher norms.
    
    # Alternative: Use the full E8 lattice?
    # E8 has 240 roots. Only 112 are integer. The other 128 are half-integer.
    # We cannot use half-integer vectors.
    # So we are limited to 112 from the E8 part if we stick to norm^2=2.
    # What if we use norm^2=4 vectors in the 8D part?
    # Vectors in Z^8 with norm^2=4.
    # Types:
    # 1. (±2, 0, ..., 0): 8 * 2 = 16 vectors.
    # 2. (±1, ±1, ±1, ±1, 0, ...): C(8,4) * 2^4 = 70 * 16 = 1120 vectors.
    # Total 1136 vectors in 8D with norm^2=4.
    # If we pick a subset of these with min dist^2 >= 4 (since max norm^2=4),
    # and combine with 3D vectors of norm^2 <= 4.
    # In 3D, max norm^2=4.
    # Vectors in Z^3 with norm^2 <= 4:
    # Norm^2=1: (±1,0,0) etc. 6 vectors.
    # Norm^2=2: (±1,±1,0) etc. 12 vectors.
    # Norm^2=3: (±1,±1,±1) 8 vectors.
    # Norm^2=4: (±2,0,0) 6 vectors. (±1,±1,±1 is norm^2=3). (±1,±1,0 is norm^2=2).
    # Wait, (2,0,0) is norm^2=4.
    # Total candidates in 3D: 6+12+8+6 = 32.
    # We need min dist^2 >= 4 for all pairs in the combined set.
    # Cross term: dist^2 = ||v8||^2 + ||v3||^2 >= 4 + 1 = 5 >= 4. OK.
    # So we just need to find max independent sets in Z^8 (norm^2=4, min dist^2>=4) and Z^3 (norm^2<=4, min dist^2>=4).
    
    # Let's write a greedy solver for this specific case.
    # 1. Generate all Z^8 vectors with norm^2=4.
    # 2. Generate all Z^3 vectors with norm^2<=4.
    # 3. Greedy select from Z^8.
    # 4. Greedy select from Z^3.
    # 5. Combine.
    
    # Generation Z^8 norm^2=4
    cand_8 = []
    # Type 1: One ±2
    for i in range(8):
        for s in [1, -1]:
            v = np.zeros(8, dtype=np.int64)
            v[i] = 2*s
            cand_8.append(v)
    # Type 2: Four ±1
    for idx in combinations(range(8), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(8, dtype=np.int64)
            for k, i in enumerate(idx):
                v[i] = signs[k]
            cand_8.append(v)
            
    # Generation Z^3 norm^2<=4
    cand_3 = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            for z in range(-2, 3):
                if x==0 and y==0 and z==0: continue
                n2 = x*x + y*y + z*z
                if n2 <= 4:
                    cand_3.append(np.array([x,y,z], dtype=np.int64))
                    
    # Greedy selection function
    def greedy_select(candidates, min_dist_sq):
        selected = []
        # Sort by norm? Or just iterate.
        # To be faster, we can randomize or use a heuristic.
        # Let's try sorting by norm descending? No, usually smaller norms are "better" for packing?
        # But here all have fixed or bounded norms.
        # Let's just iterate.
        for v in candidates:
            compatible = True
            for u in selected:
                diff = v - u
                if np.dot(diff, diff) < min_dist_sq:
                    compatible = False
                    break
            if compatible:
                selected.append(v)
        return selected
        
    sel_8 = greedy_select(cand_8, 4)
    sel_3 = greedy_select(cand_3, 4)
    
    # Pad and combine
    result_8 = np.zeros((len(sel_8), 11), dtype=np.int64)
    for i, v in enumerate(sel_8):
        result_8[i, :8] = v
        
    result_3 = np.zeros((len(sel_3), 11), dtype=np.int64)
    for i, v in enumerate(sel_3):
        result_3[i, 8:] = v
        
    result = np.vstack([result_8, result_3])
    
    # Validate
    if len(result) > 1:
        norms_sq = np.sum(result**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        min_dist_sq = np.inf
        n = len(result)
        for i in range(n):
            diffs = result[i+1:] - result[i]
            dists_sq = np.sum(diffs**2, axis=1)
            if len(dists_sq) > 0:
                min_d = np.min(dists_sq)
                if min_d < min_dist_sq:
                    min_dist_sq = min_d
        if min_dist_sq < max_norm_sq:
            # Fallback to previous logic or simple set
            # This shouldn't happen if greedy is correct
            return result[:1]
            
    return result

# Need to import product
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    D = 11

    # Strategy: Decompose 11D into 8D + 3D.
    # Use integer vectors in Z^8 with norm^2=4 and Z^3 with norm^2<=4.
    # Constraint: min dist^2 >= 4 (since max norm^2 will be 4).
    
    # Generate candidates in 8D
    cand_8 = []
    # Type 1: One coordinate is ±2
    for i in range(8):
        for s in [1, -1]:
            v = np.zeros(8, dtype=np.int64)
            v[i] = 2 * s
            cand_8.append(v)
            
    # Type 2: Four coordinates are ±1
    for idx in combinations(range(8), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(8, dtype=np.int64)
            for k, i in enumerate(idx):
                v[i] = signs[k]
            cand_8.append(v)
            
    # Generate candidates in 3D
    cand_3 = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            for z in range(-2, 3):
                if x == 0 and y == 0 and z == 0:
                    continue
                n2 = x*x + y*y + z*z
                if n2 <= 4:
                    cand_3.append(np.array([x, y, z], dtype=np.int64))
                    
    def greedy_select(candidates, min_dist_sq):
        selected = []
        # Optimization: Vectorize distance checks? 
        # Candidates ~1136, selected grows. 
        # 1136 * 1136 / 2 checks is ~600k. Each check is dot product of 8 or 3 dims. Fast enough.
        for v in candidates:
            compatible = True
            for u in selected:
                diff = v - u
                if np.dot(diff, diff) < min_dist_sq:
                    compatible = False
                    break
            if compatible:
                selected.append(v)
        return selected
        
    sel_8 = greedy_select(cand_8, 4)
    sel_3 = greedy_select(cand_3, 4)
    
    # Pad to 11D
    result_8 = np.zeros((len(sel_8), 11), dtype=np.int64)
    if len(sel_8) > 0:
        result_8[:, :8] = np.array(sel_8)
        
    result_3 = np.zeros((len(sel_3), 11), dtype=np.int64)
    if len(sel_3) > 0:
        result_3[:, 8:] = np.array(sel_3)
        
    if len(sel_8) == 0 and len(sel_3) == 0:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    if len(sel_8) == 0:
        result = result_3
    elif len(sel_3) == 0:
        result = result_8
    else:
        result = np.vstack([result_8, result_3])
        
    # Final Validation
    if len(result) > 1:
        norms_sq = np.sum(result**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        min_dist_sq = np.inf
        n = len(result)
        for i in range(n):
            diffs = result[i+1:] - result[i]
            dists_sq = np.sum(diffs**2, axis=1)
            if len(dists_sq) > 0:
                min_d = np.min(dists_sq)
                if min_d < min_dist_sq:
                    min_dist_sq = min_d
        
        if min_dist_sq < max_norm_sq:
            return result[:1]
            
    return result
