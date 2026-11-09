import time
import numpy as np
from itertools import product

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def time_remaining():
        return deadline - time.time()

    # 1. Generate candidate pool: All non-zero integer vectors in R^11 with norm^2 <= 4.
    # This includes:
    # - Norm^2 = 1: 22 vectors (±1 at one pos)
    # - Norm^2 = 2: 110 vectors (±1 at two pos)
    # - Norm^2 = 4: 220 vectors (±2 at one pos) + 440 vectors (±2, ±1, 0... wait, 2^2+1^2=5 > 4)
    # Actually, norm^2 <= 4 implies:
    # - 1^2 + 0... : 22 vectors (norm 1)
    # - 1^2 + 1^2 : 110 vectors (norm sqrt(2))
    # - 2^2 + 0... : 22 vectors (norm 2)
    # - 2^2 + 1^2 = 5 > 4, so no mixed 2 and 1.
    # Total candidates: 22 + 110 + 22 = 154.
    
    # Wait, the task says "minimum pairwise distance is at least the maximum vector norm".
    # If we pick vectors of different norms, the "maximum vector norm" is determined by the largest one.
    # If we pick a vector of norm 2, all other vectors must be at distance >= 2 from it.
    # Distance between v and w is ||v-w||.
    # If v, w are orthogonal, ||v-w||^2 = ||v||^2 + ||w||^2.
    # We need ||v-w|| >= max(||v||, ||w||).
    # This is always true for orthogonal vectors since ||v-w||^2 = ||v||^2 + ||w||^2 >= max(||v||^2, ||w||^2).
    # However, if they are NOT orthogonal, the distance might be small.
    # Specifically, we want to maximize k such that for all i != j, ||v_i - v_j|| >= R, where R = max_i ||v_i||.
    
    # Let's stick to a uniform norm to simplify the constraint to "mutually orthogonal" or "sufficiently separated".
    # If we use only norm^2 = 2 vectors (110 candidates), then R = sqrt(2).
    # Condition: ||v_i - v_j|| >= sqrt(2) => ||v_i - v_j||^2 >= 2.
    # For v, w with ||v||^2=||w||^2=2, ||v-w||^2 = 4 - 2<v,w>.
    # We need 4 - 2<v,w> >= 2 => 2<v,w> <= 2 => <v,w> <= 1.
    # Since v, w are integer vectors with norm^2=2, they have two ±1s.
    # <v,w> can be -2, -1, 0, 1, 2.
    # If <v,w> = 2, they are identical (not allowed in set).
    # If <v,w> = 1, distance^2 = 2, which is >= 2. So this is ALLOWED.
    # If <v,w> = -1, distance^2 = 6, allowed.
    # If <v,w> = 0, distance^2 = 4, allowed.
    # If <v,w> = -2, distance^2 = 8, allowed.
    # So for norm^2=2 vectors, the constraint is simply v != w.
    # This means we can take ALL 110 vectors of norm^2=2?
    # Let's check: max norm is sqrt(2). Min dist is sqrt(2) (when <v,w>=1).
    # Yes! The set of all 110 vectors with norm^2=2 is a valid kissing configuration?
    # Wait, is it a "kissing configuration"? Usually kissing number counts spheres of radius 1/2 touching a central sphere of radius 1/2? No, standard definition:
    # "kissing number in dimension d is the maximum number of non-overlapping unit spheres that can touch a central unit sphere."
    # This is equivalent to finding max number of vectors v_i with ||v_i|| = 1 such that ||v_i - v_j|| >= 1 for all i != j.
    # The task says: "Normalizing each vector to norm 2 certifies a kissing configuration... minimum pairwise distance is at least the maximum vector norm."
    # If we normalize to norm 2, then R=2. Condition: ||v_i - v_j|| >= 2.
    # If we pick vectors of norm 2, we need ||v_i - v_j|| >= 2.
    # As derived above, for norm^2=2 vectors, min dist is sqrt(2) which is < 2.
    # So we CANNOT use norm^2=2 vectors if the target norm is 2.
    
    # The task says: "Construct nonzero integer vectors... whose minimum pairwise distance is at least the maximum vector norm."
    # It does NOT say we must normalize to norm 2 in the output. It says "Normalizing... certifies...".
    # The score is k (number of vectors).
    # We want to maximize k.
    # Let R = max ||v_i||. We need min_{i!=j} ||v_i - v_j|| >= R.
    
    # Strategy: Use vectors of norm 1? (22 vectors). R=1. Min dist between distinct unit integer vectors is sqrt(2) >= 1.
    # So we can take all 22 vectors of norm 1. Score = 22.
    
    # Can we do better?
    # What if we mix norms?
    # If we include a vector of norm 2, R becomes 2.
    # Then all pairs must be dist >= 2.
    # If we have v with ||v||=2, and w with ||w||=1.
    # ||v-w||^2 = 5 - 2<v,w>.
    # Min possible ||v-w||^2 is when <v,w> is max.
    # v has one ±2. w has one ±1.
    # If they share the coordinate, <v,w> = ±2.
    # If <v,w>=2, ||v-w||^2 = 1. Dist=1 < 2. Invalid.
    # If <v,w>=-2, ||v-w||^2 = 9. Dist=3 >= 2. Valid.
    # If they don't share coordinate, <v,w>=0. Dist=sqrt(5) >= 2. Valid.
    # So we can mix norm 1 and norm 2 vectors IF we ensure no "close" pairs.
    
    # However, the known kissing number in 11D is 252.
    # The E8⊕A3 construction gives 252 vectors of norm 2 (in the lattice sense, but scaled).
    # Let's look at the E8⊕A3 vectors provided in the seed.
    # E8 part: norm^2 = 4 or 8?
    # Type 1: two ±2, six 0s. Norm^2 = 8.
    # Type 2: all ±1. Norm^2 = 8.
    # So E8 vectors have norm sqrt(8) = 2*sqrt(2).
    # A3 part: two ±2, one 0. Norm^2 = 8.
    # So all 252 vectors have norm sqrt(8).
    # R = sqrt(8).
    # Condition: min dist >= sqrt(8).
    # For E8 lattice, min distance between distinct vectors is 2 (sqrt(4)).
    # Wait, E8 lattice min dist is 2.
    # If we take vectors from E8 lattice, ||v||=2sqrt(2).
    # Dist between two distinct E8 vectors can be 2.
    # 2 < 2sqrt(2) ~ 2.82.
    # So the standard E8 lattice vectors do NOT satisfy "min dist >= max norm" if max norm is 2sqrt(2).
    
    # Let's re-read the seed code.
    # Seed code returns 252 vectors.
    # It checks: `if min_dist_sq >= max_norm_sq`.
    # In the seed, max_norm_sq is 8 (for E8 part) or 8 (for A3 part).
    # Wait, A3 vectors in seed: `v[i] = 2*si, v[j] = 2*sj`. Norm^2 = 4+4=8.
    # E8 Type 1: `v[i]=2*si, v[j]=2*sj`. Norm^2 = 8.
    # E8 Type 2: all ±1. Norm^2 = 8.
    # So all vectors have norm^2 = 8.
    # The seed claims this is valid.
    # This implies min_dist_sq >= 8.
    # Is the min distance in E8⊕A3 configuration >= sqrt(8)?
    # E8 lattice min dist is 2 (dist^2=4).
    # But we are not taking the whole lattice, we are taking a specific subset (the "kissing" vectors).
    # The kissing vectors of E8 are the 240 vectors of norm^2=2 (in standard E8 scaling where min dist is sqrt(2)).
    # Here, the vectors are scaled by sqrt(2) or 2?
    # Standard E8 roots have norm^2=2. There are 240.
    # The seed uses vectors of norm^2=8. This is scaling by 2.
    # If we scale by 2, distances also scale by 2.
    # Min dist in standard E8 roots is sqrt(2).
    # Scaled by 2, min dist is 2*sqrt(2) = sqrt(8).
    # Max norm is 2*sqrt(2) = sqrt(8).
    # So min dist = max norm. This satisfies "at least".
    # So the E8⊕A3 construction is valid and yields 252.
    
    # Can we do better than 252?
    # The kissing number in 11D is known to be 252.
    # So 252 is likely the maximum.
    # The current program scores 1.1455.
    # The seed score is 1.0 (252 vectors? No, seed score is defined relative to a baseline).
    # "score is k". "search score is value/seed".
    # If seed program returns 252, and score is 1.0, then baseline is 252.
    # Current program returns 252?
    # Let's check current program output count.
    # E8: 112 + 128 = 240.
    # A3: 12.
    # Total 252.
    # So current program returns 252.
    # Score 1.1455 implies 252 / seed_value = 1.1455?
    # Or is the "seed" in the prompt the baseline program which might return less?
    # "1.0 equals the seed program".
    # If current program is the one shown, and it scores 1.1455, it must be returning MORE than the seed program.
    # But the code shown IS the seed program?
    # "CURRENT PROGRAM (score 1.1455) and its strategy:"
    # The code block is the current program.
    # If the code block returns 252, and it scores 1.1455, then the "seed program" (baseline) must return ~220.
    # Or the scoring is different.
    # "score is k".
    # If k=252, and score=1.1455, then seed_k = 252 / 1.1455 = 220.
    # So the baseline returns 220 vectors.
    # The current program returns 252.
    
    # Inspiration 1 (score 1.5773) used a greedy solver and got higher score.
    # This implies it found MORE than 252 vectors?
    # Or the baseline is different.
    # If kissing number is 252, you cannot get more than 252.
    # Unless the "kissing number" definition in the task is different.
    # "minimum pairwise distance is at least the maximum vector norm".
    # This is the standard definition.
    # Max kissing number in 11D is 252.
    # How did Inspiration 1 get 1.5773?
    # 252 * 1.5773 / 1.1455 = 345?
    # This is impossible if the bound is 252.
    # Unless... the vectors don't have to be distinct? No, "pairwise distance" implies distinct.
    # Unless the "maximum vector norm" is small?
    # If we use vectors of norm 1, we get 22.
    # If we use vectors of norm 2 (sqrt(4)), we can potentially get more?
    # No, larger norms make the distance constraint harder relative to the norm.
    # Wait.
    # Condition: min_dist >= max_norm.
    # If we use small vectors, max_norm is small, so min_dist requirement is small.
    # Example: Norm 1 vectors. Max norm 1. Min dist must be >= 1.
    # Integer vectors of norm 1: e_i and -e_i.
    # Dist between e_i and e_j (i!=j) is sqrt(2) >= 1.
    # Dist between e_i and -e_i is 2 >= 1.
    # So all 22 are valid.
    
    # What if we use norm 2 vectors (e.g., 2*e_i)?
    # Max norm 2. Min dist must be >= 2.
    # Vectors: ±2e_i.
    # Dist between 2e_i and 2e_j is 2*sqrt(2) ~ 2.82 >= 2.
    # Dist between 2e_i and -2e_i is 4 >= 2.
    # So all 22 are valid.
    
    # Can we mix?
    # If we include a vector of norm 10, max norm is 10.
    # Then all other vectors must be at least 10 away.
    # This restricts the set heavily.
    
    # So, to maximize k, we should use SMALL norms.
    # The smallest non-zero integer vector norm is 1.
    # We found 22 vectors of norm 1.
    
    # Can we find more vectors with norm 1? No, only ±e_i.
    
    # Can we find vectors with norm sqrt(2)? (e_i ± e_j).
    # There are 110 such vectors.
    # If we use ONLY these, max norm is sqrt(2).
    # Min dist between two distinct vectors of norm sqrt(2)?
    # v = e1+e2, w = e1-e2. Dist^2 = (e1+e2 - (e1-e2))^2 = (2e2)^2 = 4. Dist=2.
    # 2 >= sqrt(2). Valid.
    # v = e1+e2, w = e1+e3. Dist^2 = (e2-e3)^2 = 2. Dist=sqrt(2).
    # sqrt(2) >= sqrt(2). Valid.
    # So ALL 110 vectors of norm sqrt(2) form a valid set?
    # Max norm = sqrt(2). Min dist = sqrt(2).
    # Yes!
    # So we can get k=110.
    
    # Can we get more by mixing norm 1 and norm sqrt(2)?
    # If we mix, max norm is sqrt(2).
    # Min dist must be >= sqrt(2).
    # We already have 110 vectors of norm sqrt(2) that satisfy this among themselves.
    # Can we add norm 1 vectors?
    # Let v be a norm 1 vector (e.g., e1).
    # Let w be a norm sqrt(2) vector (e.g., e1+e2).
    # Dist^2 = ||e1 - (e1+e2)||^2 = ||-e2||^2 = 1.
    # Dist = 1.
    # Requirement: Dist >= sqrt(2).
    # 1 < 1.414. Invalid.
    # So we CANNOT mix norm 1 and norm sqrt(2) if we want to keep the norm sqrt(2) vectors.
    
    # What if we use ONLY norm 1 vectors? k=22.
    # What if we use ONLY norm sqrt(2) vectors? k=110.
    # What if we use ONLY norm 2 vectors (2e_i, 2e_j...)?
    # Norm 2 vectors: 22 of type 2e_i.
    # Also vectors like e_i+e_j+e_k? No, norm sqrt(3).
    # Vectors with norm 2:
    # 1. ±2e_i (22 vectors).
    # 2. No others with integer coords and norm 2?
    #    x^2+y^2+z^2... = 4.
    #    4+0... (22)
    #    1+1+1+1... (4 ones). C(11,4) * 2^4 = 330 * 16 = 5280 vectors.
    #    Norm sqrt(4) = 2.
    #    So we have 22 + 5280 = 5302 candidates of norm 2.
    #    If we use these, max norm is 2.
    #    Min dist must be >= 2.
    #    Let v, w be two such vectors.
    #    ||v-w|| >= 2.
    #    Consider v = 2e1, w = 2e2. Dist = 2sqrt(2) >= 2. OK.
    #    Consider v = 2e1, w = e1+e2+e3+e4.
    #    v-w = e1 - e2 - e3 - e4.
    #    Norm^2 = 1+1+1+1 = 4. Dist=2. OK.
    #    Consider v = e1+e2+e3+e4, w = e1+e2+e5+e6.
    #    v-w = e3+e4 - e5 - e6.
    #    Norm^2 = 4. Dist=2. OK.
    #    Consider v = e1+e2+e3+e4, w = e1+e2+e3+e5.
    #    v-w = e4 - e5.
    #    Norm^2 = 2. Dist=sqrt(2) < 2. INVALID.
    #    So we cannot take all 5280. We need a subset where min dist >= 2.
    #    This is equivalent to finding a code with min distance 2 in the set of weight-4 vectors (and weight-1 scaled vectors).
    #    Actually, if we restrict to weight-4 vectors (1s), min dist is determined by overlap.
    #    If overlap is 3, dist^2 = 2.
    #    If overlap is 2, dist^2 = 4.
    #    So we need no two vectors to share 3 or more coordinates.
    #    This is a packing problem.
    
    # Comparison:
    # Norm sqrt(2) set: k=110.
    # Norm 2 set: Potentially larger?
    # Total candidates 5302.
    # If we can pack ~10% of them, we get 500+.
    # This would beat 252.
    
    # Let's try to construct a large set of norm 2 vectors.
    # Candidates:
    # A: ±2e_i (22 vectors).
    # B: Vectors with four ±1s (5280 vectors).
    
    # If we include A, we must ensure dist to B is >= 2.
    # v in A (2e1), w in B (has e1 term).
    # If w has +1 at pos 1: w = e1 + ...
    # v-w = e1 - ...
    # If w = e1+e2+e3+e4, v-w = e1 - (e1+e2+e3+e4) = -e2-e3-e4. Norm sqrt(3) < 2. INVALID.
    # So if we use 2e1, we cannot use any B vector with +1 at pos 1.
    # We CAN use B vectors with -1 at pos 1?
    # w = -e1 + e2 + e3 + e4.
    # v-w = 2e1 - (-e1+e2+e3+e4) = 3e1 - e2 - e3 - e4.
    # Norm^2 = 9+1+1+1 = 12. Dist > 2. OK.
    # We CAN use B vectors with 0 at pos 1.
    # w = e2+e3+e4+e5.
    # v-w = 2e1 - w. Norm^2 = 4+4=8. OK.
    
    # So, to maximize count, maybe we should drop A and just use B?
    # Or mix carefully.
    # Given the complexity, let's try to solve the Max Independent Set on the B vectors (5280 nodes) with constraint dist >= 2.
    # Conflict if dist < 2.
    # Dist < 2 for integer vectors of norm 2 means Dist^2 < 4.
    # Dist^2 can be 1 or 2.
    # Dist^2 = 1: v=w (same vector) or v=-w?
    # v=w is same node.
    # v=-w: Norm of v-w is 2*||v|| = 4. Dist=2. Not < 2.
    # So Dist^2=1 only if v=w.
    # Dist^2 = 2:
    # v, w have norm 2.
    # ||v-w||^2 = 8 - 2<v,w>.
    # 8 - 2<v,w> = 2 => 2<v,w> = 6 => <v,w> = 3.
    # So conflict if <v,w> = 3.
    # For weight-4 vectors (±1s), <v,w> = sum of products.
    # If they share 4 positions with same sign: <v,w>=4. (Same vector)
    # If they share 3 positions with same sign, 1 diff: <v,w> = 3 - 1 = 2?
    # Let's check.
    # v = 1 1 1 1 0...
    # w = 1 1 1 -1 0...
    # <v,w> = 1+1+1-1 = 2.
    # Dist^2 = 8 - 4 = 4. Dist=2. OK.
    # w = 1 1 -1 1 0...
    # <v,w> = 1+1-1+1 = 2.
    # What gives <v,w>=3?
    # Impossible for distinct weight-4 vectors?
    # Max <v,w> for distinct is 2.
    # So NO conflicts among B vectors?
    # Let's re-verify.
    # v, w weight 4.
    # <v,w> = 4 - 2 * (number of sign mismatches) - 2 * (number of position mismatches)?
    # No.
    # Let S_v be support of v, S_w support of w.
    # <v,w> = sum_{i in S_v ∩ S_w} v_i w_i.
    # Max sum is when S_v = S_w and signs match. Then <v,w>=4 (v=w).
    # If S_v != S_w, say |S_v ∩ S_w| = 3.
    # One element in S_v not in S_w, one in S_w not in S_v.
    # v = 1 1 1 1 0...
    # w = 1 1 1 0 1...
    # <v,w> = 1+1+1 = 3.
    # Ah!
    # If they share 3 positions with same signs, and the 4th is different (one has 1, other has 0), then <v,w>=3.
    # Dist^2 = 8 - 2(3) = 2.
    # Dist = sqrt(2) < 2.
    # So CONFLICT if they share exactly 3 coordinates with the same sign.
    
    # So we need a set of weight-4 vectors (with signs) such that no two share 3 coordinates with the same sign.
    # This is a combinatorial design problem.
    # Number of candidates: 5280.
    # We can use a greedy algorithm to pick as many as possible.
    
    # Let's implement a greedy solver for this specific pool.
    
    candidates = []
    # Type B: 4 ones, 7 zeros. Signs ±1.
    # Iterate over combinations of 4 indices from 11.
    from itertools import combinations
    for comb in combinations(range(11), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(11, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Conflict: <v, w> == 3.
    # Compute dot products.
    # n_cand is 5280. 5280x5280 matrix is 27M entries. Doable in numpy.
    dots = candidates @ candidates.T
    
    # We want to select max subset with no pair having dot 3.
    # Greedy: Sort by degree (number of conflicts), pick lowest degree, remove neighbors.
    
    # Build conflict graph (boolean matrix)
    conflict = (dots == 3)
    np.fill_diagonal(conflict, False)
    
    degrees = np.sum(conflict, axis=1)
    
    selected_indices = []
    remaining = set(range(n_cand))
    
    # Greedy with degree ordering
    # To speed up, we can use a heap or just sort once and iterate?
    # Degree changes as we remove nodes.
    # Simple greedy:
    # 1. Sort candidates by initial degree.
    # 2. Iterate. If node is still remaining and not conflicting with already selected, select it.
    #    (This is not optimal MIS, but good heuristic).
    
    order = np.argsort(degrees)
    
    for i in order:
        if i not in remaining:
            continue
        # Check if i conflicts with any already selected
        # To make this fast, keep a set of selected.
        # Conflict with selected j if conflict[i, j] is True.
        # We can maintain a "blocked" array.
        pass

    # Better Greedy:
    # Maintain a list of available nodes.
    # At each step, pick node with min degree in the induced subgraph of available nodes.
    # This is O(N^3) or worse. N=5280 is too slow for Python loops.
    
    # Approximation:
    # 1. Compute initial degrees.
    # 2. Sort by degree.
    # 3. Iterate through sorted list.
    # 4. If node is not "blocked" by previous selections, select it and block its neighbors.
    
    blocked = np.zeros(n_cand, dtype=bool)
    selected = []
    
    for i in order:
        if blocked[i]:
            continue
        selected.append(i)
        # Block neighbors
        neighbors = np.where(conflict[i])[0]
        blocked[neighbors] = True
        blocked[i] = True # Mark as processed/selected
        
    selected_vectors = candidates[selected]
    
    # Add Type A vectors (±2e_i) if they fit?
    # Type A: 22 vectors.
    # Check if they conflict with selected B vectors.
    # Conflict if dist < 2.
    # v_A (2e_i), v_B (weight 4).
    # Dist^2 = ||v_A - v_B||^2.
    # If v_B has 0 at i: Dist^2 = 4 + 4 = 8. OK.
    # If v_B has 1 at i: Dist^2 = (2-1)^2 + 3 = 1+3=4. Dist=2. OK.
    # If v_B has -1 at i: Dist^2 = (2-(-1))^2 + 3 = 9+3=12. OK.
    # So Type A vectors NEVER conflict with Type B vectors (dist >= 2).
    # Wait, did I miss something?
    # v_A = 2e1. v_B = e1+e2+e3+e4.
    # v_A - v_B = e1 - e2 - e3 - e4.
    # Norm^2 = 1+1+1+1 = 4. Dist=2.
    # Condition: Dist >= Max Norm.
    # Max Norm is 2 (from Type A and Type B).
    # 2 >= 2. OK.
    # So we can add ALL 22 Type A vectors!
    
    a_vectors = []
    for i in range(11):
        v1 = np.zeros(11, dtype=np.int64); v1[i] = 2
        v2 = np.zeros(11, dtype=np.int64); v2[i] = -2
        a_vectors.append(v1)
        a_vectors.append(v2)
        
    final_vectors = np.vstack([selected_vectors, np.array(a_vectors)])
    
    # Final Validation
    norms_sq = np.sum(final_vectors**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    n = len(final_vectors)
    min_dist_sq = np.inf
    for i in range(n):
        diffs = final_vectors[i+1:] - final_vectors[i]
        dists_sq = np.sum(diffs**2, axis=1)
        if len(dists_sq) > 0:
            min_d = np.min(dists_sq)
            if min_d < min_dist_sq:
                min_dist_sq = min_d
                
    if min_dist_sq >= max_norm_sq:
        return final_vectors
    else:
        # Fallback to E8⊕A3 if greedy failed validation (shouldn't happen)
        e8_vectors = []
        for i in range(8):
            for j in range(i + 1, 8):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(8, dtype=np.int64)
                        v[i] = 2 * si
                        v[j] = 2 * sj
                        e8_vectors.append(v)
        for bits in range(256):
            v = np.zeros(8, dtype=np.int64)
            minus_count = 0
            for i in range(8):
                if (bits >> i) & 1:
                    v[i] = -1
                    minus_count += 1
                else:
                    v[i] = 1
            if minus_count % 2 == 0:
                e8_vectors.append(v)
        a3_vectors = []
        for i in range(3):
            for j in range(i + 1, 3):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(3, dtype=np.int64)
                        v[i] = 2 * si
                        v[j] = 2 * sj
                        a3_vectors.append(v)
        selected = []
        for v in e8_vectors:
            full_v = np.zeros(D, dtype=np.int64)
            full_v[:8] = v
            selected.append(full_v)
        for v in a3_vectors:
            full_v = np.zeros(D, dtype=np.int64)
            full_v[8:] = v
            selected.append(full_v)
        return np.array(selected, dtype=np.int64)
