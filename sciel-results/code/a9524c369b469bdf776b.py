import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    D = 11
    
    # Strategy:
    # The kissing number in dimension 11 is 252.
    # This is achieved by the lattice E8 x A3 (or A11, but E8 x A3 is easier to construct explicitly).
    # The norm of the minimal vectors in E8 x A3 is sqrt(2) (in E8 part) + 0 (in A3 part) -> wait.
    # Let's check the norms.
    # E8 minimal vectors have norm squared 2.
    # A3 minimal vectors have norm squared 2.
    # In E8 x A3, the minimal vectors are of type (e8, 0) and (0, a3).
    # Norm squared of (e8, 0) is 2.
    # Norm squared of (0, a3) is 2.
    # So all minimal vectors have norm sqrt(2).
    # The distance between any two distinct minimal vectors must be at least sqrt(2) to satisfy the condition?
    # Condition: min pairwise distance >= max norm.
    # Max norm is sqrt(2).
    # So we need dist >= sqrt(2), i.e., dist^2 >= 2.
    
    # Let's verify if E8 x A3 minimal vectors satisfy dist^2 >= 2.
    # E8 has 240 minimal vectors.
    # A3 has 12 minimal vectors.
    # Total 240 + 12 = 252 vectors.
    
    # We need to construct these 252 vectors in R^11.
    # E8 lives in R^8.
    # A3 lives in R^3.
    # We can embed E8 in the first 8 coordinates and A3 in the last 3 coordinates.
    
    # 1. Generate E8 minimal vectors.
    # E8 minimal vectors are:
    # - All vectors with coordinates in {0, +-1} with even sum of coordinates and exactly 4 non-zero entries.
    #   Number: C(8,4) * 2^4 = 70 * 16 = 1120? No.
    #   Wait, the standard E8 lattice definition:
    #   Vectors with all integer coordinates and even sum, or all half-integer coordinates with even sum.
    #   Minimal vectors (norm^2=2):
    #   Type 1: (x1, ..., x8) with xi in {0, +-1}, exactly four non-zero, sum even.
    #           Since non-zeros are +-1, sum is even iff number of +1s is even.
    #           Choose 4 positions: C(8,4) = 70.
    #           Assign signs: 2^4 = 16. But sum must be even.
    #           Number of +1s can be 0, 2, 4.
    #           C(4,0) + C(4,2) + C(4,4) = 1 + 6 + 1 = 8.
    #           So 70 * 8 = 560.
    #   Type 2: (x1, ..., x8) with xi in {1/2, -1/2}, sum even.
    #           Sum is even iff number of +1/2s is even.
    #           Assign signs: 2^8 = 256.
    #           Number of +1/2s can be 0, 2, 4, 6, 8.
    #           C(8,0)+C(8,2)+C(8,4)+C(8,6)+C(8,8) = 1 + 28 + 70 + 28 + 1 = 128.
    #           So 128 vectors.
    #   Total E8 minimal vectors = 560 + 128 = 688?
    #   Wait, the kissing number of E8 is 240.
    #   Where is the error?
    #   Ah, the condition for E8 is:
    #   1. All integer coords, even sum.
    #   2. All half-integer coords, even sum.
    #   Minimal vectors have norm^2 = 2.
    #   Type 1: 4 non-zero integers +-1.
    #           Sum of 4 integers +-1 is even.
    #           Number of ways to choose 4 positions: C(8,4) = 70.
    #           Number of sign assignments with even sum:
    #           Let k be number of +1s. k=0,2,4.
    #           C(4,0) + C(4,2) + C(4,4) = 1 + 6 + 1 = 8.
    #           70 * 8 = 560.
    #   Type 2: 8 non-zero half-integers +-1/2.
    #           Sum of 8 half-integers is even integer.
    #           Let k be number of +1/2s. Sum = (k - (8-k))/2 = (2k-8)/2 = k-4.
    #           k-4 must be even => k is even.
    #           k=0,2,4,6,8.
    #           C(8,0)+C(8,2)+C(8,4)+C(8,6)+C(8,8) = 128.
    #           128 vectors.
    #   Total 560 + 128 = 688.
    #   Why is the kissing number 240?
    #   Because the E8 lattice is not just these vectors.
    #   Wait, the minimal vectors of E8 are indeed 240.
    #   Let's re-read the definition.
    #   E8 = { x in Z^8 : sum xi even } union { x in (Z+1/2)^8 : sum xi even }
    #   Norm squared = sum xi^2.
    #   For Type 1: xi in {0, +-1}. Norm^2 = 2 => exactly two non-zeros?
    #   No, 1^2 + 1^2 = 2. So exactly two non-zeros!
    #   My previous calculation assumed 4 non-zeros (norm^2=4).
    #   Correct Type 1: Exactly two non-zero entries, each +-1.
    #   Choose 2 positions: C(8,2) = 28.
    #   Signs: (+,+) and (-,-) give even sum. (+,-) and (-,+) give odd sum.
    #   So 2 sign patterns per pair.
    #   28 * 2 = 56.
    #   For Type 2: xi in {1/2, -1/2}. Norm^2 = 2 => sum xi^2 = 2.
    #   Since each xi^2 = 1/4, we need 8 non-zeros.
    #   Sum xi = k/2 - (8-k)/2 = k-4.
    #   Sum must be even integer. k-4 even => k even.
    #   Number of such vectors: 128.
    #   Total 56 + 128 = 184? Still not 240.
    #   What is missing?
    #   Ah, the definition of E8 often includes vectors like (1, -1, 0, ...) which are in Type 1.
    #   Let's check the count again.
    #   Kissing number of E8 is 240.
    #   Maybe I am missing a type?
    #   No, E8 is 8-dimensional.
    #   Let's list them:
    #   1. (x1, ..., x8) in Z^8, sum even. Min norm^2=2.
    #      Vectors with two +-1 and six 0s.
    #      C(8,2) * 2 (signs: ++, --) = 28 * 2 = 56.
    #   2. (x1, ..., x8) in (Z+1/2)^8, sum even. Min norm^2=2.
    #      Vectors with eight +-1/2.
    #      Sum even => number of +1/2 is even.
    #      C(8,0)+C(8,2)+C(8,4)+C(8,6)+C(8,8) = 128.
    #   Total 56 + 128 = 184.
    #   Where are the other 56?
    #   Ah! The condition "sum even" for half-integers:
    #   Sum = (k - (8-k))/2 = k-4.
    #   If k is even, sum is integer.
    #   Is it required to be EVEN integer?
    #   The lattice definition is: sum xi is even.
    #   If sum xi = 0, 2, -2, etc., it is even.
    #   If sum xi = 1, 3, etc., it is odd.
    #   My calculation k-4 even covers sum = -4, -2, 0, 2, 4.
    #   So 128 is correct for half-integers.
    #   
    #   Wait, I might be confusing E8 with D8?
    #   D8 has 112 minimal vectors (Type 1 only, 2 non-zeros).
    #   E8 adds the half-integers.
    #   112 + 128 = 240.
    #   Ah, Type 1 count:
    #   C(8,2) * 2 (signs ++, --) = 56?
    #   No, for D8, signs can be (+,+) and (+,-) etc?
    #   D8: even sum of integers.
    #   Two non-zeros:
    #   (+,+) sum=2 (even)
    #   (+,-) sum=0 (even)
    #   (-,+) sum=0 (even)
    #   (-,-) sum=-2 (even)
    #   All 4 sign combinations are valid for D8!
    #   So C(8,2) * 4 = 28 * 4 = 112.
    #   Total E8 = 112 + 128 = 240. Correct.
    
    # So, E8 minimal vectors:
    # 1. 112 vectors with two +-1 and six 0s.
    # 2. 128 vectors with eight +-1/2.
    
    # A3 minimal vectors:
    # A3 is the root system of SU(4).
    # Roots are vectors in R^3 (or R^4 with sum 0) of form (e_i - e_j).
    # In R^3 coordinates (orthonormal basis for the hyperplane sum=0):
    # It is easier to work in R^4 with sum=0 and project, or just use the standard R^3 embedding.
    # Standard A3 roots in R^3:
    # (1, -1, 0), (1, 0, -1), (0, 1, -1) and their negatives.
    # Also permutations?
    # The 12 roots of A3 are:
    # (+-1, +-1, 0) with signs such that?
    # Let's list them:
    # (1, 1, 0), (1, 0, 1), (0, 1, 1)
    # (-1, -1, 0), (-1, 0, -1), (0, -1, -1)
    # (1, -1, 0), (-1, 1, 0), (1, 0, -1), (-1, 0, 1), (0, 1, -1), (0, -1, 1)
    # Total 12.
    # Norm squared of each is 2.
    
    # Now we combine E8 (8D) and A3 (3D) to get 11D vectors.
    # Vector v = (e, a) where e in E8_min, a in R^3 zero, OR e in R^8 zero, a in A3_min.
    
    vectors = []
    
    # Part 1: E8 vectors embedded in first 8 dims, last 3 are 0.
    # Type 1: Two +-1, six 0s.
    for i in range(8):
        for j in range(i+1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0]*11
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                    
    # Type 2: Eight +-1/2.
    # We need to multiply by 2 to keep integers?
    # The task requires INTEGER vectors.
    # E8 half-integer vectors are not integers.
    # However, we can scale the whole lattice?
    # If we scale E8 by 2, the minimal vectors become integers with norm^2 = 2 * 2^2 = 8?
    # Let's check.
    # Original E8 min norm^2 = 2.
    # If we use 2*E8, min norm^2 = 8.
    # If we use A3, min norm^2 = 2.
    # If we combine 2*E8 and A3, the norms are mixed (sqrt(8) and sqrt(2)).
    # Max norm would be sqrt(8).
    # Distances must be >= sqrt(8).
    # This might break the packing.
    
    # Alternative: Use only the integer part of E8?
    # The 112 integer vectors of E8 (Type 1) form the D8 lattice.
    # D8 kissing number is 112.
    # If we use D8 x A3:
    # D8 min norm^2 = 2.
    # A3 min norm^2 = 2.
    # Total vectors: 112 + 12 = 124.
    # This is less than 224 (seed).
    
    # We need the half-integers to get the full 240.
    # To make them integers, we can scale the E8 part by 2.
    # Let's try a different lattice.
    
    # What about the A11 lattice?
    # A11 is the root system of SU(12).
    # Roots are e_i - e_j in R^12 (sum 0).
    # Number of roots: 12*11 = 132.
    # This is also less than 224.
    
    # The Leech lattice is 24D.
    
    # Let's reconsider the E8 x A3 with scaling.
    # If we scale E8 by 2, vectors are integers.
    # v_e = 2 * e (e in E8_min). Norm^2 = 4 * 2 = 8.
    # v_a = a (a in A3_min). Norm^2 = 2.
    # Set S = { (2e, 0) } U { (0, a) }.
    # Max norm^2 = 8.
    # Condition: dist^2 >= 8.
    # Check distances:
    # 1. Between two E8 vectors: dist^2 = ||2e1 - 2e2||^2 = 4 ||e1-e2||^2.
    #    Min dist in E8 is sqrt(2). So min dist^2 = 4*2 = 8. OK.
    # 2. Between two A3 vectors: dist^2 = ||a1-a2||^2.
    #    Min dist in A3?
    #    Roots of A3: (1,1,0) etc.
    #    Distance between (1,1,0) and (1,0,1) is sqrt(0+1+1) = sqrt(2).
    #    Distance between (1,1,0) and (-1,-1,0) is sqrt(4+4+0) = sqrt(8).
    #    Min dist^2 in A3 is 2.
    #    We need dist^2 >= 8.
    #    So A3 vectors will conflict with each other!
    #    We can only pick a subset of A3 vectors that are mutually distance >= sqrt(8).
    #    In A3, max distance between roots is sqrt(8) (opposites).
    #    So we can pick at most one pair of opposites? Or just one vector?
    #    If we pick (1,1,0) and (-1,-1,0), dist^2=8. OK.
    #    Can we pick more?
    #    (1,1,0) and (1,0,1) dist^2=2 < 8. Conflict.
    #    So we can pick very few A3 vectors.
    #    This strategy yields ~112 (scaled E8 int part) + few A3. Not good.
    
    # What if we scale A3 by 2?
    # v_a = 2a. Norm^2 = 8.
    # v_e = e (integer part of E8, D8). Norm^2 = 2.
    # Max norm^2 = 8.
    # Dist between E8: min 2 < 8. Conflict.
    
    # We need a uniform scaling.
    # Let's use the D8 lattice (112 vectors) and A3 lattice (12 vectors).
    # Both have min norm^2 = 2.
    # If we just take D8 x A3, we have 112+12=124 vectors.
    # Distances:
    # E8-E8: min 2.
    # A3-A3: min 2.
    # E8-A3: orthogonality => dist^2 = ||e||^2 + ||a||^2 = 2+2=4.
    # Max norm^2 = 2.
    # Condition: dist^2 >= 2.
    # All distances are >= 2.
    # So 124 is a valid configuration.
    # But seed is 224.
    
    # How to get more?
    # The seed program likely uses the full E8 lattice but scaled?
    # If the seed score is 224, it corresponds to 224 vectors.
    # 224 is the kissing number of D11? No.
    # 224 is the number of vectors of norm sqrt(2) in the D11 lattice?
    # D11 minimal vectors: 2 * C(11,2) = 110.
    # Not 224.
    
    # Wait, the seed program in the prompt says:
    # "If no improvement, return the seed 224 vectors (norm sqrt(2))"
    # "Construct the 224 vectors... v[i]=si, v[j]=sj"
    # This is the D11 lattice vectors?
    # C(11,2) * 4 = 55 * 4 = 220.
    # Plus 4 more?
    # The code:
    # for i in range(D): for j in range(i+1, D): for si in [1,-1]: for sj in [1,-1]:
    # This generates 220 vectors.
    # Why does it say 224?
    # Maybe it adds (1,1,1,1,1,1,1,1,1,1,1)? No, norm sqrt(11).
    # Maybe it adds (2,0...)? Norm 2.
    # If max norm is sqrt(2), we can't add norm 2 vectors.
    
    # Let's assume the seed is 220 or 224 and valid.
    # To beat it, we need > 224.
    
    # The inspiration says E8 x A3 gives 252.
    # How to make E8 x A3 integer?
    # The half-integer vectors in E8 are the problem.
    # However, note that 2*E8 is an integer lattice.
    # And 2*A3 is an integer lattice.
    # Consider the lattice L = 2*E8 x 2*A3.
    # Minimal vectors:
    # 2*e (norm^2=8) and 2*a (norm^2=8).
    # Total 240 + 12 = 252 vectors.
    # All have norm^2 = 8.
    # Distances:
    # Between 2*e1 and 2*e2: 4||e1-e2||^2 >= 4*2 = 8. OK.
    # Between 2*a1 and 2*a2: 4||a1-a2||^2 >= 4*2 = 8. OK.
    # Between 2*e and 2*a: 4(||e||^2 + ||a||^2) = 4(2+2) = 16 >= 8. OK.
    # So this is a valid configuration of 252 integer vectors!
    
    # We just need to generate them.
    
    vectors = []
    
    # 1. Generate 2*E8 minimal vectors.
    # Type 1: 2 * (two +-1, six 0s).
    # Values: +-2, 0.
    for i in range(8):
        for j in range(i+1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0]*11
                    v[i] = 2*si
                    v[j] = 2*sj
                    vectors.append(v)
                    
    # Type 2: 2 * (eight +-1/2).
    # Values: +-1.
    # All 8 coordinates are +-1.
    # Sum of original E8 vector was even.
    # Original sum = k-4 (k even).
    # Scaled sum = 2*(k-4) = 2k-8 (even).
    # The condition for E8 half-integers is just that they are in the lattice.
    # We generate all 2^8 = 256 sign combinations?
    # No, only those with even sum of the ORIGINAL half-integers.
    # Original half-integers: x_i = s_i / 2.
    # Sum x_i = (sum s_i) / 2.
    # Condition: sum x_i is even integer => sum s_i is divisible by 4.
    # Let's check:
    # If sum s_i = 0, sum x_i = 0 (even).
    # If sum s_i = 4, sum x_i = 2 (even).
    # If sum s_i = 8, sum x_i = 4 (even).
    # If sum s_i = -4, sum x_i = -2 (even).
    # If sum s_i = -8, sum x_i = -4 (even).
    # So we need sum s_i % 4 == 0.
    
    for signs in itertools.product([1, -1], repeat=8):
        s_sum = sum(signs)
        if s_sum % 4 == 0:
            v = [0]*11
            for k in range(8):
                v[k] = signs[k] # Since 2 * (s_k / 2) = s_k
            vectors.append(v)
            
    # 2. Generate 2*A3 minimal vectors.
    # A3 roots: (1,1,0), (1,0,1), (0,1,1) and permutations/negatives.
    # Scaled by 2: (2,2,0), (2,0,2), (0,2,2).
    # And negatives.
    # And the other 6 roots: (1,-1,0) etc.
    # Scaled: (2,-2,0) etc.
    
    a3_roots = [
        (1, 1, 0), (1, 0, 1), (0, 1, 1),
        (-1, -1, 0), (-1, 0, -1), (0, -1, -1),
        (1, -1, 0), (-1, 1, 0),
        (1, 0, -1), (-1, 0, 1),
        (0, 1, -1), (0, -1, 1)
    ]
    
    for root in a3_roots:
        v = [0]*11
        v[8] = 2*root[0]
        v[9] = 2*root[1]
        v[10] = 2*root[2]
        vectors.append(v)
        
    # Convert to numpy
    vec_arr = np.array(vectors, dtype=np.int64)
    
    # Verify validity
    if len(vec_arr) > 0:
        # Check max norm
        norms_sq = np.sum(vec_arr**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Check min distance
        # This is O(N^2 * D). N=252. 252^2 * 11 ~ 700k ops. Fast.
        min_dist_sq = float('inf')
        for i in range(len(vec_arr)):
            for j in range(i+1, len(vec_arr)):
                d = vec_arr[i] - vec_arr[j]
                d_sq = np.sum(d**2)
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
                    
        if min_dist_sq >= max_norm_sq:
            return vec_arr
        else:
            # Fallback to seed if construction failed (should not happen)
            pass
            
    # Fallback: Seed 220 vectors
    base_vectors = []
    for i in range(11):
        for j in range(i + 1, 11):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * 11
                    v[i] = si
                    v[j] = sj
                    base_vectors.append(v)
    return np.array(base_vectors, dtype=np.int64)
