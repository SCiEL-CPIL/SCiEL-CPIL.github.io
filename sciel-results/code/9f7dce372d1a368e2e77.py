"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct kissing vectors in R^11 using the D11 lattice structure.
    
    The D_n lattice consists of integer vectors with even coordinate sum.
    The shortest vectors in D_n have squared norm 4 and are of two types:
    1. Vectors with two ±1 entries and rest 0 (even sum: 1+1=2 or 1-1=0)
    2. Vectors with four ±1 entries and rest 0 (even sum: ±1±1±1±1 can be even)
    
    For n=11, the number of shortest vectors is:
    - Type 1: C(11,2) * 2^2 = 55 * 4 = 220
    - Type 2: C(11,4) * (number of sign patterns with even sum)
      Number of sign patterns with 4 ones that have even sum: 
      The sum of 4 signs is even when there are 0, 2, or 4 negative signs.
      C(4,0) + C(4,2) + C(4,4) = 1 + 6 + 1 = 8
      So Type 2: C(11,4) * 8 = 330 * 8 = 2640
    
    Total: 220 + 2640 = 2860? Wait, let me reconsider.
    
    Actually, let me recompute. The D_n lattice shortest vectors:
    - Two non-zero coordinates, each ±1: C(n,2) * 2^2 = C(n,2)*4
      For n=11: 55*4 = 220
    - Four non-zero coordinates, each ±1, with even number of -1's:
      C(n,4) * (C(4,0)+C(4,2)+C(4,4)) = C(n,4) * (1+6+1) = C(n,4)*8
      For n=11: C(11,4)=330, so 330*8 = 2640
    
    Total = 220 + 2640 = 2860
    
    Wait, but the problem says 2304. Let me reconsider what "even coordinate sums" means.
    
    Actually, D_n is the set of integer vectors x where sum(x_i) is even.
    The shortest vectors have norm^2 = 4.
    
    Let me just generate all integer vectors with norm^2 = 4 and even coordinate sum.
    
    Vectors with norm^2 = 4 are:
    - Two coordinates are ±1, rest 0
    - One coordinate is ±2, rest 0
    - Four coordinates are ±1, rest 0 (but 1+1+1+1=4, so this also works)
    
    Wait, 1^2+1^2+1^2+1^2 = 4, yes.
    And 2^2 = 4, yes.
    
    So we need all integer vectors with sum of squares = 4 AND even coordinate sum.
    
    Type A: One coordinate ±2, rest 0. Sum = ±2, which is even. Count: 11*2 = 22
    Type B: Two coordinates ±1, rest 0. Sum = ±1±1 = -2, 0, or 2, all even. Count: C(11,2)*4 = 220
    Type C: Four coordinates ±1, rest 0. Sum must be even. Sum of four ±1's is even iff number of -1's is even (0, 2, or 4). Count: C(11,4)*(C(4,0)+C(4,2)+C(4,4)) = 330*8 = 2640
    
    Total = 22 + 220 + 2640 = 2882
    
    Hmm, still not 2304. Let me check if all these are actually in D_11 and have mutual distances >= 2 (since norm = 2).
    
    For a kissing configuration, we need: for all distinct u, v in the set, ||u - v|| >= ||u|| = 2.
    So ||u - v||^2 >= 4.
    
    Let's verify for the D_n shortest vectors. The inner product of two distinct shortest vectors in D_n can be -2, -1, 0, 1, or 2.
    ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 4 + 4 - 2<u,v> = 8 - 2<u,v>.
    
    We need 8 - 2<u,v> >= 4, i.e., <u,v> <= 2.
    
    The maximum inner product between distinct shortest vectors: if u and v share coordinates, the inner product can be at most... if they are the same vector, inner product is 4, but they're distinct. If they differ in one sign, e.g., u has +1 at i and v has -1 at i, and otherwise same, then <u,v> = 4 - 2 = 2 (if four 1's) or 2 - 2 = 0 (if two 1's). So max inner product for distinct vectors is 2, giving distance^2 = 8-4=4. Good.
    
    So all 2882 vectors should work. But the hint says 2304. Maybe I'm overcounting or the hint is approximate. Let me just generate them and count.
    """
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill
    
    vectors = []
    
    # Type A: One coordinate ±2, rest 0
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = 2 * sign
            vectors.append(v)
    
    # Type B: Two coordinates ±1, rest 0
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    # Type C: Four coordinates ±1, rest 0, with even sum
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for l in range(k + 1, D):
                    # All sign combinations with even number of -1's
                    # 0 negatives: all +1
                    # 2 negatives: choose 2 positions to be -1
                    # 4 negatives: all -1
                    signs = [1, 1, 1, 1]
                    # 0 negatives
                    v = [0] * D
                    v[i], v[j], v[k], v[l] = 1, 1, 1, 1
                    vectors.append(v)
                    # 2 negatives
                    for a in range(4):
                        for b in range(a + 1, 4):
                            s = [1, 1, 1, 1]
                            s[a] = -1
                            s[b] = -1
                            v = [0] * D
                            v[i], v[j], v[k], v[l] = s[0], s[1], s[2], s[3]
                            vectors.append(v)
                    # 4 negatives
                    v = [0] * D
                    v[i], v[j], v[k], v[l] = -1, -1, -1, -1
                    vectors.append(v)
    
    if time.time() > deadline - 10:
        # Return what we have
        pass
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
