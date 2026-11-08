import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find a lower-rank decomposition using structured patterns
    # The tensor is T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # We want to write T as sum of rank-1 tensors U_c ⊗ V_c ⊗ W_c
    
    # Strategy: use a set of candidate rank-1 terms and solve ILP
    # For efficiency, we'll try known constructions for rectangular matrix mult
    
    # Known: for 4x5 x 5x5, we can potentially do better than 100
    # Let's try a construction based on block decomposition
    
    # Attempt 1: Try rank 90-100 with structured patterns
    # Use the fact that some entries can be shared
    
    # Build the tensor T as a 3D array: T[a, b, c] where
    # a = i*m + j (0..n*m-1), b = j*p + k (0..m*p-1), c = k*n + i (0..p*n-1)
    # T[a, b, c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # We seek rank-r decomposition: T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # and entries are integers or half-integers
    
    # For a direct approach, try to find a decomposition using
    # the structure of the tensor
    
    # Let's try a greedy/structured approach:
    # Each rank-1 term u⊗v⊗w contributes to positions (a,b,c) where
    # T[a,b,c] should equal sum of u[a]*v[b]*w[c] over all terms
    
    # Try construction: group by (i,k) pairs and share j-dependent terms
    # Or group by (j,i) and share k-dependent terms
    
    # Construction attempt: Use the fact that for fixed i, the slice
    # T[i*m+*, *, *] has structure related to matrix multiplication row i
    
    # Let's try a rank reduction by combining terms that differ only in one index
    
    # For 4x5x5, try rank = 95, 90, 85, etc.
    # Use a simple structured construction
    
    # Construction: For each (i, k) pair, we have m terms (one for each j)
    # But we can potentially combine some using linearity in j
    
    # Actually, let's try a different approach: use the known result that
    # M(4,5) <= M(5,5) * something, but let's just try to find a good rank
    
    # Try a construction where we use rank-1 terms of the form:
    # For each (i, j, k): term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # That's the schoolbook. To reduce rank, we need to combine terms.
    
    # Key insight: if two terms differ only in one factor, we might combine them
    # using linearity. For example, terms with same (i,k) but different j:
    # sum_j u_j ⊗ v_j ⊗ w = (sum_j u_j) ⊗ ... but that doesn't work directly
    
    # Let's try a more systematic approach: use the tensor's symmetry
    
    # For now, return schoolbook if we can't find better quickly
    # But let's try a few structured constructions
    
    # Construction 2: Try to use rank-1 terms where U, V, W have multiple nonzeros
    # This can reduce the total number of terms needed
    
    # Let's try: for each pair (i, k), create terms that cover all j at once
    # Term for (i, k): U has 1s at positions i*m+j for all j, 
    #                   V has 1s at positions j*p+k for all j,
    #                   W has 1 at position k*n+i
    # But this gives T[i*m+j, j*p+k, k*n+i] = sum over... no, that's not right
    
    # The issue is that u⊗v⊗w at (a,b,c) = u[a]*v[b]*w[c]
    # So if u has multiple 1s, v has multiple 1s, w has one 1, then
    # the product is 1 at all (a,b,c) where u[a]=1, v[b]=1, w[c]=1
    # That creates too many 1s
    
    # So we need to be more careful. Let's try a different construction.
    
    # For the 4x5x5 case, let's try to find a decomposition of rank < 100
    # by solving a small ILP or using a structured pattern
    
    # Given time constraints, let's try a few specific ranks with random/structured search
    
    # Actually, let's just return the schoolbook for now if we can't find better
    # The task says to find rank-minimal, but we may not be able to prove minimality
    
    # Let's try one more construction: use the fact that n=4 is small
    # We can enumerate possible U patterns
    
    # For each column c, U[:,c] is a vector in {0, 0.5, 1}^{n*m} = {0,0.5,1}^{20}
    # Similarly for V and W
    
    # This is a huge search space, so let's use a heuristic
    
    # Heuristic: start with schoolbook and try to merge terms
    
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be merged if
    # they share two indices and the third can be handled by linearity
    
    # For example, if i1=i2 and j1=j2 but k1≠k2, we can't merge directly
    # because W would need to be 1 at both k1*n+i and k2*n+i
    
    # But if we allow W to have two 1s, then the term contributes to both
    # (i*m+j, j*p+k1, k1*n+i) and (i*m+j, j*p+k2, k2*n+i)
    # but also to cross terms which are 0 in T, so that's a problem
    
    # So merging only works if the cross terms are also 0 in T, which they aren't
    # in general
    
    # Given the difficulty, let's return the schoolbook decomposition
    # which is valid and has rank 100
    
    return best_result
# EVOLVE-BLOCK-END
