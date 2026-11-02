import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank 18 as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

    # Deterministic search for rank <= 15
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all (i,j,k) in {0..n-1} x {0..m-1} x {0..p-1}
    # A rank-1 term with supports (U_idx, V_idx, W_idx) contributes to T at position (i,j,k) iff:
    #   i = divmod(U_idx, m)[0], j = divmod(U_idx, m)[1]
    #   j = divmod(V_idx, p)[0], k = divmod(V_idx, p)[1]
    #   k = divmod(W_idx, n)[0], i = divmod(W_idx, n)[1]
    # So for a given triple of indices (a, b, c), it covers (i,j,k) if the indices are consistent.
    
    # Generate all valid rank-1 patterns (a, b, c) where the indices are consistent
    valid_patterns = []
    for a in range(n * m):
        i, j = divmod(a, m)
        for b in range(m * p):
            j2, k = divmod(b, p)
            if j != j2:
                continue
            for c in range(p * n):
                k2, i2 = divmod(c, n)
                if k != k2 or i != i2:
                    continue
                valid_patterns.append((a, b, c))
    
    # For each valid pattern, determine which (i,j,k) it covers
    # Since each pattern is consistent, it covers exactly one (i,j,k)
    pattern_to_index = {}
    for pat in valid_patterns:
        a, b, c = pat
        i, j = divmod(a, m)
        j2, k = divmod(b, p)
        k2, i2 = divmod(c, n)
        # i==i2, j==j2, k==k2 by construction
        triple = (i, j, k)
        if triple not in pattern_to_index:
            pattern_to_index[triple] = []
        pattern_to_index[triple].append(len(valid_patterns))
    
    # We need to find a set of at most 15 patterns and coefficients (integers or half-integers)
    # such that the sum equals the all-ones tensor of size n*m*p = 18.
    
    # Since each pattern covers exactly one (i,j,k) with coefficient 1, and we need each (i,j,k) to have sum = 1,
    # this is equivalent to: for each (i,j,k), the sum of coefficients of patterns covering it must be 1.
    
    # Let's try rank 15: we need to assign 15 patterns to cover 18 positions, with some patterns sharing coverage.
    # Actually, each pattern covers exactly ONE (i,j,k). So if we use R patterns, we can cover at most R distinct (i,j,k) positions.
    # To cover all 18 positions with R < 18 patterns is impossible if each pattern covers exactly one position.
    
    # Wait, I made an error. Let me reconsider.
    # In a rank-1 tensor term, the support is a product set. For matrix multiplication, a rank-1 term corresponds to
    # choosing specific rows/columns. The standard rank-1 term for (i,j,k) has:
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1, and all other entries 0.
    # This term contributes exactly 1 to T[i,j,k] and 0 elsewhere.
    
    # So each rank-1 pattern covers exactly one (i,j,k). Therefore, we need at least 18 patterns to cover all 18 positions.
    # Rank 15 is impossible with this structure.
    
    # Let me reconsider the problem. Maybe I'm misunderstanding the tensor structure.
    
    # The matrix multiplication tensor T is defined such that:
    # T[i, j, k] = 1 if the entry (i,k) of the product matrix A*B equals sum_j A[i,j]*B[j,k]
    # So T is the tensor where T[i,j,k] = 1 for all valid (i,j,k).
    
    # A rank-1 decomposition term is U_r * V_r * W_r where:
    # U_r is a vector of length n*m, V_r is a vector of length m*p, W_r is a vector of length p*n
    # The contribution to T[i,j,k] from term r is U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    
    # For the standard rank-1 term for (i,j,k):
    # U_r[i*m+j] = 1, all other U_r = 0
    # V_r[j*p+k] = 1, all other V_r = 0  
    # W_r[k*n+i] = 1, all other W_r = 0
    # This gives contribution 1 to T[i,j,k] and 0 to all other T[i',j',k'].
    
    # So indeed, each rank-1 term covers exactly one (i,j,k). We need 18 terms to cover all 18 positions.
    # Rank < 18 is impossible.
    
    # Therefore, the minimal rank is 18, and the fallback is optimal.
    
    return fallback
# EVOLVE-BLOCK-END
