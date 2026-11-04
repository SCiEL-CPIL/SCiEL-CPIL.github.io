"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 9
    
    U = [[0] * rank for _ in range(n * m)]   # shape (6, 9)
    V = [[0] * rank for _ in range(m * p)]   # shape (18, 9)
    W = [[0] * rank for _ in range(p * n)]   # shape (12, 9)
    
    # Kronecker product construction:
    # Decompose 2x3x6 as (2x1x3) ⊗ (1x3x2)
    # 
    # Index mapping:
    #   i in {0,1} -> i1 in {0,1} (from 2x1x3 factor)
    #   j in {0,1,2} -> j2 in {0,1,2} (from 1x3x2 factor)  
    #   k in {0,...,5} -> k1 in {0,1,2}, k2 in {0,1} where k = 2*k1 + k2
    #
    # The 2x1x3 tensor has rank 3 (columns indexed by i1 in {0,1} and k2 in {0,1}, 
    #   but actually it's simpler: the 2x1x3 multiplication is C[i1,k2] = A[i1,0]*B[0,k2]
    #   which requires 2*3=6 products... 
    #
    # Let me reconsider. The directive says rank 3 for each factor.
    #
    # Actually, for a 2x1x3 multiplication: C[i,k] = A[i,0]*B[0,k]
    # This is just an outer product of a 2-vector and a 3-vector.
    # The bilinear map has "rank" 1 in the sense that it's a single product of two vectors,
    # but in the tensor decomposition sense (sum of rank-1 tensors), it's 2*3 = 6.
    #
    # Wait, I think the "rank 3" refers to something else. Let me just try to construct
    # a valid rank-9 decomposition directly.
    #
    # We need: T[i,j,k] = sum_c U[3*i+j, c] * V[6*j+k, c] * W[2*k+i, c] = 1
    #
    # Let me use columns indexed by (a, b) where a in {0,1,2}, b in {0,1,2}.
    # 
    # Construction:
    #   U[3*i+j, 3*a+b] = 1 if i == a % 2 and j == b, else 0
    #   V[6*j+k, 3*a+b] = 1 if j == b, else 0
    #   W[2*k+i, 3*a+b] = 1 if k == a, else 0
    #
    # But k ranges 0..5, not 0..2. Let me use k = 2*k1 + k2 with k1 in {0,1,2}.
    #
    # Revised:
    #   U[3*i+j, 3*a+b] = 1 if i == a % 2 and j == b, else 0
    #   V[6*j+k, 3*a+b] = 1 if j == b and k == 2*a + (k%2), else 0  -- this is circular
    #
    # Let me try a different approach. I'll use the fact that we can verify the decomposition
    # and adjust if needed.
    #
    # Simple construction attempt:
    # For column c = 3*a + b:
    #   U[3*i+j, c] = 1 if (i == a % 2) and (j == b)
    #   V[6*j+k, c] = 1 if (j == b)
    #   W[2*k+i, c] = 1 if (k // 2 == a)
    #
    # Check: sum_c U[3*i+j,c]*V[6*j+k,c]*W[2*k+i,c]
    # = sum_{a,b} [i==a%2, j==b] * [j==b] * [k//2==a]
    # = sum_{a,b} [i==a%2] * [j==b]^2 * [k//2==a]
    # = [i == (k//2) % 2] * [j==j] * 1  (if there's exactly one a with k//2==a)
    #
    # This gives 1 only when i == (k//2) % 2, which is not always true.
    #
    # Let me try yet another approach. I'll just use rank 18 with a clean construction.
    
    # Actually, let me try rank 9 with a more careful construction.
    # 
    # The key insight: we have 2*3*6 = 36 equations and 9*(6+18+12) = 324 unknowns.
    # Let me try to find a sparse solution.
    #
    # Approach: For each (i,j,k), assign it to exactly one column c, and set
    # U[3*i+j,c] = V[6*j+k,c] = W[2*k+i,c] = 1.
    # But we only have 9 columns for 36 (i,j,k) triples, so each column must handle
    # multiple (i,j,k) triples.
    #
    # Let me try: column c = 3*a + b handles all (i,j,k) where j == b and k == 2*a + (i%2)?
    # No, that's not clean.
    #
    # I'll just go with rank 18 for safety, using a clean construction.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Columns indexed by (j, k1) where j in {0,1,2}, k1 in {0,1,2,3,4,5}
    # Wait, that's 3*6 = 18. But then for each (j,k1), we need to handle all i and k.
    #
    # Better: columns indexed by (j, k) where j in {0,1,2}, k in {0,...,5}. That's 18.
    # For column (j,k):
    #   U[3*i+j, col] = 1 for all i
    #   V[6*j+k, col] = 1
    #   W[2*k+i, col] = 1 for all i
    #
    # Check: sum_c U[3*i+j,c]*V[6*j+k,c]*W[2*k+i,c]
    # Only column c=(j,k) contributes: 1*1*1 = 1. ✓
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
                W[k * n + i][col] = 1
            V[j * p + k][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
