"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 110  # Safety margin before 120s kill
    n, m, p = DIMENSIONS
    # Tensor T(i,j,k) = delta_{j,k} for multiplying (n x m) by (m x p)
    # Decomposition: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = T(i,j,k)
    # Bilinear constraint: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{j,k}
    
    # Build schoolbook as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank via structured search
    # The tensor has structure: T(i,j,k) = delta_{j,k}
    # For each output index i, we need to compute C[i,:] = A[i,:] * B
    # where A is n x m and B is m x p
    # So for each i, we need rank of the matrix multiplication tensor for 1 x m by m x p
    # which is m*p = 16 for each i, but we can share across i's
    
    # Key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T(i,j,k) = delta_{j,k}
    # Rank of this tensor: we can write it as sum over j of (e_i in U) * (row j of B in V) * (col j in W)
    # But we need to be more careful.
    
    # Let's try rank r from lower bounds up
    # Lower bound: max(n*p, n*m, m*p) = max(16, 4, 16) = 16
    # Also, the tensor has "slice rank" considerations
    
    # For 2x2 x 2x8:
    # T(i,j,k) = delta_{j,k}
    # For fixed i: slice is identity-like, rank = min(m,p) = 2
    # For fixed j: slice has entries T(i,j,k) = delta_{j,k}, nonzero only when k=j
    #   so for fixed j, the slice over (i,k) is: T(i,j,k) = 1 if k==j else 0
    #   This is a column vector in k (all zeros except k=j) times a row vector in i (all ones)
    #   Rank of this slice = 1
    # For fixed k: slice T(i,j,k) = delta_{j,k}, nonzero only when j=k
    #   So for fixed k, T(i,k,k) = 1 for all i, T(i,j,k)=0 for j!=k
    #   This is a row vector in i (all ones) times column vector in j (1 at j=k)
    #   Rank = 1
    
    # The tensor rank is at least max over all slices of slice rank = 2
    # But we need a better lower bound.
    
    # Let's try to construct a decomposition of rank < 32
    # Strategy: group the 8 columns of B. For each j (0,1) and each k (0..7),
    # we need A[i,j]*B[j,k] to contribute to C[i,k].
    
    # Try rank = 16 (the theoretical lower bound from max dimension product)
    # Can we do it? For each (j,k) pair, we need one rank-1 term:
    # U[i*m+j, c] = delta_{i,0} or similar... no, we need it for all i.
    
    # Actually for each (j,k), the contribution is sum_i A[i,j] * B[j,k] * 1
    # So U[i*m+j, c_{j,k}] should be 1 for all i, V[j*p+k, c_{j,k}] = 1, W[k*n+i, c_{j,k}] = 1
    # But wait, that gives rank = m*p = 16 terms, and each term handles all i's simultaneously!
    
    # Let's verify: with rank r = m*p = 16, index c by (j,k) pair
    # U[i*m+j, c(j,k)] = 1 for all i
    # V[j*p+k, c(j,k)] = 1
    # W[k*n+i, c(j,k)] = 1 for all i
    
    # Then sum_c U[i*m+j', c] * V[j'*p+k', c] * W[k'*n+i, c]
    # = sum_{(j,k)} U[i*m+j', c(j,k)] * V[j'*p+k', c(j,k)] * W[k'*n+i, c(j,k)]
    # U[i*m+j', c(j,k)] = 1 if j' == j (since U[i*m+j', c(j,k)] = 1 for all i, but indexed by j')
    # Wait, U is indexed by row = i*m+j'. U[i*m+j', c(j,k)] = 1 for ALL i and for the specific j' that matches...
    # No! U[i*m+j', c] = 1 means for row index i*m+j', column c(j,k), the value is 1.
    # So U[i*m+j', c(j,k)] = 1 for all i, for the fixed j' that is part of the row index.
    # But we set U[i*m+j, c(j,k)] = 1, which means for row i*m+j (specific j), col c(j,k), value is 1.
    # For row i*m+j' where j' != j, U[i*m+j', c(j,k)] = 0 (default).
    
    # So sum_c U[i*m+j', c] * V[j'*p+k', c] * W[k'*n+i, c]
    # Only terms where j'=j and k'=k survive (from V and W constraints)
    # = U[i*m+j, c(j,k)] * V[j*p+k, c(j,k)] * W[k*n+i, c(j,k)]
    # = 1 * 1 * 1 = 1 when j'=j and k'=k, else 0.
    # = delta_{j',k'}... wait that's delta_{j',j} * delta_{k',k} which is delta_{j',k'} only if j=k.
    
    # Hmm, that's not right. Let me re-examine.
    # We want: sum_c U[i*m+j', c] * V[j'*p+k', c] * W[k'*n+i, c] = delta_{j',k'}
    
    # With my construction: term (j,k) contributes:
    # U[i*m+j', c(j,k)] * V[j'*p+k', c(j,k)] * W[k'*n+i, c(j,k)]
    # U[i*m+j', c(j,k)] is 1 iff j' == j (for any i)
    # V[j'*p+k', c(j,k)] is 1 iff j' == j AND k' == k
    # W[k'*n+i, c(j,k)] is 1 iff k' == k (for any i)
    
    # So the product is 1 iff j'==j AND k'==k, else 0.
    # Sum over all (j,k): gives 1 iff j'==k', else 0. YES! That's delta_{j',k'}.
    
    # So rank 16 works! Let me construct it.
    
    rank_candidate = m * p  # = 16
    U_cand = [[0] * rank_candidate for _ in range(n * m)]
    V_cand = [[0] * rank_candidate for _ in range(m * p)]
    W_cand = [[0] * rank_candidate for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            for i in range(n):
                U_cand[i * m + j][c] = 1
            V_cand[j * p + k][c] = 1
            for i in range(n):
                W_cand[k * n + i][c] = 1
    
    # Verify
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    expected = 1 if j == k else 0
                    if total != expected:
                        return False
        return True
    
    if verify(U_cand, V_cand, W_cand, n, m, p, rank_candidate):
        return {"U": U_cand, "V": V_cand, "W": W_cand}
    
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
