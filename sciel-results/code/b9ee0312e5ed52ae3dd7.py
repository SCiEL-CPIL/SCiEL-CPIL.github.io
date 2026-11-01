"""Matrix multiplication 2,2,13 - F_13-structured decomposition."""
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 13
    
    # Rank: n*m + p - 1 = 4 + 13 - 1 = 16
    rank = n * m + p - 1  # 16
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 16)
    V = [[0] * rank for _ in range(m * p)]  # (26, 16)
    W = [[0] * rank for _ in range(p * n)]  # (26, 16)
    
    # Construction based on F_13 structure:
    # For each k in 0..12, we use the field element k mod 13
    # The decomposition uses 16 rank-1 terms.
    #
    # Terms 0-3: for each (i,j) pair, a "base" term
    # Terms 4-15: for k=0..11, field-structured terms
    #
    # The key idea: for each output (i,k), the value is
    # out[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # We construct terms such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = sum_j A[i,j]*B[j,k]
    
    # Let's use a concrete construction:
    # For c in 0..3: U[i*m+j, c] = 1 if (i,j) == (c//2, c%2) else 0
    #                V[j*p+k, c] = 1 for all j,k (this won't work directly)
    
    # Alternative approach - use the fact that we can express the tensor
    # using polynomial interpolation over F_13.
    
    # Actually, let me try a direct constructive approach:
    # The tensor T[i,j,k] = delta (we need T[i,j,k] such that 
    # sum_j T[i,j,k] * A[i,j] * B[j,k] gives the product)
    # The matmul tensor has entries: for each (i,j,k), the coefficient is 1
    # when we consider the trilinear form.
    
    # The matmul tensor M_{n,m,p} has:
    # M[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # (in the sense that the tensor T has T_{i,j,k} = 1)
    # And the decomposition is: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i]
    
    # Wait, the convention is:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Sum over c should give: sum_j A[i,j]*B[j,k] for the output at (i,k)
    
    # So the tensor entry at (i,j,k) in the trilinear form is 1 for all valid (i,j,k).
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # This is the matmul tensor which has rank at most n*m*p = 52.
    
    # Let me try to construct a rank-16 decomposition.
    # 
    # Approach: Use the structure that for fixed j, the (i,k) submatrix is rank 1
    # (all ones). So T = sum_j (ones_{n} ⊗ e_j ⊗ ones_p)
    # This gives rank m * (something). But each term (ones_n ⊗ e_j ⊗ ones_p) 
    # is a single rank-1 tensor if we index properly.
    # 
    # Actually T[i,j,k] = 1 for all i,j,k.
    # T = sum over j of: (vector of ones in i) ⊗ (e_j in j) ⊗ (vector of ones in k)
    # But e_j in j is not a rank-1 factor in the (j,k) space...
    
    # Let me think differently. The tensor T in the space U⊗V⊗W where:
    # U-space has basis e_{i,j} for i in 0..1, j in 0..1
    # V-space has basis e_{j,k} for j in 0..1, k in 0..12
    # W-space has basis e_{k,i} for k in 0..12, i in 0..1
    # T(e_{i,j}, e_{j',k}, e_{k',i'}) = 1 if j=j' and i=i' and k=k', else 0
    # Wait no - the matmul tensor is:
    # T[i,j,k] = 1 (it's the all-ones tensor in the appropriate indices)
    
    # The decomposition requires:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Let me try: for each k in 0..12, use one term where:
    # U[i*m+j, c_k] = 1 for all i,j
    # V[j*p+k, c_k] = 1 for all j
    # W[k'*n+i, c_k] = 1 if k'=k else 0
    # Then sum over c: for term c_k, contributes U[i*m+j,c_k]*V[j*p+k,c_k]*W[k*n+i,c_k]
    # = 1*1*1 = 1 only when the k in W matches.
    # But we need the sum to be 1 for ALL (i,j,k), and with this construction:
    # For a given (i,j,k), only term c_k contributes (since W only nonzero at k'=k)
    # So the sum = 1. This works! Rank = 13.
    #
    # Wait, but U[i*m+j, c_k] = 1 for ALL i,j means U is the same for all (i,j)
    # in each term. Let me verify:
    # Term c_k (k=0..12):
    #   U[i*m+j, c_k] = 1 for all i in {0,1}, j in {0,1}
    #   V[j*p+k, c_k] = 1 for all j in {0,1}
    #   W[k'*n+i, c_k] = 1 if k'=k else 0, for all i in {0,1}
    #
    # For output (i,j,k):
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    #   = sum_k' U[i*m+j,c_{k'}] * V[j*p+k,c_{k'}] * W[k*n+i,c_{k'}]
    #   U[i*m+j,c_{k'}] = 1 always
    #   V[j*p+k,c_{k'}] = 1 if k=k' else 0  (since V[j*p+k', c_k'] = 1 only when the k index matches)
    #   Wait: V has shape (m*p, rank). V[j*p+k, c] is the entry at row j*p+k.
    #   For term c_{k'}, V[j*p+k, c_{k'}] = 1 if k == k' (because we set V[j*p+k', c_{k'}] = 1 for all j)
    #   So V[j*p+k, c_{k'}] = 1 if k=k' else 0.
    #   W[k*n+i, c_{k'}] = 1 if k=k' else 0 (by construction W[k'*n+i, c_{k'}] = 1 for all i when k'=k')
    #   So W[k*n+i, c_{k'}] = 1 if k=k' else 0.
    #
    # Therefore the sum = sum_k' [1 * (1 if k=k' else 0) * (1 if k=k' else 0)]
    #                     = sum_k' (1 if k=k' else 0)
    #                     = 1 (since there's exactly one k' equal to k)
    #
    # This gives rank 13! Let me implement this.
    
    # Actually wait - I need to double check. The condition is:
    # V[j*p+k, c_{k'}] should be 1 when the k in V's row index equals k'.
    # V's row index is j*p + k, so the k component is (j*p+k) % p = k.
    # So V[j*p+k, c_{k'}] = 1 if k == k'. Yes.
    #
    # And W's row index is k'*n + i, so the k component is (k'*n+i) // n = k'.
    # W[k*n+i, c_{k'}] = 1 if k == k'. Yes.
    #
    # So this works with rank 13!
    
    rank = p  # 13
    U = [[0] * rank for _ in range(n * m)]  # (4, 13)
    V = [[0] * rank for _ in range(m * p)]  # (26, 13)
    W = [[0] * rank for _ in range(p * n)]  # (26, 13)
    
    for c in range(p):
        # U: all ones for this term
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = 1
        
        # V: 1 for all j when k == c
        for j in range(m):
            for k in range(p):
                if k == c:
                    V[j * p + k][c] = 1
        
        # W: 1 for all i when k == c
        for k in range(p):
            for i in range(n):
                if k == c:
                    W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
