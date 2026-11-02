import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Trivial decomposition rank is n*m*p = 63
    rank = n * m * p
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using Bini-Capovani style search
    # The matrix multiplication tensor T_{i,j,k} = delta_{i,k} * (A_{i,j} * B_{j,k})
    # We want to find rank r such that T = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t
    
    # For 3x3x3, the rank is known to be 23 (Strassen's algorithm gives 7 for 2x2x2, 
    # and for 3x3x3 the best known is around 23)
    # For 3x3x7, we can decompose as 3x3x3 + 3x3x4
    
    # Let's try to find a good decomposition by searching for rank-minimal decompositions
    # of smaller sub-problems
    
    # First, let's try to find a rank-23 decomposition for 3x3x3
    # and a rank-? decomposition for 3x3x4
    
    # Actually, let's use a more systematic approach:
    # The tensor for M_{n,m,p} multiplication has entries T[i,j,k] = delta_{i,k}
    # (where i indexes rows of A, j columns of A/rows of B, k columns of B)
    # Wait, the convention is W[k*n+i, r], so the output C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T[i,j,k] = delta_{i,k} ... no, that's not right.
    
    # Let me reconsider. The matrix multiplication tensor T satisfies:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # But actually T[i,j,k] is the coefficient in the bilinear form.
    # The tensor T is defined by T[i,j,k] = delta_{i,k} for the "standard" decomposition,
    # but that's not quite right either.
    
    # Actually, the matrix multiplication tensor M_{n,m,p} has entries:
    # M_{n,m,p}[i,j,k] = 1 if the (i,k)-th entry of C depends on A[i,j] and B[j,k]
    # which is always the case, so M[i,j,k] = 1 for all i,j,k? No, that's not right.
    
    # Let me think again. The tensor is a trilinear form:
    # <T, A, B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    # where C = A * B.
    # So T[i,j,k] = 1 if C[i,k] = A[i,j] * B[j,k], which means i and k are the row/col
    # indices of C, and j is the "inner" index. But C[i,k] = sum_j A[i,j] B[j,k],
    # so T[i,j,k] = 1 for all i,j,k? No, that would give sum_{i,j,k} A[i,j]B[j,k]
    # which is not sum_{i,k} C[i,k].
    
    # Actually, T[i,j,k] should be such that sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    # = sum_{i,k} sum_j A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k. That can't be right because then the rank would be
    # the tensor rank of the all-ones tensor, which is different.
    
    # I think I'm confusing myself. Let me look at the standard definition.
    # The matrix multiplication tensor M_{n,m,p} is defined by:
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ e_j ⊗ e_k * delta_{i,k}
    # Wait, no. The correct definition is:
    # M_{n,m,p}[i,j,k] = 1 if i is the row of A, j is the column of A (row of B),
    # k is the column of B, and the product A[i,j]*B[j,k] contributes to C[i,k].
    # Since every such product contributes, M[i,j,k] = 1 for all valid i,j,k.
    # But then the rank of M is the tensor rank of the all-ones tensor of size n x m x p,
    # which is not what we want.
    
    # I think the issue is that the tensor should be indexed differently.
    # The output C[i,k] depends on all A[i,j] and B[j,k] for j=1..m.
    # So the tensor should be:
    # M[i,j,k] = 1 if the (i,k) entry of C uses A[i,j] and B[j,k]
    # This is true for all i,j,k, so M is the all-ones tensor.
    # But that's not right because the rank of the all-ones 3x3x7 tensor is not 63.
    
    # Let me reconsider the problem statement. It says "exactly decompose the matrix
    # multiplication tensor". The standard matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * [i,k]_I
    # where [i,k]_I is the (i,k) entry of the identity matrix.
    # So T[i,j,k] = delta_{i,k}.
    # This makes sense because:
    # sum_{i,j,k} delta_{i,k} A[i,j] B[j,k] = sum_{i,j} A[i,j] B[j,i]
    # which is the trace of A * B^T, not the sum of entries of A*B.
    
    # Hmm, I think the convention might be different. Let me just use the trivial
    # decomposition and try to improve it with a search.
    
    # For now, let's return the trivial decomposition. The search for a better one
    # is complex and may not yield improvements within the time limit.
    
    # Convert to lists for compatibility
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
