import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Build the multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid index
    # T has shape (n, m, p) and T[i,j,k] = 1 for all valid (i,j,k)
    # We want to find rank-1 tensors u ⊗ v ⊗ w such that sum of products = T
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # A rank-1 decomposition term is u[i] * v[j] * w[k]
    # We need sum_c u_c[i] * v_c[j] * w_c[k] = 1 for all i,j,k
    
    # This is equivalent to: the matrix T_k[i,j] = 1 for each k, and we need
    # sum_c u_c[i] * w_c[k] * v_c[j] = 1
    
    # Actually, let's think differently. The matrix multiplication tensor is:
    # T[i,j,k] = delta[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # This means T[i,j,k] = 1 for all valid (i,j,k) since each (i,j,k) triple is valid
    
    # Wait, that's not right. The tensor for matrix multiplication has T[i,j,k] = 1 
    # when the product A[i,j]*B[j,k] appears in C[i,k]. So T[i,j,k] = 1 for all (i,j,k).
    
    # So we need: sum_c U_c[i] * V_c[j] * W_c[k] = 1 for all i,j,k
    
    # This is a tensor of all ones! The rank of the all-ones tensor of size n*m*p
    # is 1 (just take u[i]=1, v[j]=1, w[k]=1).
    
    # But that's wrong - let me reconsider. The matrix multiplication tensor is NOT
    # the all-ones tensor. Let me re-examine.
    
    # For matrix multiplication C = A * B where C is n x p, A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 for each (i,j,k) that appears in the sum.
    # Since for each (i,k), the sum is over j, we have T[i,j,k] = 1 for all i,j,k.
    
    # Hmm, but then the rank would be 1, which is clearly wrong. Let me think again.
    
    # Oh wait, I think I'm confusing things. The tensor decomposition for matrix 
    # multiplication is:
    # C[i,k] = sum_c (sum_j U_c[i,j] * A[i,j]) * (sum_j V_c[j,k] * B[j,k])
    # No, that's not right either.
    
    # The correct formulation: we want to write
    # C[i,k] = sum_c U_c[i] * V_c[k] * (intermediate)
    # Actually, the standard formulation is:
    # C[i,k] = sum_c (sum_j U_c[i,j] * A[i,j]) * (sum_j V_c[j,k] * B[j,k])
    # No...
    
    # Let me use the standard formulation. We have three vectors u, v, w for each term:
    # u is indexed by (i,j) pairs, v is indexed by (j,k) pairs, w is indexed by (k,i) pairs
    # The tensor entry T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i]
    # And we need T[i,j,k] = 1 for all valid (i,j,k)
    
    # Wait, that gives us the all-ones tensor again, which has rank 1.
    
    # I think I'm making an error. Let me look at this more carefully.
    
    # The matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * T_{ijk}
    # where T_{ijk} = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give
    # the (i,k) entry of C.
    
    # So T_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # And a rank-1 decomposition term is:
    # u ⊗ v ⊗ w where u is indexed by i (output row), v by j (inner dimension), w by k (output col)
    # No wait...
    
    # Let me just use the schoolbook approach which is known to work, and try to improve it.
    
    # Schoolbook: rank = n*m*p = 4*4*7 = 112
    # Each term corresponds to one (i,j,k) triple
    
    # Can we do better? The Strassen algorithm reduces the rank for 2x2 matrices.
    # For 4x4, we can use block Strassen or other algorithms.
    
    # For now, let's implement the schoolbook and verify it works, then try optimizations.
    
    terms = []
    
    def make_u(idx_i, idx_j, val):
        u = [0] * (n * m)
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val):
        v = [0] * (m * p)
        v[idx_j * p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val):
        w = [0] * (p * n)
        w[idx_k * n + idx_i] = val
        return w
    
    # Schoolbook: rank 112
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = make_u(i, j, 1)
                v = make_v(j, k, 1)
                w = make_w(k, i, 1)
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
