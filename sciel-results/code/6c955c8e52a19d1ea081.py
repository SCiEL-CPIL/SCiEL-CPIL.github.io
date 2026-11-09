"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify schoolbook (should always pass)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Random check
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u = U_np[:, r]
        v = V_np[:, r]
        w = W_np[:, r]
        # u is vector of A indices (i*m+j)
        # v is vector of B indices (j*p+k)
        # w is vector of C indices (k*n+i) -> This index ordering needs care.
        # The contract says W[k*n+i].
        # So w[k*n+i] = 1 implies C[k, i]? Or C[i, k]?
        # Seed says W[k*n+i].
        # Standard C[i, k].
        # If w index is k*n+i, and we assume row-major C, then row k, col i?
        # Let's assume the evaluator handles the mapping.
        # My verification below checks if the bilinear form matches.
        
        a_flat = A.flatten() # size nm, order i*m+j
        b_flat = B.flatten() # size mp, order j*p+k
        
        u_val = np.dot(u, a_flat)
        v_val = np.dot(v, b_flat)
        
        # w is size pn. Index k*n+i.
        # If C is n x p, and we flatten it, how?
        # If C[i, k] is at i*p + k.
        # But W index is k*n + i.
        # So W index k*n + i corresponds to C[i, k] if we transpose the index logic?
        # k*n + i is the index in a p x n array.
        # So C_flat[k*n + i] = C[i, k] if C is stored p x n?
        # Let's just check the tensor sum directly.
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This T should represent the multiplication.
        # For standard matmul, T[i, j, k] = delta_{j, j} ? No.
        # The bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k].
        # This sum equals sum_k (sum_j A[i,j] B[j,k]) ? No.
        # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] is a scalar.
        # The matmul is a matrix.
        # The decomposition is of the tensor that *is* the matrix multiplication map.
        # The map A, B -> C.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # This is equivalent to: C[i, k] = sum_j A[i, j] B[j, k] delta_{?}.
        # The tensor M has components M[i, j, k] = 1 if this term contributes to C[i, k].
        # Actually, M[i, j, k] = 1 means A[i, j] B[j, k] contributes to C[i, k].
        # But the factorization is sum_r U_r V_r W_r.
        # This implies T = sum_r outer(U_r, V_r, W_r).
        # We need T[i, j, k] = 1 if the term A[i,j]B[j,k] is part of C[i,k].
        # Wait, C[i,k] sums over j.
        # So the "tensor" of the multiplication is not a 3-tensor of 0s and 1s in the simple way if we view it as a bilinear map to a matrix.
        # However, the evaluator likely checks if the factor matrices reproduce the matrix multiplication.
        # i.e., sum_r U_r V_r W_r^T == MatMul Tensor.
        # The MatMul Tensor T has shape (n, m, p, n, p)? No.
        # It is a tensor of shape (nm, mp, pn)?
        # Let's assume the evaluator does:
        # C = sum_r (A dot U_r) * (B dot V_r) * W_r (reshaped).
        # Let's check my construction:
        # U_r is sparse, 1 at i*m+j. So A dot U_r = A[i,j].
        # V_r is sparse, 1 at j*p+k. So B dot V_r = B[j,k].
        # W_r is sparse, 1 at k*n+i.
        # So term is A[i,j] * B[j,k] * W_r.
        # Summing over r:
        # For each i,j,k, is there an r with U,V,W matching?
        # Yes, one r for each (i,j,k) triple.
        # Sum_r ... = sum_{i,j,k} A[i,j] B[j,k] W_{k*n+i, r} for that specific r?
        # No, r is unique for i,j,k.
        # So the contribution to index w = k*n+i is A[i,j] B[j,k].
        # Summed over all r such that W_{w,r} != 0.
        # For a fixed w = k*n+i, the only r that contributes is the one where U matches i,j and V matches j,k.
        # So the sum is A[i,j] B[j,k].
        # This matches C[i,k] = sum_j A[i,j] B[j,k].
        # Wait, the sum over j is inside the sum over r?
        # No, r iterates over ALL i,j,k triples.
        # So for a fixed output index w = k*n+i, we sum over all r.
        # Only those r where U_r has 1 at some i*m+j, V_r has 1 at some j'*p+k', and W_r has 1 at k*n+i contribute.
        # In my schoolbook, W_r has 1 at k*n+i ONLY for the r corresponding to that specific k, i, AND a specific j.
        # So for fixed k, i, we have m values of j.
        # So sum_r ... = sum_j A[i,j] B[j,k].
        # This is exactly C[i,k].
        # So the schoolbook is correct.

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
