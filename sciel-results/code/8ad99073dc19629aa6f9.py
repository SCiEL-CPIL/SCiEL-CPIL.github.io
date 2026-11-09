import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Naive decomposition: rank = n * m * p
    # For each (i, j, k) in A, B, C, we have a term A[i,j] * B[j,k] * C[i,k]
    # In the tensor decomposition T = sum_r U[:,r] V[:,r] W[:,r]^T
    # The tensor T is defined by T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    # Wait, the standard matrix multiplication tensor is T[i,j,k] = delta_{j, j} ... no.
    # The tensor for matrix multiplication C = AB is T_{i,j,k} = 1 if the indices align for the product?
    # No, the tensor is the identity map for the bilinear form.
    # Let's look at the interface: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The condition is: for all i, j, k:
    # (sum_{j'} A[i,j'] B[j',k]) must be reconstructed.
    # The evaluator checks if the tensor formed by U, V, W matches the matrix multiplication tensor.
    # The matrix multiplication tensor T has entries T[i, j, k] = 1 if we are looking at the coefficient of A[i,j]B[j,k] in C[i,k]?
    # Actually, the standard tensor decomposition for matmul is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as:
    # C = sum_{r} u_r (A) * v_r (B) * w_r (C)
    # Where for a specific term r corresponding to indices (i0, j0, k0):
    # u_r has 1 at (i0, j0) and 0 elsewhere.
    # v_r has 1 at (j0, k0) and 0 elsewhere.
    # w_r has 1 at (i0, k0) and 0 elsewhere.
    # Then sum_r u_r[i,j] v_r[j,k] w_r[i',k'] = delta_{i,i0} delta_{j,j0} delta_{j,j0} delta_{k,k0} delta_{i',i0} delta_{k',k0}
    # Summing over r (which iterates over all i0, j0, k0), we get:
    # sum_{i0, j0, k0} delta_{i,i0} delta_{j,j0} delta_{j,j0} delta_{k,k0} delta_{i',i0} delta_{k',k0}
    # This equals 1 if i=i', j=j, k=k' and j=j0? No.
    # Let's re-evaluate.
    # The tensor T is such that <T, A \otimes B \otimes C^T> = Tr(C^T A B).
    # Or simply, the decomposition must satisfy:
    # For all A, B: (sum_r (U^T A)[r] (V^T B)[r]) W = A B
    # Let U be n*m x R, V be m*p x R, W be p*n x R.
    # (U^T A) is R x 1. Let a_r = U[:,r]^T A.
    # (V^T B) is R x 1. Let b_r = V[:,r]^T B.
    # Result C = sum_r a_r b_r W[:,r].
    # We want C[i,k] = sum_j A[i,j] B[j,k].
    # If we choose R = n*m*p, and index r by (i0, j0, k0):
    # U[:, r] has 1 at (i0, j0). So a_r = A[i0, j0].
    # V[:, r] has 1 at (j0, k0). So b_r = B[j0, k0].
    # W[:, r] has 1 at (k0, i0) ? Or (i0, k0)?
    # The output W is indexed by (p*n). The trace convention is W[k*n+i, r].
    # So W index is k*n + i.
    # We want the contribution to C[i0, k0] to be A[i0, j0] B[j0, k0].
    # So W[:, r] should have 1 at index (k0, i0) in the flattened W?
    # The problem states: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies W is a matrix of size (p*n) x R.
    # The entry W[k*n + i, r] contributes to C[i, k]?
    # Let's check the trace convention. Usually, vec(C) = (I \otimes A) vec(B) ...
    # If W[k*n+i, r] is the coefficient for C[i,k], then:
    # C[i,k] = sum_r (sum_{i',j'} U[i'*m+j', r] A[i',j']) (sum_{j',k'} V[j'*p+k', r] B[j',k']) W[k*n+i, r]
    # For the naive term r = (i0, j0, k0):
    # U part: 1 at (i0, j0). Sum is A[i0, j0].
    # V part: 1 at (j0, k0). Sum is B[j0, k0].
    # W part: 1 at (k0, i0) in the sense of W[k0*n + i0, r] = 1.
    # Then C[i,k] gets contribution A[i0,j0]B[j0,k0] * 1 if i=i0 and k=k0.
    # Summing over all r (i0,j0,k0), we get C[i,k] = sum_{j0} A[i,j0] B[j0,k].
    # This is correct.
    
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, idx] = 1
                # V index: j*p + k
                V[j * p + k, idx] = 1
                # W index: k*n + i
                W[k * n + i, idx] = 1
                idx += 1
                
    if time.time() - start_time > 100:
        # Safety check, though this is instant
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
