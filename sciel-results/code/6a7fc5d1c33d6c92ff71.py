"""Matrix multiplication 3x4x8 tensor decomposition."""
DIMENSIONS = [3, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Schoolbook baseline: rank 96
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0), dtype=np.float64)
    V0 = np.zeros((m * p, rank0), dtype=np.float64)
    W0 = np.zeros((p * n, rank0), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    result = {"U": to_list(U0), "V": to_list(V0), "W": to_list(W0)}
    
    if time.time() > deadline - 5.0:
        return result
    
    # Verify schoolbook
    def verify(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify(U0, V0, W0):
        return result
    
    # Try to reduce rank using structured search
    # Strategy: For each pair of terms that share the same (i,j) or (j,k) or (k,i),
    # try to merge them using bilinear identities.
    
    # Better approach: Use the fact that for fixed i and j, varying k gives
    # terms that all have the same U entry. We can try to express combinations.
    
    # Let's try a direct search for lower rank using random initialization + local optimization
    # This is complex, so let's try a simpler heuristic:
    
    # For 3x4x8, try rank 88, 80, 72, etc. by constructing structured decompositions.
    
    # Approach: Split j into {0,1} and {2,3}. For each half, use a rank-24 decomposition
    # of the 3x2x8 sub-problem. Total rank 48. But 3x2x8 has rank 48, so no gain.
    
    # Alternative: Use the identity that a 2x2x2 tensor (matrix mult 2x2x2) has rank 7
    # (Strassen). But our sub-problems are 3x2x8, not 2x2x2.
    
    # Let's try: for each i, the sub-tensor T[i,j,k] = 1 for all j,k is a 4x8 all-ones tensor.
    # The rank of the all-ones tensor of size a x b is 1 (as a matrix), but as part of the 3D
    # tensor with the j,k structure, it's different.
    
    # Actually, the key structure: T[i,j,k] = 1 for all i,j,k. This is the all-ones tensor.
    # The rank of the all-ones tensor of size n x m x p is 1! Because:
    # T[i,j,k] = 1 = (1)(1)(1) for all i,j,k.
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all i,j,k.
    # This gives rank 1!
    
    # Wait, that can't be right. Let me check:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For rank 1: U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0]
    # If U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1, then the product is 1 for all i,j,k.
    # So the all-ones tensor has rank 1!
    
    # But that's not the matrix multiplication tensor. The matrix multiplication tensor
    # T[i,j,k] is 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor entry T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all i,j,k. Yes, it's the all-ones tensor!
    
    # So the rank is 1?! That seems too good to be true. Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix
    # by a 4 by 8 matrix."
    
    # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 for all i,j,k,
    # and the bilinear form is sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k].
    # But C[i,k] = sum_j A[i,j] B[j,k], so the full bilinear form is sum_{i,k} sum_j A[i,j]B[j,k].
    # This is correct.
    
    # So the tensor is indeed all-ones, and its rank is 1.
    
    # Let me verify: U has shape (12, 1), V has shape (32, 1), W has shape (24, 1).
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 for all k,i
    # Then sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1 for all i,j,k.
    # This matches T[i,j,k] = 1.
    
    # So the answer is rank 1!
    
    U1 = np.ones((n * m, 1), dtype=np.float64)
    V1 = np.ones((m * p, 1), dtype=np.float64)
    W1 = np.ones((p * n, 1), dtype=np.float64)
    
    if verify(U1, V1, W1):
        return {"U": to_list(U1), "V": to_list(V1), "W": to_list(W1)}
    
    # Fallback to schoolbook
    return result
# EVOLVE-BLOCK-END
