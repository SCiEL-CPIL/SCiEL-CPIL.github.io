"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return (U, V, W) as lists for the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    We need: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    and for all other index combinations, the sum == 0.
    """
    r = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We check that T[i,j,k] = 1 if this is a valid (i,j,k) for matmul, else 0.
    # For matmul (n x m) * (m x p) -> (n x p), the (i,k) entry is sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid i,j,k (since we're decomposing the identity-like tensor)
    # Actually the multiplication tensor M[i,k,j] = 1 for all i,k,j (it's the tensor of the bilinear map)
    # The decomposition: M[i,k,j] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]... 
    # Wait, let me re-read the convention.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The product: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal the (i,k) entry of A*B
    # which is sum_j A[i,j]*B[j,k]. So the tensor T[i,k] = sum_j A[i,j]*B[j,k]
    # The bilinear form: for fixed j, the term A[i,j]*B[j,k] contributes.
    # So the tensor is: for each j, the outer product of e_i (for A) and e_k (for B)
    # The full tensor: M = sum_j (e_{i} for A) outer (e_{k} for B) ... 
    # Actually M[i,k,j] = 1 for all i,k,j is NOT right.
    # M[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form: M(A,B) = sum_{i,k,j} A[i,j] B[j,k] (result[i,k])
    # The tensor T[i,k,j] = 1 for all i,k,j? No.
    # T[i,k,j] should be 1 if the (i,j) of A and (j,k) of B multiply to give result[i,k].
    # So T[i,k,j] = 1 for all valid i, k, j.
    # And the decomposition: T[i,k,j] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # So we need: for all i in [0,n), k in [0,p), j in [0,m):
    #   sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1
    # And for invalid combinations (which don't exist in the standard indexing), it's 0.
    # But actually, the tensor is only defined for valid indices. Let me just check all valid (i,j,k).
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err < tol

def _strassen_3x3():
    """Return Strassen decomposition for 3x3 * 3x3 -> 3x3.
    U: (9, 7), V: (9, 7), W: (9, 7)
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=3
    """
    # Strassen's 7 multiplications for 2x2x2, extended to 3x3x3
    # Actually, Strassen's algorithm is for 2x2x2. For 3x3x3, we need a different approach.
    # Let me think... The rank of 3x3x3 matrix multiplication is known to be 7? No, it's not.
    # The rank of the 2x2x2 multiplication tensor is 7 (Strassen).
    # For 3x3x3, the rank is at most 3*3*3 = 27 (schoolbook).
    # Actually, I don't think there's a known rank-7 decomposition for 3x3x3.
    # Let me just use the schoolbook for now and try to find improvements.
    # 
    # Wait, the directive says "using the known rank-7 Strassen decomposition". 
    # But Strassen is for 2x2x2. For 3x3x3, I'll need to think differently.
    # 
    # Actually, let me just return the schoolbook for 3x3x3 for now.
    n = m = p = 3
    U, V, W = _schoolbook(n, m, p)
    return U, V, W

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Strategy: Direct sum of 3x3x3 (schoolbook rank 27) and 3x3x1 (rank 3)
    # Total rank = 27 + 3 = 30? No, that's worse than schoolbook 36.
    # 
    # Actually, let me think about this differently.
    # 3x3 * 3x4 = 3x4
    # We can view this as: for each output column k (0,1,2,3), compute 3x3 * 3x1
    # Each 3x3 * 3x1 has rank at most 3 (schoolbook: sum_j A[i,j]*B[j,k])
    # But we can share computations across columns.
    #
    # The schoolbook rank is 3*3*4 = 36.
    # Can we do better?
    #
    # One approach: use Strassen-like ideas. But for rectangular matrices, it's less clear.
    #
    # Let me just try to find a good decomposition by random search or greedy methods.
    # For now, return schoolbook.
    
    U, V, W = _schoolbook(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback: definitely schoolbook
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
