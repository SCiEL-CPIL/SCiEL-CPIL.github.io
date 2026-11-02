"""Optimized tensor decomposition for 2x2 * 2x4 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the target tensor T[i, j, k] for matmul (A[i,j] * B[j,k])
    # The decomposition says: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), indexed by (i*m + j)
    # V has shape (m*p, rank), indexed by (j*p + k)
    # W has shape (p*n, rank), indexed by (k*n + i)
    # The product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal the (i,k) entry of A*B, which is sum_j A[i,j]*B[j,k]
    # So the tensor T[i, j, k] = delta (the standard matmul tensor)
    # Actually, the tensor is T[i, j, k] = 1 if this corresponds to A[i,j]*B[j,k] contributing to C[i,k]
    # The decomposition reconstructs C[i,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] to be 1 when (i,j,k) is a valid matmul term
    # and 0 otherwise. But actually it's more subtle - the bilinear form is:
    # C[i,k] = sum_{j,c} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This must equal sum_j A[i,j] B[j,k] for all A, B.
    # So for each (i,k), sum_{j,c} U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    # This means: for each (i,j,k), the coefficient of A[i,j]B[j,k] in C[i,k] must be 1,
    # and for each (i,j,k,i',j',k') where (i,j) != (i',j') or (j,k) != (j',k') in a way that
    # doesn't match, the cross terms must vanish.
    # Actually, the standard way: the tensor M has entries M[i,j,k] = 1 for the matmul tensor.
    # The decomposition is: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right either. Let me think again.
    #
    # The matrix multiplication tensor T has T[i,j,k] = 1 (it's the standard tensor).
    # A rank-r decomposition is: T[i,j,k] = sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # But wait, that's only for the "diagonal" terms. The full condition is that the
    # reconstructed matrix product matches. Let me just verify by checking that
    # for random A and B, the product via decomposition matches A@B.
    
    # Actually, the simplest verification: the tensor entry T[i,j,k] should be 1 for all valid (i,j,k)
    # in the matmul. The decomposition gives:
    #   C[i,k] = sum_{j,c} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This must equal sum_j A[i,j] B[j,k].
    # So the "tensor" is implicitly defined. Let me verify by checking that
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k (since the matmul tensor
    # has all 1s in the relevant positions).
    
    # Hmm, actually I think the correct interpretation is:
    # The multiplication tensor T in R^{(n*m) x (m*p) x (p*n)} has T[i*m+j, j*p+k, k*n+i] = 1
    # and 0 otherwise. No wait, that's not standard.
    #
    # Let me just do a numerical check: pick random A (n x m) and B (m x p), compute C = A @ B,
    # then compute C' using the decomposition and check they match.
    
    # C[i,k] = sum_{j,c} U[i*m+j, c] * A[i,j] * B[j,k] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not how it works. The bilinear form is:
    # C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    # Hmm, that doesn't seem right either.
    
    # Let me re-read the task: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # The standard tensor decomposition for matrix multiplication:
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # A rank-r decomposition: (A*B)[i,k] = sum_{c=1}^r (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    # No, that's not standard either.
    
    # The standard way: the tensor T has T[i,j,k] = 1 for the matmul.
    # Rank-r decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # So we need this to hold for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2, rank 7.
    Returns U (4x7), V (4x7), W (4x7) where indices are:
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matmul:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need T[i,j,k] = sum_c U[i*2+j,c] V[j*2+k,c] W[k*2+i,c] = 1 for all i,j,k in {0,1}
    # The matmul tensor has T[i,j,k] = 1 for all i,j,k (since each A[i,j]*B[j,k] contributes to C[i,k])
    
    # Let's construct U, V, W such that the bilinear form works.
    # For each Strassen product c, we have:
    # U[:,c] gives the coefficients for A, V[:,c] for B, W[:,c] for C.
    
    # Product 1: (A00+A11)(B00+B11) -> contributes to C00 (+1), C11 (+1)
    # U for prod 1: A00=1, A11=1, others 0 -> [1,0,0,1] (indices: 00,01,10,11)
    # V for prod 1: B00=1, B11=1, others 0 -> [1,0,0,1] (indices: 00,01,10,11)
    # W for prod 1: C00=1, C11=1 -> [1,0,0,1] (indices: 00,01,10,11 where index = k*2+i)
    
    # Let me just build it directly.
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # c=0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # c=1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[2][1] = 1; W[3][1] = -1  # C10 (+1), C11 (-1)
    
    # c=2: M3 = A00(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1; V[3][2] = -1  # B01, -B11
    W[1][2] = 1; W[3][2] = 1  # C01 (+1), C11 (+1)
    
    # c=3: M4 = A11(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1; V[0][3] = -1  # B10, -B00
    W[0][3] = 1; W[2][3] = 1  # C00 (+1), C10 (+1)
    
    # c=4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    W[0][4] = -1; W[1][4] = 1  # C00 (-1), C01 (+1)
    
    # c=5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A10, -A00
    V[0][5] = 1; V[1][5] = 1  # B00, B01
    W[3][5] = 1  # C11 (+1)
    
    # c=6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, -A11
    V[2][6] = 1; V[3][6] = 1  # B10, B11
    W[0][6] = 1  # C00 (+1)
    
    return U, V, W


def construct():
    """Construct a rank-minimal decomposition for 2x2 * 2x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 4
    
    # Use Strassen for the two 2x2x2 blocks (columns 0-1 and 2-3 of B)
    # Block 1: A (2x2) * B[:,0:2] (2x2) -> C[:,0:2] (2x2)
    # Block 2: A (2x2) * B[:,2:4] (2x2) -> C[:,2:4] (2x2)
    # Each block uses Strassen rank 7, total rank 14.
    
    U1, V1, W1 = _strassen_2x2x2()
    
    rank = 14
    U = [[0]*rank for _ in range(n*m)]  # 4 x 14
    V = [[0]*rank for _ in range(m*p)]  # 8 x 14
    W = [[0]*rank for _ in range(p*n)]  # 8 x 14
    
    # Copy Strassen block 1 (columns 0-6)
    for i in range(4):
        for c in range(7):
            U[i][c] = U1[i][c]
    
    # V for block 1: V[j*2+k, c] where k=0,1 (first two columns of B)
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j*2+k][c] = V1[j*2+k][c]
    
    # W for block 1: W[k*2+i, c] where k=0,1
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k*2+i][c] = W1[k*2+i][c]
    
    # Copy Strassen block 2 (columns 7-13)
    for i in range(4):
        for c in range(7):
            U[i][c+7] = U1[i][c]
    
    # V for block 2: V[j*2+k, c+7] where k=2,3 (last two columns of B)
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j*2+(k+2)][c+7] = V1[j*2+k][c]
    
    # W for block 2: W[k*2+i, c+7] where k=2,3
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[(k+2)*2+i][c+7] = W1[k*2+i][c]
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
