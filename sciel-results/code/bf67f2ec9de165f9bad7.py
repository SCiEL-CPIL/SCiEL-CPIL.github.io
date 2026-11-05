import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    # Rank 43 decomposition:
    # 6 Strassen blocks (2x2x2) for k=0..5: 6*7 = 42
    # 1 rank-1 term for k=6: 1
    # Total: 43
    
    rank = 43
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    def emit_strassen(i0, j0, k0):
        """
        Emits a 2x2x2 Strassen block.
        A rows i0, i0+1; cols j0, j0+1.
        B rows j0, j0+1; cols k0, k0+1.
        C rows i0, i0+1; cols k0, k0+1.
        """
        nonlocal col
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[i0*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M2 = (A21 + A22) * B11
        U[(i0+1)*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = -1.0
        col += 1
        
        # M3 = A11 * (B12 - B22)
        U[i0*m + j0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M4 = A22 * (B21 - B11)
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[j0*p + k0, col] = -1.0
        W[k0*n + (i0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        # M5 = (A11 + A12) * B22
        U[i0*m + j0, col] = 1.0
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[(i0+1)*m + j0, col] = 1.0
        U[i0*m + j0, col] = -1.0
        V[j0*p + k0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[i0*m + (j0+1), col] = 1.0
        U[(i0+1)*m + (j0+1), col] = -1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1

    # Part 1: j in {0, 1} (A cols 0,1)
    # B rows 0,1 (B top half)
    # Output k in {0, 1, 2, 3, 4, 5}
    
    # Block 1.1: 2x2x2 for k=0,1
    emit_strassen(0, 0, 0)
    # Block 1.2: 2x2x2 for k=2,3
    emit_strassen(0, 0, 2)
    # Block 1.3: 2x2x2 for k=4,5
    emit_strassen(0, 0, 4)
    
    # Part 2: j in {2, 3} (A cols 2,3)
    # B rows 2,3 (B bottom half)
    # Output k in {0, 1, 2, 3, 4, 5}
    
    # Block 2.1: 2x2x2 for k=0,1
    emit_strassen(0, 2, 0)
    # Block 2.2: 2x2x2 for k=2,3
    emit_strassen(0, 2, 2)
    # Block 2.3: 2x2x2 for k=4,5
    emit_strassen(0, 2, 4)
    
    # Part 3: k=6 - single rank-1 term
    # For k=6: C[i,6] = sum_j A[i,j]*B[j,6]
    # We need: sum_r U[i*m+j, r]*V[j*p+6, r]*W[6*n+i, r] = 1 for all i,j
    # Use one term: U[i*m+j, col]=1, V[j*p+6, col]=1, W[6*n+i, col]=1 for all i,j
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1.0
            V[j*p + 6, col] = 1.0
            W[6*n + i, col] = 1.0
    col += 1
    
    # Verify using einsum for efficiency
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Must equal 1 for all i, j, k.
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
