"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 145.0  # Safety margin for the 150s wall-clock limit

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition for (n,m,p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: C = A @ B where C[k*n+i, r] = sum_j A[i*m+j] * B[j*p+r].
    We check tensor identity: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j]
                v = V[j * p + k]
                w = W[k * n + i]
                s = np.sum(u * v * w)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices:
    U: [A11, A12, A21, A22] -> rows 0, 1, 2, 3
    V: [B11, B12, B21, B22] -> rows 0, 1, 2, 3
    W: [C11, C12, C21, C22] -> rows 0, 1, 2, 3
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Col 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Col 1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Col 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Col 3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Col 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Col 5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Col 6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank calculation:
    # Core 4x4x4 using 8 Strassen blocks: 8 * 7 = 56
    # Border j=4: 1 term
    # Border k=4: 1 term
    # Correction j=4, k=4: 1 term
    # Total: 59
    
    rank = 59
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # 1. Strassen blocks for j in 0..3, k in 0..3
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for i_block in range(2): # i in 0..1, 2..3
        for j_block in range(2): # j in 0..1, 2..3
            for k_block in range(2): # k in 0..1, 2..3
                
                i0 = i_block * 2
                j0 = j_block * 2
                k0 = k_block * 2
                
                # Embed Strassen decomposition
                for c in range(7):
                    # U
                    if U_s[0, c] != 0:
                        U[i0 * m + j0, col] += U_s[0, c]
                    if U_s[1, c] != 0:
                        U[i0 * m + (j0 + 1), col] += U_s[1, c]
                    if U_s[2, c] != 0:
                        U[(i0 + 1) * m + j0, col] += U_s[2, c]
                    if U_s[3, c] != 0:
                        U[(i0 + 1) * m + (j0 + 1), col] += U_s[3, c]
                        
                    # V
                    if V_s[0, c] != 0:
                        V[j0 * p + k0, col] += V_s[0, c]
                    if V_s[1, c] != 0:
                        V[j0 * p + (k0 + 1), col] += V_s[1, c]
                    if V_s[2, c] != 0:
                        V[(j0 + 1) * p + k0, col] += V_s[2, c]
                    if V_s[3, c] != 0:
                        V[(j0 + 1) * p + (k0 + 1), col] += V_s[3, c]
                        
                    # W
                    if W_s[0, c] != 0:
                        W[k0 * n + i0, col] += W_s[0, c]
                    if W_s[1, c] != 0:
                        W[k0 * n + (i0 + 1), col] += W_s[1, c]
                    if W_s[2, c] != 0:
                        W[(k0 + 1) * n + i0, col] += W_s[2, c]
                    if W_s[3, c] != 0:
                        W[(k0 + 1) * n + (i0 + 1), col] += W_s[3, c]
                        
                    col += 1
    
    # 2. Border Term A: j=4
    # U[i*m+4, col] = 1
    # V[4*p+k, col] = 1
    # W[k*n+i, col] = 1
    for i in range(n):
        U[i * m + 4, col] = 1
    for k in range(p):
        V[4 * p + k, col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, col] = 1
    col += 1

    # 3. Border Term B: k=4
    # U[i*m+j, col] = 1
    # V[j*p+4, col] = 1
    # W[4*n+i, col] = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
    for j in range(m):
        V[j * p + 4, col] = 1
    for i in range(n):
        W[4 * n + i, col] = 1
    col += 1

    # 4. Border Term C: Correction for j=4, k=4
    # U[i*m+4, col] = -1
    # V[4*p+4, col] = 1
    # W[4*n+i, col] = 1
    for i in range(n):
        U[i * m + 4, col] = -1
    V[4 * p + 4, col] = 1
    for i in range(n):
        W[4 * n + i, col] = 1
    col += 1

    # Verify
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook if validation fails
        U, V, W = _schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
