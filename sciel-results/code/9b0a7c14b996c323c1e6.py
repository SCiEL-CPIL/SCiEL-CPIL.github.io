"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strategy: Use Strassen (rank 7) for a 2x2x2 sub-block, schoolbook for the rest
    # Split: A is 4x5, B is 5x6, C is 4x6
    # Use rows 0-1 of A, cols 0-1 of B, rows 0-1 of C for Strassen (2x2x2, rank 7)
    # Remaining: schoolbook for all other (i,j,k) triples
    
    # Strassen 2x2x2 decomposition
    # U22: shape (4, 7) - rows indexed by i*2+j for i,j in {0,1}
    # V22: shape (4, 7) - rows indexed by j*2+k for j,k in {0,1}
    # W22: shape (4, 7) - rows indexed by k*2+i for k,i in {0,1}
    
    U22 = np.zeros((4, 7))
    V22 = np.zeros((4, 7))
    W22 = np.zeros((4, 7))
    
    # M1 = (A11+A22)*(B11+B22) -> C11 + C22
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[3, 0] = 1
    
    # M2 = (A21+A22)*B11 -> C21
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[2, 1] = 1
    
    # M3 = A11*(B11-B21) -> C12 - C22
    U22[0, 2] = 1
    V22[0, 2] = 1; V22[2, 2] = -1
    W22[1, 2] = 1; W22[3, 2] = -1
    
    # M4 = A22*(B22-B12) -> -C11 + C12
    U22[3, 3] = 1
    V22[3, 3] = 1; V22[1, 3] = -1
    W22[0, 3] = -1; W22[1, 3] = 1
    
    # M5 = (A11+A12)*B22 -> C11 + C21
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[0, 4] = 1; W22[2, 4] = 1
    
    # M6 = (A21-A11)*(B11+B12) -> C22
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[1, 5] = 1
    W22[3, 5] = 1
    
    # M7 = (A12-A22)*(B21+B22) -> C11 - C22
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[2, 6] = 1; V22[3, 6] = 1
    W22[0, 6] = 1; W22[3, 6] = -1
    
    # Verify 2x2x2 Strassen
    def verify_decomp(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U22, V22, W22, 2, 2, 2):
        # Fall back to schoolbook for 2x2x2
        U22 = np.zeros((4, 8))
        V22 = np.zeros((4, 8))
        W22 = np.zeros((4, 8))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U22[i*2+j, col] = 1
                    V22[j*2+k, col] = 1
                    W22[k*2+i, col] = 1
                    col += 1
    
    strassen_rank = U22.shape[1]
    
    # Count schoolbook terms for remaining (i,j,k) not in Strassen block
    # Strassen covers i in {0,1}, j in {0,1}, k in {0,1}
    # Total triples: 4*5*6 = 120
    # Strassen covers: 2*2*2 = 8 triples
    # Schoolbook terms: 120 - 8 = 112
    
    total_rank = strassen_rank + 112
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Fill Strassen part (columns 0 to strassen_rank-1)
    # Strassen handles i in {0,1}, j in {0,1}, k in {0,1}
    # U row index: i*m + j = i*5 + j
    # V row index: j*p + k = j*6 + k
    # W row index: k*n + i = k*4 + i
    
    for r in range(strassen_rank):
        # U22 row index is i*2 + j for i,j in {0,1}
        for ui in range(2):
            for uj in range(2):
                u22_idx = ui * 2 + uj
                if U22[u22_idx, r] != 0:
                    U[ui * m + uj, r] = U22[u22_idx, r]
        
        # V22 row index is j*2 + k for j,k in {0,1}
        for vj in range(2):
            for vk in range(2):
                v22_idx = vj * 2 + vk
                if V22[v22_idx, r] != 0:
                    V[vj * p + vk, r] = V22[v22_idx, r]
        
        # W22 row index is k*2 + i for k,i in {0,1}
        for wk in range(2):
            for wi in range(2):
                w22_idx = wk * 2 + wi
                if W22[w22_idx, r] != 0:
                    W[wk * n + wi, r] = W22[w22_idx, r]
    
    # Fill schoolbook part (columns strassen_rank to total_rank-1)
    col = strassen_rank
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this triple is covered by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify the full decomposition
    if not verify_decomp(U, V, W, n, m, p):
        # Fall back to pure schoolbook
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
