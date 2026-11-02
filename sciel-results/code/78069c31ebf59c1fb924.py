"""Matrix multiplication 2x4 * 4x7 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication (Rank 50)."""
    deadline = time.time() + 125  # 125s deadline, 5s safety margin before 130s hard kill
    
    n, m, p = 2, 4, 7
    
    # We construct a rank 50 solution based on block decomposition.
    # Split m=4 into two blocks of size 2: j in {0,1} and j in {2,3}.
    # For each block, split p=7 into three blocks of size 2 (k in {0,1}, {2,3}, {4,5}) 
    # and one block of size 1 (k=6).
    # For 2x2 x 2x2 subproblems, use Strassen (rank 7).
    # For 2x2 x 2x1 subproblems, use schoolbook (rank 4).
    # Total rank = 2 * (3 * 7 + 1 * 4) = 2 * (21 + 4) = 50.
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # A indices:
        a_idx = i0 * m + j0          # A[i0, j0]
        b_idx = i0 * m + (j0 + 1)    # A[i0, j0+1]
        c_idx = (i0 + 1) * m + j0    # A[i0+1, j0]
        d_idx = (i0 + 1) * m + (j0 + 1) # A[i0+1, j0+1]
        
        # B indices:
        e_idx = j0 * p + k0          # B[j0, k0]
        f_idx = j0 * p + (k0 + 1)    # B[j0, k0+1]
        g_idx = (j0 + 1) * p + k0    # B[j0+1, k0]
        h_idx = (j0 + 1) * p + (k0 + 1) # B[j0+1, k0+1]
        
        # W indices:
        c00_idx = k0 * n + i0        # C[i0, k0]
        c01_idx = (k0 + 1) * n + i0  # C[i0, k0+1]
        c10_idx = k0 * n + (i0 + 1)  # C[i0+1, k0]
        c11_idx = (k0 + 1) * n + (i0 + 1) # C[i0+1, k0+1]
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        # A indices: A[i0:i0+2, j0:j0+2]
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices: B[j0:j0+2, k0:k0+1] (only 1 column)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        # W indices: W[k0:k0+1, i0:i0+2]
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P2 = b*g
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P3 = c*e
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        # P4 = d*g
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    # Process the two j-blocks
    # j_start = 0 covers j=0,1
    # j_start = 2 covers j=2,3
    for j_start in [0, 2]:
        # k_start = 0, 2, 4 for Strassen (2x2 blocks)
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        # k_start = 6 for Schoolbook (2x1 block)
        add_schoolbook_221_block(0, j_start, 6)

    # Verify correctness
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook if something went wrong (should not happen)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
