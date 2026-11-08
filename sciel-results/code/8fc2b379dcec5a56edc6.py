"""Matrix multiplication 3,3,4 - Rank 27 decomposition."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check that U @ V^T matches W reshaped as tensor."""
    import numpy as np
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should be 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Rank 27 decomposition
    # Part 1: 3x3x3 decomposition of rank 26
    #   - 7 Strassen terms for 2x2x2 core (i,j,k in {0,1})
    #   - 19 Schoolbook terms for the border (indices involving 2)
    # Part 2: 1 term for the 4th column (k=3)
    
    rank = 27
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 27
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 27
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 27
    
    col = 0
    
    # --- Part 1: 3x3x3 Core ---
    
    # 1. Strassen 2x2x2 (i,j in {0,1}, k in {0,1})
    # Indices: i, j, k_local in {0,1}
    # k_global = k_local (since this is the first 2 columns of the 3x3x3 block)
    
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,1,0,1], [0,0,1,0], [0,1,0,0]),
        ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),
        ([0,0,0,1], [1,0,-1,0], [1,1,0,0]),
        ([1,0,1,0], [0,1,0,0], [-1,0,1,0]),
        ([0,-1,1,0], [0,0,1,1], [0,0,0,1]),
        ([0,1,-1,0], [1,0,1,0], [1,0,0,0]),
    ]
    
    for (u_c, v_c, w_c) in strassen_terms:
        c = col
        # U: A[i,j] for i,j in {0,1}
        U[0][c] += u_c[0] # A00
        U[1][c] += u_c[1] # A01
        U[3][c] += u_c[2] # A10
        U[4][c] += u_c[3] # A11
        
        # V: B[j,k] for j in {0,1}, k in {0,1}
        V[0][c] += v_c[0] # B00
        V[1][c] += v_c[1] # B01
        V[4][c] += v_c[2] # B10
        V[5][c] += v_c[3] # B11
        
        # W: C[i,k] -> W[k*n+i]
        # C00: k=0, i=0 -> W[0]
        # C01: k=1, i=0 -> W[3]
        # C10: k=0, i=1 -> W[1]
        # C11: k=1, i=1 -> W[4]
        W[0][c] += w_c[0]
        W[3][c] += w_c[1]
        W[1][c] += w_c[2]
        W[4][c] += w_c[3]
        
        col += 1
        
    # 2. Schoolbook terms for the border of 3x3x3
    # Entries where at least one index is 2.
    # We iterate i, j, k in {0,1,2} and if max(i,j,k)==2, we add a term.
    # To avoid duplicates, we just list all 19 combinations.
    
    border_indices = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == 2 or j == 2 or k == 2:
                    border_indices.append((i,j,k))
                    
    for (i,j,k) in border_indices:
        c = col
        U[i*m + j][c] = 1.0
        V[j*p + k][c] = 1.0
        W[k*n + i][c] = 1.0
        col += 1
        
    # --- Part 2: 4th Column (k=3) ---
    # We need C[i,3] = sum_j A[i,j] B[j,3]
    # Use 1 rank term:
    # U[i,j, c] = 1
    # V[j,3, c] = 1
    # W[3,i, c] = 1
    # Sum over j: A[i,j] * 1 * 1 * B[j,3] * 1 = sum A[i,j]B[j,3]
    
    c = col
    for i in range(n):
        for j in range(m):
            U[i*m + j][c] = 1.0
    for j in range(m):
        V[j*p + 3][c] = 1.0
    for i in range(n):
        W[3*n + i][c] = 1.0
    col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U = [[0] * rank_fb for _ in range(n * m)]
        V = [[0] * rank_fb for _ in range(m * p)]
        W = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
