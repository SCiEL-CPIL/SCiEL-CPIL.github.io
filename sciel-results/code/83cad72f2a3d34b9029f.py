"""
Matrix multiplication 5x5x5 tensor decomposition.
Uses 2+3 block decomposition with Strassen on 2x2 corner blocks.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    U, V, W are 4x7 matrices (row-major indexing).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3: M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # Term 5: M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: M6 = (c-d)(f+g)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (b-d)(e+f)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _verify_fast(n, m, p, U, V, W):
    """
    Verify exact decomposition using NumPy.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _construct_5x5x5():
    """
    Construct a 5x5x5 matrix multiplication decomposition.
    
    Strategy: Use 2+3 block decomposition:
    - Split indices into {0,1} (size 2) and {2,3,4} (size 3)
    - Use Strassen (rank 7) for the 2x2x2 corner block
    - Use schoolbook (rank 27) for the 3x3x3 corner block
    - Use rank-1 terms for all cross-block interactions
    
    Block structure for (i,j,k) where each index is in {0,1} or {2,3,4}:
    - (2,2,2): Strassen, rank 7
    - (3,3,3): schoolbook, rank 27
    - Cross terms: 125 - 8 - 27 = 90 rank-1 terms
    
    Total rank = 7 + 27 + 90 = 124
    
    But we can do better by also using Strassen on overlapping 2x2 blocks!
    Let's try a different approach: use two non-overlapping 2x2 Strassen blocks
    and schoolbook for everything else.
    
    Actually, let me try: place Strassen on indices {0,1} for all three dimensions,
    and also on {3,4} for all three dimensions, with schoolbook for the rest.
    
    That gives: 7 + 7 + (125 - 8 - 8) = 7 + 7 + 109 = 123
    """
    n, m, p = 5, 5, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # Get Strassen decomposition for 2x2x2
    U_s, V_s, W_s = _strassen_2x2()
    
    # Initialize full matrices with rank 123
    rank = 123
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Place Strassen block 1 (indices 0,1) in columns 0-6
    for c in range(7):
        for r in range(4):
            U[r][c] = U_s[r][c]
            V[r][c] = V_s[r][c]
            W[r][c] = W_s[r][c]
    
    # Place Strassen block 2 (indices 3,4) in columns 7-13
    u_map = [18, 19, 23, 24]  # i*m+j for (3,3),(3,4),(4,3),(4,4)
    v_map = [18, 19, 23, 24]  # j*p+k for (3,3),(3,4),(4,3),(4,4)
    w_map = [18, 19, 23, 24]  # k*n+i for (3,3),(3,4),(4,3),(4,4)
    
    for c in range(7):
        for r in range(4):
            U[u_map[r]][7 + c] = U_s[r][c]
            V[v_map[r]][7 + c] = V_s[r][c]
            W[w_map[r]][7 + c] = W_s[r][c]
    
    # Add schoolbook terms for all other triples
    col = 14
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Handled by Strassen block 1
                if i >= 3 and j >= 3 and k >= 3:
                    continue  # Handled by Strassen block 2
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}, rank


def _verify_full(n, m, p, U, V, W):
    """
    Full verification of the tensor decomposition.
    Checks that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k).
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if err > 1e-9:
                    return False, max_err
    return True, max_err


def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 110  # Safety margin before 120s limit
    
    # Construct the 123-rank decomposition (two Strassen blocks)
    result, rank = _construct_5x5x5()
    
    # Verify it's correct
    ok, max_err = _verify_full(n, m, p, result["U"], result["V"], result["W"])
    if ok:
        return result
    
    # Fallback: Pure schoolbook (rank 125) if verification fails
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
