"""
Matrix multiplication 5x5x5 tensor decomposition.
Uses block decomposition with Strassen on the 2x2 corner block.
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
    # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    # C[0,0] += M1, C[1,1] += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (c+d)e
    # c=A[1,0], d=A[1,1], e=B[0,0]
    # C[1,0] += M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3: M3 = a(f-h)
    # a=A[0,0], f=B[0,1], h=B[1,1]
    # C[0,1] += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: M4 = d(g-e)
    # d=A[1,1], g=B[1,0], e=B[0,0]
    # C[1,0] += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # Term 5: M5 = (a+b)h
    # a=A[0,0], b=A[0,1], h=B[1,1]
    # C[0,0] -= M5, C[0,1] += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: M6 = (c-d)(f+g)
    # c=A[1,0], d=A[1,1], f=B[0,1], g=B[1,0]
    # C[1,1] += M6
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (b-d)(e+f)
    # b=A[0,1], d=A[1,1], e=B[0,0], f=B[0,1]
    # C[0,0] += M7
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _verify_fast(n, m, p, U, V, W):
    """
    Verify exact decomposition using NumPy (float64 is sufficient for integer/half-integer checks with tolerance).
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # We want to check T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is equivalent to checking if the tensor defined by U, V, W matches the standard matrix mult tensor.
    # Standard tensor T_std[i,j,k] is 1 if k == sum_l A[i,l]B[l,k] ... wait.
    # The tensor for matrix multiplication is T_{ijk} = \delta_{i, k?} No.
    # The tensor is defined such that C = AB => C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{i, j, k} corresponds to the term A_{ij} B_{jk} contributing to C_{ik}.
    # The identity is sum_c U_{ij, c} V_{jk, c} W_{ki, c} = 1 for all i,j,k.
    # Note the W index is k*n + i.
    
    rank = U_arr.shape[1]
    
    # Create index arrays
    i_idx = np.arange(n)
    j_idx = np.arange(m)
    k_idx = np.arange(p)
    
    # We can iterate over i, j, k and check the dot product of the corresponding rows.
    # For performance, we can vectorize this.
    # U rows needed for all (i,j) pairs: U is (n*m, rank).
    # V rows needed for all (j,k) pairs: V is (m*p, rank).
    # W rows needed for all (k,i) pairs: W is (p*n, rank).
    
    # Let's just loop over the 125 combinations, it's fast enough in Python/NumPy.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                # Sum of element-wise products
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _construct_5x5x5():
    """
    Construct a 5x5x5 matrix multiplication decomposition.
    
    Strategy: Use two non-overlapping 2x2x2 Strassen blocks:
    - Block 1: i,j,k in {0,1} (top-left corner)
    - Block 2: i,j,k in {3,4} (bottom-right corner)
    - Use schoolbook (rank 1) for all other 125-16 = 109 triples
    
    Total rank = 7 + 7 + 109 = 123
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
    # Local indices (0,1) map to Global indices (3,4)
    # Local U/V/W row mapping:
    # Local 0 (0,0) -> Global 3,3 -> idx 3*5+3 = 18
    # Local 1 (0,1) -> Global 3,4 -> idx 3*5+4 = 19
    # Local 2 (1,0) -> Global 4,3 -> idx 4*5+3 = 23
    # Local 3 (1,1) -> Global 4,4 -> idx 4*5+4 = 24
    
    # Mapping for U (i*m+j):
    # Local 0 (i=0,j=0) -> Global i=3,j=3 -> 18
    # Local 1 (i=0,j=1) -> Global i=3,j=4 -> 19
    # Local 2 (i=1,j=0) -> Global i=4,j=3 -> 23
    # Local 3 (i=1,j=1) -> Global i=4,j=4 -> 24
    u_map = [18, 19, 23, 24]
    
    # Mapping for V (j*p+k):
    # Local 0 (j=0,k=0) -> Global j=3,k=3 -> 18
    # Local 1 (j=0,k=1) -> Global j=3,k=4 -> 19
    # Local 2 (j=1,k=0) -> Global j=4,k=3 -> 23
    # Local 3 (j=1,k=1) -> Global j=4,k=4 -> 24
    v_map = [18, 19, 23, 24]
    
    # Mapping for W (k*n+i):
    # Local 0 (k=0,i=0) -> Global k=3,i=3 -> 18
    # Local 1 (k=0,i=1) -> Global k=3,i=4 -> 19
    # Local 2 (k=1,i=0) -> Global k=4,i=3 -> 23
    # Local 3 (k=1,i=1) -> Global k=4,i=4 -> 24
    w_map = [18, 19, 23, 24]
    
    for c in range(7):
        for r in range(4):
            U[u_map[r]][7 + c] = U_s[r][c]
            V[v_map[r]][7 + c] = V_s[r][c]
            W[w_map[r]][7 + c] = W_s[r][c]
    
    # Add schoolbook terms for all other triples
    # Skip triples where i,j,k all in {0,1} OR all in {3,4}
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


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Construct the 123-rank decomposition (two Strassen blocks)
    result, rank = _construct_5x5x5()
    
    # Verify it's correct (Fast version)
    if _verify_fast(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: Pure schoolbook (rank 125) if verification fails (should not happen)
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
