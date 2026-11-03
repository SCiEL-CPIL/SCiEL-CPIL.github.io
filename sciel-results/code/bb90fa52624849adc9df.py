"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_factors(n, m, p):
    """Return (U, V, W) for schoolbook [n,m,p] with rank n*m*p."""
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
    return U, V, W, rank

def _strassen_222_factors():
    """
    Return U (4x7), V (4x7), W (4x7) for 2x2*2x2=2x2 using Strassen (rank 7).
    Corrected W coefficients to ensure validity.
    """
    # Verify Strassen numerically
    def strassen_2x2(A, B):
        a, b = A[0][0], A[0][1]
        c, d = A[1][0], A[1][1]
        e, f = B[0][0], B[0][1]
        g, h = B[1][0], B[1][1]
        M1 = (a + d) * (e + h)
        M2 = (c + d) * e
        M3 = a * (f - h)
        M4 = d * (g - e)
        M5 = (a + b) * h
        M6 = (c - b) * (e + f)
        M7 = (a + d) * (g + h)
        C00 = M1 - M4 + M6 - M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M3 + M5 - M2
        return [[C00, C01], [C10, C11]]
    
    # Verify with test case
    A_test = [[1, 2], [3, 4]]
    B_test = [[5, 6], [7, 8]]
    C_direct = [[1*5+3*7, 1*6+3*8], [2*5+4*7, 2*6+4*8]]
    C_strassen = strassen_2x2(A_test, B_test)
    
    if C_direct != C_strassen:
        # Fall back to schoolbook for 2x2
        U, V, W, rank = _schoolbook_factors(2, 2, 2)
        return U, V, W, rank
    
    # Build factor matrices for Strassen
    U = [[0]*7 for _ in range(4)]  # 4 A-entries
    V = [[0]*7 for _ in range(4)]  # 4 B-entries
    W = [[0]*7 for _ in range(4)]  # 4 C-entries
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C[0,0] +1, C[1,1] +1
    W[0][0] = 1
    W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C[1,0] +1, C[1,1] -1
    W[1][1] = 1  # C[1,0] is index 1*2+0? No.
    # W index: k*n + i.
    # C[1,0]: k=0, i=1 -> 0*2+1 = 1.
    # C[1,1]: k=1, i=1 -> 1*2+1 = 3.
    W[1][1] = 1
    W[3][1] = -1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C[0,1] +1, C[1,1] -1
    # C[0,1]: k=1, i=0 -> 1*2+0 = 2.
    # C[1,1]: k=1, i=1 -> 1*2+1 = 3.
    W[2][2] = 1
    W[3][2] = -1
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C[0,0] -1, C[1,0] +1
    # C[0,0]: k=0, i=0 -> 0.
    # C[1,0]: k=0, i=1 -> 1.
    W[0][3] = -1
    W[1][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C[0,1] +1, C[1,1] +1
    # C[0,1]: 2. C[1,1]: 3.
    W[2][4] = 1
    W[3][4] = 1
    
    # M6 = (c-b)(e+f)
    U[2][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C[0,0] +1
    # C[0,0]: 0.
    W[0][5] = 1
    
    # M7 = (a+d)(g+h)
    U[0][6] = 1; U[3][6] = 1
    V[2][6] = 1; V[3][6] = 1
    # C[0,0] -1, C[1,1] -1
    # C[0,0]: 0. C[1,1]: 3.
    W[0][6] = -1
    W[3][6] = -1
    
    return U, V, W, 7

def _strassen_221_factors():
    n, m, p = 2, 2, 1
    rank = n * m * p  # 4
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
    return U, V, W, rank

def _strassen_223_split_factors():
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    U_t, V_t, W_t, r_t = _strassen_221_factors()
    
    total_rank = r_s + r_t  # 7 + 4 = 11
    
    U = [[0] * total_rank for _ in range(4)]
    V = [[0] * total_rank for _ in range(6)]
    W = [[0] * total_rank for _ in range(6)]
    
    for i in range(4):
        for r in range(r_s):
            U[i][r] = U_s[i][r]
    for j in range(2):
        for k in range(2):
            for r in range(r_s):
                V[j * 3 + k][r] = V_s[j * 2 + k][r]
    for k in range(2):
        for i in range(2):
            for r in range(r_s):
                W[k * 2 + i][r] = W_s[k * 2 + i][r]
    
    for i in range(4):
        for r in range(r_t):
            U[i][r_s + r] = U_t[i][r]
    for j in range(2):
        for k in range(1):
            for r in range(r_t):
                V[j * 3 + 2][r_s + r] = V_t[j * 1 + k][r]
    for k in range(1):
        for i in range(2):
            for r in range(r_t):
                W[2 * 2 + i][r_s + r] = W_t[k * 2 + i][r]
    
    return U, V, W, total_rank

def _block_decomposition():
    n, m, p = 3, 5, 5
    
    total_rank = 18 + 28 + 10 + 15
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_subproblem(sub_n, sub_m, sub_p, row_offset_A, col_offset_B, mid_offset, 
                       sub_U, sub_V, sub_W, sub_rank):
        nonlocal col
        for i in range(sub_n):
            for j in range(sub_m):
                for r in range(sub_rank):
                    U[(row_offset_A + i) * m + (mid_offset + j)][col + r] = sub_U[i * sub_m + j][r]
        for j in range(sub_m):
            for k in range(sub_p):
                for r in range(sub_rank):
                    V[(mid_offset + j) * p + (col_offset_B + k)][col + r] = sub_V[j * sub_p + k][r]
        for k in range(sub_p):
            for i in range(sub_n):
                for r in range(sub_rank):
                    W[(col_offset_B + k) * n + (row_offset_A + i)][col + r] = sub_W[k * sub_n + i][r]
        col += sub_rank
    
    # 1: [2,5,2]
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    add_subproblem(2, 2, 2, 0, 0, 0, U_s, V_s, W_s, r_s)
    
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    add_subproblem(2, 2, 2, 0, 0, 2, U_s, V_s, W_s, r_s)
    
    U_s, V_s, W_s, r_s = _schoolbook_factors(2, 1, 2)
    add_subproblem(2, 1, 2, 0, 0, 4, U_s, V_s, W_s, r_s)
    
    # 2: [2,5,3]
    U_s, V_s, W_s, r_s = _strassen_223_split_factors()
    add_subproblem(2, 2, 3, 0, 2, 0, U_s, V_s, W_s, r_s)
    
    U_s, V_s, W_s, r_s = _strassen_223_split_factors()
    add_subproblem(2, 2, 3, 0, 2, 2, U_s, V_s, W_s, r_s)
    
    U_s, V_s, W_s, r_s = _schoolbook_factors(2, 1, 3)
    add_subproblem(2, 1, 3, 0, 2, 4, U_s, V_s, W_s, r_s)
    
    # 3: [1,5,2]
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 2)
    add_subproblem(1, 2, 2, 2, 0, 0, U_s, V_s, W_s, r_s)
    
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 2)
    add_subproblem(1, 2, 2, 2, 0, 2, U_s, V_s, W_s, r_s)
    
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 1, 2)
    add_subproblem(1, 1, 2, 2, 0, 4, U_s, V_s, W_s, r_s)
    
    # 4: [1,5,3]
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 3)
    add_subproblem(1, 2, 3, 2, 2, 0, U_s, V_s, W_s, r_s)
    
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 3)
    add_subproblem(1, 2, 3, 2, 2, 2, U_s, V_s, W_s, r_s)
    
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 1, 3)
    add_subproblem(1, 1, 3, 2, 2, 4, U_s, V_s, W_s, r_s)
    
    return U, V, W, col

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W, rank = _block_decomposition()
        
        if time.time() < deadline:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            np.random.seed(42)
            A_test = np.random.randint(-3, 4, size=(n, m))
            B_test = np.random.randint(-3, 4, size=(m, p))
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        for r in range(rank):
                            s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    C_computed[i, k] = s
            
            if np.allclose(C_computed, C_expected, atol=1e-9):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    U, V, W, rank = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
