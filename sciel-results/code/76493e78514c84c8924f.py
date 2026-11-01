import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 135

    n, m, p = DIMENSIONS

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Improved decomposition: Strassen for 2x2 blocks + Schoolbook for remainder
    # p=13 -> 6 blocks of 2 cols (12 cols) + 1 col of 1
    # Rank: 6 * 7 + 4 = 46
    
    total_rank = 46
    U = [[0] * total_rank for _ in range(n * m)]   # 4 rows
    V = [[0] * total_rank for _ in range(m * p)]   # 26 rows
    W = [[0] * total_rank for _ in range(p * n)]   # 26 rows

    col_idx = 0
    
    # Helper to add Strassen terms for a 2x2 block at columns c0, c1
    def add_strassen_block(c0, c1, start_col):
        # V indices for B block:
        # B[0, c0] -> 0*p + c0 = c0
        # B[0, c1] -> 0*p + c1 = c1
        # B[1, c0] -> 1*p + c0 = p + c0
        # B[1, c1] -> 1*p + c1 = p + c1
        
        # W indices for C block:
        # C[0, c0] -> c0*n + 0 = 2*c0
        # C[0, c1] -> c1*n + 0 = 2*c1
        # C[1, c0] -> c0*n + 1 = 2*c0 + 1
        # C[1, c1] -> c1*n + 1 = 2*c1 + 1
        
        # Strassen terms
        # 1. (a+b)(e+f) -> +C00, +C11
        U[0][start_col] = 1
        U[1][start_col] = 1
        V[c0][start_col] = 1
        V[c1][start_col] = 1
        W[2*c0][start_col] = 1
        W[2*c1+1][start_col] = 1
        
        # 2. (c+d)e -> +C10
        U[2][start_col+1] = 1
        U[3][start_col+1] = 1
        V[c0][start_col+1] = 1
        W[2*c0+1][start_col+1] = 1
        
        # 3. a(h-f) -> +C01
        U[0][start_col+2] = 1
        V[p+c1][start_col+2] = 1
        V[c1][start_col+2] = -1
        W[2*c1][start_col+2] = 1
        
        # 4. d(g-e) -> +C10
        U[3][start_col+3] = 1
        V[p+c0][start_col+3] = 1
        V[c0][start_col+3] = -1
        W[2*c0+1][start_col+3] = 1
        
        # 5. (a+d)(e+h) -> +C01
        U[0][start_col+4] = 1
        U[3][start_col+4] = 1
        V[c0][start_col+4] = 1
        V[p+c1][start_col+4] = 1
        W[2*c1][start_col+4] = 1
        
        # 6. (b+c)(g-e) -> +C11
        U[1][start_col+5] = 1
        U[2][start_col+5] = 1
        V[p+c0][start_col+5] = 1
        V[c0][start_col+5] = -1
        W[2*c1+1][start_col+5] = 1
        
        # 7. (b+d)(f-h) -> -C00
        U[1][start_col+6] = 1
        U[3][start_col+6] = 1
        V[c1][start_col+6] = 1
        V[p+c1][start_col+6] = -1
        W[2*c0][start_col+6] = -1

    # Add 6 Strassen blocks
    for b in range(6):
        c0 = 2 * b
        c1 = 2 * b + 1
        add_strassen_block(c0, c1, col_idx)
        col_idx += 7

    # Add schoolbook for last column (c=12)
    c_last = 12
    # V indices:
    # B[0, 12] -> 12
    # B[1, 12] -> 25
    # W indices:
    # C[0, 12] -> 24
    # C[1, 12] -> 25
    
    # Term 1: a * B[0, 12] -> C[0, 12]
    U[0][col_idx] = 1
    V[12][col_idx] = 1
    W[24][col_idx] = 1
    
    # Term 2: b * B[1, 12] -> C[0, 12]
    U[1][col_idx+1] = 1
    V[25][col_idx+1] = 1
    W[24][col_idx+1] = 1
    
    # Term 3: c * B[0, 12] -> C[1, 12]
    U[2][col_idx+2] = 1
    V[12][col_idx+2] = 1
    W[25][col_idx+2] = 1
    
    # Term 4: d * B[1, 12] -> C[1, 12]
    U[3][col_idx+3] = 1
    V[25][col_idx+3] = 1
    W[25][col_idx+3] = 1
    
    col_idx += 4

    # Verify numerically
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for t in range(total_rank):
                    C_decomp[i, k] += U_np[i*m+j, t] * V_np[j*p+k, t] * W_np[k*n+i, t]
    
    if np.allclose(C_decomp, C_expected, atol=1e-10):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = total_rank

    if time.time() - start_time > deadline - 5:
        return best_result

    return best_result
