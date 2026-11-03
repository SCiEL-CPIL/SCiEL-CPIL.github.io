"""Matrix multiplication 3x5x5 - rank-minimal tensor decomposition."""
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
    A[i,j]: i=0,1 rows; j=0,1 cols
    B[j,k]: j=0,1 rows; k=0,1 cols
    C[i,k] = sum_j A[i,j]*B[j,k]
    U index: i*2+j (A entry)
    V index: j*2+k (B entry)
    W index: k*2+i (C entry)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1
    W[1][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1
    W[3][4] = 1
    
    # M6 = (c-b)(e+f)
    U[2][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # M7 = (a+d)(g+h)
    U[0][6] = 1; U[3][6] = 1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1
    W[3][6] = -1
    
    return U, V, W, 7


def _combine_subproblems(subprobs, n, m, p):
    """
    Combine sub-problems into full [n,m,p] factors.
    subprobs: list of (sub_n, sub_m, sub_p, row_off, col_off, mid_off, U_s, V_s, W_s, r_s)
    """
    total_rank = sum(sp[8] for sp in subprobs)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for (sub_n, sub_m, sub_p, row_off, col_off, mid_off, U_s, V_s, W_s, r_s) in subprobs:
        for i in range(sub_n):
            for j in range(sub_m):
                for r in range(r_s):
                    U[(row_off + i) * m + (mid_off + j)][col + r] = U_s[i * sub_m + j][r]
        for j in range(sub_m):
            for k in range(sub_p):
                for r in range(r_s):
                    V[(mid_off + j) * p + (col_off + k)][col + r] = V_s[j * sub_p + k][r]
        for k in range(sub_p):
            for i in range(sub_n):
                for r in range(r_s):
                    W[(col_off + k) * n + (row_off + i)][col + r] = W_s[k * sub_n + i][r]
        col += r_s
    
    return U, V, W, col


def _block_decomposition_v1():
    """
    Block decomposition of [3,5,5]:
    Split rows: 2+1, split cols: 2+3, split mid: 2+2+1
    Uses Strassen for 2x2x2 blocks.
    """
    n, m, p = 3, 5, 5
    subprobs = []
    
    # [2,2,2] Strassen: rows 0-1, cols 0-1, mid 0-1
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    subprobs.append((2, 2, 2, 0, 0, 0, U_s, V_s, W_s, r_s))
    
    # [2,2,2] Strassen: rows 0-1, cols 0-1, mid 2-3
    U_s, V_s, W_s, r_s = _strassen_222_factors()
    subprobs.append((2, 2, 2, 0, 0, 2, U_s, V_s, W_s, r_s))
    
    # [2,1,2] schoolbook: rows 0-1, cols 0-1, mid 4
    U_s, V_s, W_s, r_s = _schoolbook_factors(2, 1, 2)
    subprobs.append((2, 1, 2, 0, 0, 4, U_s, V_s, W_s, r_s))
    
    # [2,2,3] split: rows 0-1, cols 2-4, mid 0-1
    U_222, V_222, W_222, r_222 = _strassen_222_factors()
    U_221, V_221, W_221, r_221 = _schoolbook_factors(2, 2, 1)
    # Combine [2,2,2]+[2,2,1] -> [2,2,3]
    U_223 = [[0]*(r_222+r_221) for _ in range(4)]
    V_223 = [[0]*(r_222+r_221) for _ in range(6)]
    W_223 = [[0]*(r_222+r_221) for _ in range(6)]
    for i in range(4):
        for r in range(r_222):
            U_223[i][r] = U_222[i][r]
        for r in range(r_221):
            U_223[i][r_222+r] = U_221[i][r]
    for j in range(2):
        for k in range(3):
            for r in range(r_222):
                V_223[j*3+k][r] = V_222[j*2+k if k<2 else 0][r] if k<2 else 0
            for r in range(r_221):
                V_223[j*3+k][r_222+r] = V_221[j*1+k if k<1 else 0][r] if k<1 else 0
    for k in range(3):
        for i in range(2):
            for r in range(r_222):
                W_223[k*2+i][r] = W_222[k*2+i if k<2 else 0][r] if k<2 else 0
            for r in range(r_221):
                W_223[k*2+i][r_222+r] = W_221[k*2+i if k<1 else 0][r] if k<1 else 0
    subprobs.append((2, 2, 3, 0, 2, 0, U_223, V_223, W_223, r_222+r_221))
    
    # [2,2,3] split: rows 0-1, cols 2-4, mid 2-3
    U_223 = [[0]*(r_222+r_221) for _ in range(4)]
    V_223 = [[0]*(r_222+r_221) for _ in range(6)]
    W_223 = [[0]*(r_222+r_221) for _ in range(6)]
    for i in range(4):
        for r in range(r_222):
            U_223[i][r] = U_222[i][r]
        for r in range(r_221):
            U_223[i][r_222+r] = U_221[i][r]
    for j in range(2):
        for k in range(3):
            for r in range(r_222):
                V_223[j*3+k][r] = V_222[j*2+k if k<2 else 0][r] if k<2 else 0
            for r in range(r_221):
                V_223[j*3+k][r_222+r] = V_221[j*1+k if k<1 else 0][r] if k<1 else 0
    for k in range(3):
        for i in range(2):
            for r in range(r_222):
                W_223[k*2+i][r] = W_222[k*2+i if k<2 else 0][r] if k<2 else 0
            for r in range(r_221):
                W_223[k*2+i][r_222+r] = W_221[k*2+i if k<1 else 0][r] if k<1 else 0
    subprobs.append((2, 2, 3, 0, 2, 2, U_223, V_223, W_223, r_222+r_221))
    
    # [2,1,3] schoolbook: rows 0-1, cols 2-4, mid 4
    U_s, V_s, W_s, r_s = _schoolbook_factors(2, 1, 3)
    subprobs.append((2, 1, 3, 0, 2, 4, U_s, V_s, W_s, r_s))
    
    # [1,2,2] schoolbook: row 2, cols 0-1, mid 0-1
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 2)
    subprobs.append((1, 2, 2, 2, 0, 0, U_s, V_s, W_s, r_s))
    
    # [1,2,2] schoolbook: row 2, cols 0-1, mid 2-3
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 2)
    subprobs.append((1, 2, 2, 2, 0, 2, U_s, V_s, W_s, r_s))
    
    # [1,1,2] schoolbook: row 2, cols 0-1, mid 4
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 1, 2)
    subprobs.append((1, 1, 2, 2, 0, 4, U_s, V_s, W_s, r_s))
    
    # [1,2,3] schoolbook: row 2, cols 2-4, mid 0-1
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 3)
    subprobs.append((1, 2, 3, 2, 2, 0, U_s, V_s, W_s, r_s))
    
    # [1,2,3] schoolbook: row 2, cols 2-4, mid 2-3
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 2, 3)
    subprobs.append((1, 2, 3, 2, 2, 2, U_s, V_s, W_s, r_s))
    
    # [1,1,3] schoolbook: row 2, cols 2-4, mid 4
    U_s, V_s, W_s, r_s = _schoolbook_factors(1, 1, 3)
    subprobs.append((1, 1, 3, 2, 2, 4, U_s, V_s, W_s, r_s))
    
    return _combine_subproblems(subprobs, n, m, p)


def _verify(U, V, W, rank, n, m, p, seed=42):
    """Verify decomposition correctness."""
    np.random.seed(seed)
    A_test = np.random.randint(-3, 4, size=(n, m))
    B_test = np.random.randint(-3, 4, size=(m, p))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p), dtype=float)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        for i in range(n):
            u_row = u_col[i*m:(i+1)*m]
            for k in range(p):
                v_col_k = v_col[k::p]
                C_computed[i, k] += w_col[k*n+i] * np.dot(u_row, v_col_k)
    
    return np.allclose(C_computed, C_expected, atol=1e-9)


def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Try block decomposition with Strassen
    try:
        U, V, W, rank = _block_decomposition_v1()
        
        if time.time() < deadline:
            if _verify(U, V, W, rank, n, m, p, seed=42) and _verify(U, V, W, rank, n, m, p, seed=123):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W, rank = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
