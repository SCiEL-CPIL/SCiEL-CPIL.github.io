"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from functools import lru_cache

DIMENSIONS = [4, 5, 5]

_start_time = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _check_time():
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError()

def _schoolbook(m, n, p):
    """Return (U, V, W) for m×n * n×p matrix multiplication using schoolbook."""
    rank = m * n * p
    U = [[0] * rank for _ in range(m * n)]
    V = [[0] * rank for _ in range(n * p)]
    W = [[0] * rank for _ in range(p * m)]
    col = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                U[i * n + j][col] = 1
                V[j * p + k][col] = 1
                W[k * m + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2_list():
    """Return (U, V, W) for 2x2x2 Strassen decomposition (rank 7) as lists.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = -1
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = 1
    return U, V, W

def _decompose_2x2x3():
    """Return (U, V, W) for 2x2 * 2x3 matrix multiplication.
    U: shape (4, rank), V: shape (6, rank), W: shape (6, rank)
    W[k*2+i, r]
    """
    # Split p=3 as p1=1, p2=2
    # Sub-problems: (2,2,1) and (2,2,2)
    # (2,2,1): rank 4, schoolbook
    # (2,2,2): rank 7, Strassen
    # Total rank = 4 + 7 = 11
    # But we can do better: try other splits
    # Split m=2 as 1+1, n=2 as 1+1, p=3 as 1+2:
    # (1,1,1)=1, (1,1,2)=2, (1,1,1)=1, (1,1,2)=2, (1,1,1)=1, (1,1,2)=2, (1,1,1)=1, (1,1,2)=2 -> total=12
    # Split m=2 as 1+1, n=2 as 1+1, p=3 as 2+1:
    # (1,1,2)=2, (1,1,1)=1, (1,1,2)=2, (1,1,1)=1, (1,1,2)=2, (1,1,1)=1, (1,1,2)=2, (1,1,1)=1 -> total=12
    # So best is 11 with p split 1+2.
    
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(6)]
    
    # Part 1: (2,2,1) schoolbook, offset_p=0
    # U[i*2+j, col], V[j*1+k, col], W[k*2+i, col] for k in range(1)
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(1):
                U[i*2+j][col] = 1
                V[j*1+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    # Part 2: (2,2,2) Strassen, offset_p=1
    U_s, V_s, W_s = _strassen_2x2x2_list()
    for r in range(7):
        for ii in range(2):
            for jj in range(2):
                U[ii*2+jj][col + r] = U_s[ii*2+jj][r]
        for jj in range(2):
            for kk in range(2):
                V[jj*2+kk][col + r] = V_s[jj*2+kk][r]
        for kk in range(2):
            for ii in range(2):
                W[(1+kk)*2+ii][col + r] = W_s[kk*2+ii][r]
    
    return U, V, W

def _decompose_2x3x2():
    """Return (U, V, W) for 2x3 * 3x2 matrix multiplication.
    U: shape (6, rank), V: shape (6, rank), W: shape (4, rank)
    W[k*2+i, r]
    """
    # Split n=3 as n1=1, n2=2
    # Sub-problems: (2,1,2) and (2,2,2)
    # (2,1,2): rank 4, schoolbook
    # (2,2,2): rank 7, Strassen
    # Total rank = 4 + 7 = 11
    
    U = [[0]*11 for _ in range(6)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(4)]
    
    # Part 1: (2,1,2) schoolbook, offset_n=0
    col = 0
    for i in range(2):
        for j in range(1):
            for k in range(2):
                U[i*3+j][col] = 1
                V[j*2+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    # Part 2: (2,2,2) Strassen, offset_n=1
    U_s, V_s, W_s = _strassen_2x2x2_list()
    for r in range(7):
        for ii in range(2):
            for jj in range(2):
                U[ii*3+(1+jj)][col + r] = U_s[ii*2+jj][r]
        for jj in range(2):
            for kk in range(2):
                V[(1+jj)*2+kk][col + r] = V_s[jj*2+kk][r]
        for kk in range(2):
            for ii in range(2):
                W[kk*2+ii][col + r] = W_s[kk*2+ii][r]
    
    return U, V, W

def _decompose_2x3x3():
    """Return (U, V, W) for 2x3 * 3x3 matrix multiplication.
    U: shape (6, rank), V: shape (9, rank), W: shape (6, rank)
    W[k*2+i, r]
    """
    # Split n=3 as n1=1, n2=2 and p=3 as p1=1, p2=2
    # Sub-problems: (2,1,1), (2,1,2), (2,2,1), (2,2,2)
    # Ranks: 2, 4, 4, 7 -> total = 17
    # Alternative: split n=3 as 2+1, p=3 as 1+2
    # (2,2,1)=4, (2,2,2)=7, (2,1,1)=2, (2,1,2)=4 -> total = 17
    # Alternative: split m=2 as 1+1, n=3 as 1+2, p=3 as 1+2
    # 8 sub-problems of size (1,1,1),(1,1,2),(1,2,1),(1,2,2) etc
    # (1,1,1)=1, (1,1,2)=2, (1,2,1)=2, (1,2,2)=4 -> x2 = 12
    # So best is 12 with m split.
    
    # Use m split: m1=1, m2=1, n1=1, n2=2, p1=1, p2=2
    U = [[0]*12 for _ in range(6)]
    V = [[0]*12 for _ in range(9)]
    W = [[0]*12 for _ in range(6)]
    
    col = 0
    # A11*B11: (1,1,1)
    U[0*3+0][col] = 1; V[0*3+0][col] = 1; W[0*2+0][col] = 1; col += 1
    # A12*B21: (1,2,1)
    for jj in range(2):
        U[0*3+(1+jj)][col] = 1; V[(1+jj)*3+0][col] = 1; W[0*2+0][col] = 1; col += 1
    # A11*B12: (1,1,2)
    for kk in range(2):
        U[0*3+0][col] = 1; V[0*3+(1+kk)][col] = 1; W[(1+kk)*2+0][col] = 1; col += 1
    # A12*B22: (1,2,2)
    for jj in range(2):
        for kk in range(2):
            U[0*3+(1+jj)][col] = 1; V[(1+jj)*3+(1+kk)][col] = 1; W[(1+kk)*2+0][col] = 1; col += 1
    # A21*B11: (1,1,1)
    U[1*3+0][col] = 1; V[0*3+0][col] = 1; W[0*2+1][col] = 1; col += 1
    # A22*B21: (1,2,1)
    for jj in range(2):
        U[1*3+(1+jj)][col] = 1; V[(1+jj)*3+0][col] = 1; W[0*2+1][col] = 1; col += 1
    # A21*B12: (1,1,2)
    for kk in range(2):
        U[1*3+0][col] = 1; V[0*3+(1+kk)][col] = 1; W[(1+kk)*2+1][col] = 1; col += 1
    # A22*B22: (1,2,2)
    for jj in range(2):
        for kk in range(2):
            U[1*3+(1+jj)][col] = 1; V[(1+jj)*3+(1+kk)][col] = 1; W[(1+kk)*2+1][col] = 1; col += 1
    
    return U, V, W

def _decompose_2x2x2_direct():
    """Direct 2x2x2 Strassen."""
    return _strassen_2x2x2_list()

def _decompose_2x1x2():
    """2x1 * 1x2 matrix multiplication. Rank 4, schoolbook.
    U: shape (2, 4), V: shape (2, 4), W: shape (4, 4)
    W[k*2+i, r]
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(1):
            for k in range(2):
                U[i*1+j][col] = 1
                V[j*2+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def _decompose_1x2x2():
    """1x2 * 2x2 matrix multiplication. Rank 4, schoolbook.
    U: shape (2, 4), V: shape (4, 4), W: shape (2, 4)
    W[k*1+i, r] = W[k+i, r] since m=1
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(2)]
    col = 0
    for i in range(1):
        for j in range(2):
            for k in range(2):
                U[i*2+j][col] = 1
                V[j*2+k][col] = 1
                W[k*1+i][col] = 1
                col += 1
    return U, V, W

def _decompose_2x1x1():
    """2x1 * 1x1 matrix multiplication. Rank 2.
    U: shape (2, 2), V: shape (1, 2), W: shape (2, 2)
    W[k*2+i, r]
    """
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(1)]
    W = [[0]*2 for _ in range(2)]
    col = 0
    for i in range(2):
        for j in range(1):
            for k in range(1):
                U[i*1+j][col] = 1
                V[j*1+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def _decompose_1x1x2():
    """1x1 * 1x2 matrix multiplication. Rank 2.
    U: shape (1, 2), V: shape (2, 2), W: shape (2, 2)
    W[k*1+i, r] = W[k, r]
    """
    U = [[0]*2]
    V = [[0]*2 for _ in range(2)]
    W = [[0]*2 for _ in range(2)]
    col = 0
    for i in range(1):
        for j in range(1):
            for k in range(2):
                U[i*1+j][col] = 1
                V[j*2+k][col] = 1
                W[k*1+i][col] = 1
                col += 1
    return U, V, W

def _decompose_1x2x1():
    """1x2 * 2x1 matrix multiplication. Rank 2.
    U: shape (2, 2), V: shape (2, 2), W: shape (1, 2)
    W[k*1+i, r] = W[k, r]
    """
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(2)]
    W = [[0]*2]
    col = 0
    for i in range(1):
        for j in range(2):
            for k in range(1):
                U[i*2+j][col] = 1
                V[j*1+k][col] = 1
                W[k*1+i][col] = 1
                col += 1
    return U, V, W

def _decompose_1x1x1():
    """1x1 * 1x1. Rank 1."""
    return [[1]], [[1]], [[1]]

def _decompose_1x1xN(n):
    """1x1 * 1xN. Rank N."""
    U = [[0]*n]
    V = [[0]*n for _ in range(n)]
    W = [[0]*n for _ in range(n)]
    for k in range(n):
        U[0][k] = 1
        V[k][k] = 1
        W[k][k] = 1
    return U, V, W

def _decompose_Nx1x1(n):
    """Nx1 * 1x1. Rank N."""
    U = [[0]*n for _ in range(n)]
    V = [[0]*n]
    W = [[0]*n for _ in range(n)]
    for i in range(n):
        U[i][i] = 1
        V[0][i] = 1
        W[i][i] = 1
    return U, V, W

def _decompose_1xNx1(n):
    """1xN * Nx1. Rank N."""
    U = [[0]*n for _ in range(n)]
    V = [[0]*n for _ in range(n)]
    W = [[0]*n]
    for j in range(n):
        U[j][j] = 1
        V[j][j] = 1
        W[0][j] = 1
    return U, V, W

@lru_cache(maxsize=None)
def _decompose(m, n, p):
    """
    Return (U, V, W) for m×n * n×p matrix multiplication.
    U: shape (m*n, rank)
    V: shape (n*p, rank)
    W: shape (p*m, rank), W[k*m+i, r] convention
    """
    _check_time()
    
    if m == 0 or n == 0 or p == 0:
        return (), (), ()
    
    # Base cases with direct implementations
    if m == 1 and n == 1 and p == 1:
        return _decompose_1x1x1()
    if m == 2 and n == 2 and p == 2:
        return _strassen_2x2x2_list()
    if m == 2 and n == 2 and p == 3:
        return _decompose_2x2x3()
    if m == 2 and n == 3 and p == 2:
        return _decompose_2x3x2()
    if m == 2 and n == 3 and p == 3:
        return _decompose_2x3x3()
    if m == 2 and n == 1 and p == 2:
        return _decompose_2x1x2()
    if m == 1 and n == 2 and p == 2:
        return _decompose_1x2x2()
    if m == 2 and n == 1 and p == 1:
        return _decompose_2x1x1()
    if m == 1 and n == 1 and p == 2:
        return _decompose_1x1x2()
    if m == 1 and n == 2 and p == 1:
        return _decompose_1x2x1()
    
    # Degenerate cases
    if m == 1 and n == 1:
        return _decompose_1x1xN(p)
    if m == 1 and p == 1:
        return _decompose_1xNx1(n)
    if n == 1 and p == 1:
        return _decompose_Nx1x1(m)
    if m == 1:
        rank = n * p
        U = [[0] * rank for _ in range(n)]
        V = [[0] * rank for _ in range(n * p)]
        W = [[0] * rank for _ in range(p)]
        col = 0
        for j in range(n):
            for k in range(p):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k][col] = 1
                col += 1
        return (U, V, W)
    if p == 1:
        rank = m * n
        U = [[0] * rank for _ in range(m * n)]
        V = [[0] * rank for _ in range(n)]
        W = [[0] * rank for _ in range(m)]
        col = 0
        for i in range(m):
            for j in range(n):
                U[i * n + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return (U, V, W)
    if n == 1:
        rank = m * p
        U = [[0] * rank for _ in range(m)]
        V = [[0] * rank for _ in range(p)]
        W = [[0] * rank for _ in range(p * m)]
        col = 0
        for i in range(m):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k * m + i][col] = 1
                col += 1
        return (U, V, W)
    
    # Try all possible splits and pick the best
    best = None
    best_rank = m * n * p  # schoolbook
    
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                _check_time()
                try:
                    U1, V1, W1 = _decompose(m1, n1, p1)
                    U2, V2, W2 = _decompose(m1, n1, p2)
                    U3, V3, W3 = _decompose(m1, n2, p1)
                    U4, V4, W4 = _decompose(m1, n2, p2)
                    U5, V5, W5 = _decompose(m2, n1, p1)
                    U6, V6, W6 = _decompose(m2, n1, p2)
                    U7, V7, W7 = _decompose(m2, n2, p1)
                    U8, V8, W8 = _decompose(m2, n2, p2)
                    
                    total_rank = (len(U1) + len(U2) + len(U3) + len(U4) +
                                 len(U5) + len(U6) + len(U7) + len(U8))
                    if total_rank < best_rank:
                        best_rank = total_rank
                        U_new = []
                        V_new = []
                        W_new = []
                        
                        def add_decomp(U_s, V_s, W_s, m_i, n_i, p_i, offset_m, offset_n, offset_p):
                            for r in range(len(U_s)):
                                u_row = [0] * (m * n)
                                v_row = [0] * (n * p)
                                w_row = [0] * (p * m)
                                for ii in range(m_i):
                                    for jj in range(n_i):
                                        u_row[(offset_m + ii) * n + (offset_n + jj)] = U_s[r][ii * n_i + jj]
                                for jj in range(n_i):
                                    for kk in range(p_i):
                                        v_row[(offset_n + jj) * p + (offset_p + kk)] = V_s[r][jj * p_i + kk]
                                for kk in range(p_i):
                                    for ii in range(m_i):
                                        w_row[(offset_p + kk) * m + (offset_m + ii)] = W_s[r][kk * m_i + ii]
                                U_new.append(u_row)
                                V_new.append(v_row)
                                W_new.append(w_row)
                        
                        add_decomp(U1, V1, W1, m1, n1, p1, 0, 0, 0)
                        add_decomp(U3, V3, W3, m1, n2, p1, 0, n1, 0)
                        add_decomp(U2, V2, W2, m1, n1, p2, 0, 0, p1)
                        add_decomp(U4, V4, W4, m1, n2, p2, 0, n1, p1)
                        add_decomp(U5, V5, W5, m2, n1, p1, m1, 0, 0)
                        add_decomp(U6, V6, W6, m2, n1, p2, m1, 0, p1)
                        add_decomp(U7, V7, W7, m2, n2, p1, m1, n1, 0)
                        add_decomp(U8, V8, W8, m2, n2, p2, m1, n1, p1)
                        
                        best = (U_new, V_new, W_new)
                except TimeoutError:
                    raise
                except Exception:
                    continue
    
    if best is not None:
        return best
    else:
        return _schoolbook(m, n, p)

def construct():
    global _start_time
    _start_time = time.time()
    m, n, p = DIMENSIONS  # m=4, n=5, p=5
    try:
        U, V, W = _decompose(m, n, p)
        return {"U": U, "V": V, "W": W}
    except TimeoutError:
        U, V, W = _schoolbook(m, n, p)
        return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = _schoolbook(m, n, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
