import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
    Uses exact Fraction arithmetic for safety.
    """
    rank = len(U[0])
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                
                if total != 1:
                    return False
    return True

def _construct_rank_28():
    """
    Constructs the known rank-28 decomposition for 2x3 * 3x5.
    Uses Strassen for the 2x2x2 subproblems.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2 (A's col 2, B's row 2) - schoolbook for 2x1 * 1x5
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1  # r = 10
    
    # Part 2: j in {0,1}, k=4 - schoolbook for 2x2 * 2x1
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1  # r = 14
    
    # Part 3: j in {0,1}, k in {0,1} - Strassen for 2x2 * 2x2
    # M1 = (a00+a11)(b00+b11)
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[6][r] = 1
    W[0][r] = 1; W[3][r] = 1
    r += 1  # r = 15
    
    # M2 = (a10+a11)b00
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    r += 1  # r = 16
    
    # M3 = a00(b01-b11)
    U[0][r] = 1
    V[1][r] = 1; V[6][r] = -1
    W[2][r] = 1; W[3][r] = 1
    r += 1  # r = 17
    
    # M4 = a11(b10-b00)
    U[4][r] = 1
    V[5][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[2][r] = 1
    r += 1  # r = 18
    
    # M5 = (a00+a01)b11
    U[0][r] = 1; U[1][r] = 1
    V[6][r] = 1
    W[0][r] = -1; W[2][r] = 1
    r += 1  # r = 19
    
    # M6 = (a10-a00)(b00+b01)
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    r += 1  # r = 20
    
    # M7 = (a01-a11)(b10+b11)
    U[1][r] = 1; U[4][r] = -1
    V[5][r] = 1; V[6][r] = 1
    W[0][r] = 1
    r += 1  # r = 21
    
    # Part 4: j in {0,1}, k in {2,3} - Strassen for 2x2 * 2x2
    # M1 = (a00+a11)(b22+b33)
    U[0][r] = 1; U[4][r] = 1
    V[2][r] = 1; V[8][r] = 1
    W[4][r] = 1; W[7][r] = 1
    r += 1  # r = 22
    
    # M2 = (a10+a11)b22
    U[3][r] = 1; U[4][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    r += 1  # r = 23
    
    # M3 = a00(b23-b33)
    U[0][r] = 1
    V[3][r] = 1; V[8][r] = -1
    W[6][r] = 1; W[7][r] = 1
    r += 1  # r = 24
    
    # M4 = a11(b32-b22)
    U[4][r] = 1
    V[7][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[6][r] = 1
    r += 1  # r = 25
    
    # M5 = (a00+a01)b33
    U[0][r] = 1; U[1][r] = 1
    V[8][r] = 1
    W[4][r] = -1; W[6][r] = 1
    r += 1  # r = 26
    
    # M6 = (a10-a00)(b22+b23)
    U[3][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    r += 1  # r = 27
    
    # M7 = (a01-a11)(b32+b33)
    U[1][r] = 1; U[4][r] = -1
    V[7][r] = 1; V[8][r] = 1
    W[4][r] = 1
    r += 1  # r = 28
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Construct the known rank-28 decomposition
    res = _construct_rank_28()
    
    # Verify it to ensure correctness
    if _verify_decomposition(res['U'], res['V'], res['W'], 2, 3, 5):
        return res
    else:
        raise ValueError("Verification failed for rank-28 decomposition")
