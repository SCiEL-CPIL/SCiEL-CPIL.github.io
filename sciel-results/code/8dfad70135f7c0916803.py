"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The tensor rank of 2x2 matrix multiplication is exactly 7 over the reals
    (and over fields of characteristic 0). This is a proven lower bound, so
    no rank-6 or lower solution exists. We return the verified Strassen
    decomposition with integer entries.
    
    Deadline guard: return immediately since the solution is known and static.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = 2, 2, 2
    rank = 7
    
    # Construct Strassen's rank-7 decomposition
    # U: (n*m, rank) = (4, 7)
    # V: (m*p, rank) = (4, 7)  
    # W: (p*n, rank) = (4, 7)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # M3 = A00(B10 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # M4 = A11(B01 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    
    # W: output reconstruction
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    solution = {"U": U, "V": V, "W": W}
    
    # Quick verification (should always pass, but check for safety)
    if time.time() < deadline:
        if not _verify_decomp(U, V, W, n, m, p):
            # Fallback: rank-8 schoolbook decomposition
            rank_fb = 8
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            idx = 0
            for i in range(n):
                for a in range(m):
                    for k in range(p):
                        U_fb[i * m + a][idx] = 1
                        V_fb[a * p + k][idx] = 1
                        W_fb[k * n + i][idx] = 1
                        idx += 1
            solution = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return solution


def _verify_decomp(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the n x m by m x p matrix multiplication tensor.
    
    The tensor equation is:
    For all i in [0,n), a in [0,m), k in [0,p):
      sum_r U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for a in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + a][r] * V[a * p + k][r] * W[k * n + i][r]
                if val != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
