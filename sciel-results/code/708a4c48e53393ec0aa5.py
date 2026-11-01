import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                lhs = 0.0
                for r in range(rank):
                    lhs += U[a][r] * V[b][r] * W[c][r]
                if abs(lhs - expected) > 1e-9:
                    return False
    return True

def _try_rank_11():
    """Return a rank-11 decomposition using Strassen + schoolbook."""
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen for columns 0,1 (the 2x2x2 part)
    # M1 = A00 * B00
    U[0][0] = 1.0
    V[0][0] = 1.0
    W[0][0] = 1.0  # C00
    W[3][0] = 1.0  # C11
    
    # M2 = A10 * (B00 + B01)
    U[2][1] = 1.0
    V[0][1] = 1.0
    V[1][1] = 1.0
    W[2][1] = 1.0  # C10
    W[3][1] = -1.0 # C11
    
    # M3 = (A00 + A01) * B11
    U[0][2] = 1.0
    U[1][2] = 1.0
    V[4][2] = 1.0
    W[1][2] = 1.0  # C01
    W[3][2] = 1.0  # C11
    
    # M4 = A11 * (B10 + B11)
    U[3][3] = 1.0
    V[3][3] = 1.0
    V[4][3] = 1.0
    W[0][3] = 1.0  # C00
    
    # M5 = (A00 + A11) * (B00 + B11)
    U[0][4] = 1.0
    U[3][4] = 1.0
    V[0][4] = 1.0
    V[4][4] = 1.0
    W[0][4] = -1.0 # C00
    W[3][4] = 1.0  # C11
    
    # M6 = (A10 + A11) * (B10 - B00)
    U[2][5] = 1.0
    U[3][5] = 1.0
    V[3][5] = 1.0
    V[0][5] = -1.0
    W[2][5] = 1.0  # C10
    
    # M7 = (A00 + A01) * (B01 - B10)
    U[0][6] = 1.0
    U[1][6] = 1.0
    V[1][6] = 1.0
    V[3][6] = -1.0
    W[0][6] = -1.0 # C00
    W[1][6] = 1.0  # C01
    
    # Schoolbook for column 2 (the 2x2x1 part)
    # C02 = A00*B02 + A01*B12
    U[0][7] = 1.0
    V[2][7] = 1.0
    W[4][7] = 1.0  # C02
    
    U[1][8] = 1.0
    V[5][8] = 1.0
    W[4][8] = 1.0  # C02
    
    # C12 = A10*B02 + A11*B12
    U[2][9] = 1.0
    V[2][9] = 1.0
    W[5][9] = 1.0  # C12
    
    U[3][10] = 1.0
    V[5][10] = 1.0
    W[5][10] = 1.0  # C12
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _try_rank_10_search(deadline):
    """
    Attempt to find a rank-10 decomposition.
    
    Strategy: The 2x2x2 part has rank 7 (Strassen). The third column adds a 2x2x1
    part which has rank 4. Total 11. To get rank 10, we need to share one term.
    
    We'll parameterize the decomposition and solve the resulting equations.
    
    Let's use the structure: 7 Strassen terms for columns 0,1 plus 3 terms for column 2
    (instead of 4). The 3 terms must span the 4-dimensional space of the 2x2x1 multiplication.
    This is impossible in general, but let's check if the specific tensor structure allows it.
    
    Actually, the 2x2x1 tensor (A 2x2, B 2x1, C 2x1) has rank 4. So we can't do it with 3.
    
    Alternative: maybe a completely different rank-10 structure exists.
    
    Let's set up the full rank-10 problem as a system of polynomial equations and try to solve.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Number of unknowns:
    # U: 4*10 = 40
    # V: 6*10 = 60
    # W: 6*10 = 60
    # Total: 160 unknowns
    
    # Constraints: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # 4*6*6 = 144 constraints
    
    # This is a system of 144 cubic equations in 160 variables.
    # We can try to solve it using a homotopy method or by fixing some variables.
    
    # Let's try a structured approach: fix the U matrix to a known good form
    # and solve for V and W.
    
    # Use the Strassen U for the first 7 columns, and add 3 more columns.
    # The first 7 columns handle the 2x2x2 part (columns 0,1 of B).
    # The last 3 columns must handle column 2 of B.
    
    # For the last 3 columns, we need:
    # C[0,2] = sum_{r=7}^{9} U[0,r]*V[0,r]*W[4,r] + ... (but U[0,r] for r>=7 may be nonzero)
    # This gets complicated.
    
    # Let me try a different approach: use numerical optimization.
    
    start_time = time.time()
    
    # Initialize with the rank-11 solution, zero out one column, and try to optimize.
    result = _try_rank_11()
    if result is None:
        return None
    
    U = [row[:] for row in result['U']]
    V = [row[:] for row in result['V']]
    W = [row[:] for row in result['W']]
    
    # Zero out column 10 (the last schoolbook term)
    for i in range(len(U)):
        U[i][10] = 0.0
    for i in range(len(V)):
        V[i][10] = 0.0
    for i in range(len(W)):
        W[i][10] = 0.0
    
    # Now we have a rank-9 "base" that handles columns 0,1 but not column 2.
    # We need to fill in column 10 (and possibly adjust others) to handle column 2.
    
    # Actually, let's just do a full numerical search for rank 10.
    # Use scipy if available, otherwise a simple gradient descent.
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    def _flatten(U, V, W, n, m, p, rank):
        vec = []
        for row in U:
            vec.extend(row)
        for row in V:
            vec.extend(row)
        for row in W:
            vec.extend(row)
        return vec
    
    def _unflatten(vec, n, m, p, rank):
        U = []
        for i in range(n*m):
            U.append(vec[i*rank:(i+1)*rank])
        V = []
        for i in range(m*p):
            V.append(vec[(n*m + i)*rank:(n*m + i + 1)*rank])
        W = []
        for i in range(p*n):
            W.append(vec[(n*m + m*p + i)*rank:(n*m + m*p + i + 1)*rank])
        return U, V, W
    
    def _residuals(vec, n, m, p, rank):
        U, V, W = _unflatten(vec, n, m, p, rank)
        res = []
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                    lhs = 0.0
                    for r in range(rank):
                        lhs += U[a][r] * V[b][r] * W[c][r]
                    res.append(lhs - expected)
        return res
    
    # Initialize with the rank-11 solution, dropping one column
    vec_init = _flatten(U, V, W, n, m, p, rank)
    
    # Try to solve
    try:
        sol = least_squares(_residuals, vec_init, args=(n, m, p, rank), max_nfev=500)
        if sol.success or np.max(np.abs(sol.fun)) < 1e-8:
            U_sol, V_sol, W_sol = _unflatten(sol.fun, n, m, p, rank)
            # Verify
            if _verify(U_sol, V_sol, W_sol, n, m, p):
                # Round to clean values
                U_clean = [[round(x, 10) for x in row] for row in U_sol]
                V_clean = [[round(x, 10) for x in row] for row in V_sol]
                W_clean = [[round(x, 10) for x in row] for row in W_sol]
                if _verify(U_clean, V_clean, W_clean, n, m, p):
                    return {"U": U_clean, "V": V_clean, "W": W_clean}
    except Exception:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First try rank 10
    result = _try_rank_10_search(deadline)
    if result is not None:
        return result
    
    # Fall back to rank 11
    result = _try_rank_11()
    if result is not None:
        return result
    
    # Fallback to schoolbook rank 12
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
