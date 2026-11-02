"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition. Returns True if correct."""
    r = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for alpha in range(r):
                    s += U_arr[i*m+j, alpha] * V_arr[j*p+k, alpha] * W_arr[k*n+i, alpha]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Standard Strassen 2x2x2 decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u: list of (li, lj, val) for A[li][lj]
    # v: list of (lj, lk, val) for B[lj][lk]
    # w: list of (li, lk, val) for C[li][lk]
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A10+A11)B00 -> contributes to C10
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3 = A00(B01-B11) -> contributes to C01 with +, C11 with -
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,-1)]),
        # M4 = A11(B10-B00) -> contributes to C00 with -, C10 with +
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1), (1,0,1)]),
        # M5 = (A00+A01)B11 -> contributes to C01
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1)]),
        # M6 = (A10-A00)(B00+B01) -> contributes to C00
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,0,1)]),
        # M7 = (A01-A11)(B10+B11) -> contributes to C11
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(1,1,1)]),
    ]
    
    # Verify Strassen works for 2x2x2
    def _verify_strassen():
        rank = 7
        U = [[0]*rank for _ in range(4)]  # 2x2 = 4
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        for col, (u_list, v_list, w_list) in enumerate(strassen_terms):
            for (li, lj, val) in u_list:
                U[li*2+lj][col] += val
            for (lj, lk, val) in v_list:
                V[lj*2+lk][col] += val
            for (li, lk, val) in w_list:
                W[lk*2+li][col] += val
        # Verify: sum_alpha U[i*2+j, alpha] * V[j*2+k, alpha] * W[k*2+i, alpha] = 1 for all i,j,k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for alpha in range(rank):
                        s += U[i*2+j][alpha] * V[j*2+k][alpha] * W[k*2+i][alpha]
                    if s != 1:
                        return False
        return True
    
    if not _verify_strassen():
        # Fallback: trivial rank 8 for 2x2x2
        return {"U": None, "V": None, "W": None}  # Should not happen
    
    # Build full 2x5x6 decomposition
    # Part 1: j=0, 2x1x6 -> rank 12
    # Part 2: j=1,2 -> 2x2x6 -> 3 blocks of 2x2x2 -> rank 21
    # Part 3: j=3,4 -> 2x2x6 -> 3 blocks of 2x2x2 -> rank 21
    # Total: 12 + 21 + 21 = 54
    
    rank = 54
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Part 1: j = 0
    for i in range(n):
        for k in range(p):
            U[i*m + 0][col] = 1
            V[0*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    # Parts 2 & 3: j in {1,2} and {3,4}
    for block_idx in range(2):
        j_base = 1 + block_idx * 2
        for s in range(3):
            k_base = s * 2
            for term in strassen_terms:
                u_list, v_list, w_list = term
                for (li, lj, val) in u_list:
                    U[li * m + (j_base + lj)][col] += val
                for (lj, lk, val) in v_list:
                    V[(j_base + lj) * p + (k_base + lk)][col] += val
                for (li, lk, val) in w_list:
                    W[(k_base + lk) * n + li][col] += val
                col += 1
    
    # Final verification
    if time.time() < deadline and _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, return trivial rank 60
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
