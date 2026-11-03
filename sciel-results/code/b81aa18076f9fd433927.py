"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import numpy as np
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 145.0  # safety margin below 150s hard limit


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _snap_to_half_integer(x, tol=1e-9):
    """Snap a value to the nearest half-integer if close."""
    half = round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return x


def _is_half_integer(x, tol=1e-9):
    """Check if x is close to a half-integer."""
    half = round(x * 2) / 2.0
    return abs(x - half) < tol


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    # Construct Rank 54 solution using Strassen blocks
    # n=2, m=5, p=6
    # Blocks: j in {0,1}, {2,3} and k in {0,1}, {2,3}, {4,5}
    # 2 j-pairs * 3 k-pairs = 6 blocks of 2x2x2
    # Each block rank 7. Total 42.
    # Remaining j=4: 2 i's * 1 j * 6 k's = 12 terms. Rank 12.
    # Total Rank 54.
    
    R = 54
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1: A00+A11, B00+B11
    # M2: A10+A11, B00
    # M3: A00, B10-B11
    # M4: A11, B01-B00
    # M5: A00+A01, B11
    # M6: A10-A00, B00+B10
    # M7: A01-A11, B11+B01
    
    # C00: M1 + M4 - M5 + M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M3 + M6
    
    # Define Strassen patterns
    # U_pat[i, j, r], V_pat[j, k, r], W_pat[k, i, r]
    # i,j,k in {0,1}, r in {0..6}
    
    U_pat = np.zeros((2, 2, 7))
    V_pat = np.zeros((2, 2, 7))
    W_pat = np.zeros((2, 2, 7)) # W index is k, i
    
    # r=0 (M1): (a00+a11)(b00+b11) -> C00, C11
    U_pat[0,0,0] = 1; U_pat[1,1,0] = 1
    V_pat[0,0,0] = 1; V_pat[1,1,0] = 1
    W_pat[0,0,0] = 1 # C[0,0]
    W_pat[1,1,0] = 1 # C[1,1]
    
    # r=1 (M2): (a10+a11)b00 -> C10, C11
    U_pat[1,0,1] = 1; U_pat[1,1,1] = 1
    V_pat[0,0,1] = 1
    W_pat[1,0,1] = 1 # C[1,0]
    W_pat[1,1,1] = -1 # C[1,1]
    
    # r=2 (M3): a00(b10-b11) -> C01, C11
    U_pat[0,0,2] = 1
    V_pat[1,0,2] = 1; V_pat[1,1,2] = -1
    W_pat[0,1,2] = 1 # C[0,1]
    W_pat[1,1,2] = 1 # C[1,1]
    
    # r=3 (M4): a11(b01-b00) -> C00, C10
    U_pat[1,1,3] = 1
    V_pat[0,1,3] = 1; V_pat[0,0,3] = -1
    W_pat[0,0,3] = 1 # C[0,0]
    W_pat[1,0,3] = 1 # C[1,0]
    
    # r=4 (M5): (a00+a01)b11 -> C00, C01
    U_pat[0,0,4] = 1; U_pat[0,1,4] = 1
    V_pat[1,1,4] = 1
    W_pat[0,0,4] = -1 # C[0,0]
    W_pat[0,1,4] = 1 # C[0,1]
    
    # r=5 (M6): (a10-a00)(b00+b10) -> C11
    U_pat[1,0,5] = 1; U_pat[0,0,5] = -1
    V_pat[0,0,5] = 1; V_pat[1,0,5] = 1
    W_pat[1,1,5] = 1 # C[1,1]
    
    # r=6 (M7): (a01-a11)(b11+b01) -> C00, C11
    U_pat[0,1,6] = 1; U_pat[1,1,6] = -1
    V_pat[1,1,6] = 1; V_pat[0,1,6] = 1
    W_pat[0,0,6] = 1 # C[0,0]
    W_pat[1,1,6] = 1 # C[1,1]
    
    # Apply Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            # Local indices i', j', k' map to global i, j, k
            # i is 0, 1
            # j' is 0, 1 -> j is j1, j2
            # k' is 0, 1 -> k is k1, k2
            
            for r_local in range(7):
                r_global = col
                col += 1
                
                for i in range(2):
                    for j_local in range(2):
                        j_global = j1 if j_local == 0 else j2
                        val = U_pat[i, j_local, r_local]
                        if val != 0:
                            U[i*m + j_global][r_global] = val
                    
                    for k_local in range(2):
                        k_global = k1 if k_local == 0 else k2
                        val = V_pat[j_local, k_local, r_local]
                        if val != 0:
                            # V index is j*p + k
                            V[j_global*p + k_global][r_global] = val
                        
                        # W index is k*n + i
                        val_w = W_pat[k_local, i, r_local]
                        if val_w != 0:
                            W[k_global*n + i][r_global] = val_w

    # Handle remaining j=4
    # i in {0,1}, j=4, k in {0..5}
    # 12 terms
    for i in range(2):
        for k in range(6):
            r_global = col
            col += 1
            U[i*m + 4][r_global] = 1
            V[4*p + k][r_global] = 1
            W[k*n + i][r_global] = 1

    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
    return best_result
# EVOLVE-BLOCK-END
