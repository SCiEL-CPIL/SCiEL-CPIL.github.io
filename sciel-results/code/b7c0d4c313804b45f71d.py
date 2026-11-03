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
    
    # Rank 42 solution
    # Partition j into {0,1,2} and {3,4}
    # Partition k into {0,1,2} and {3,4,5}
    
    # Total rank 42
    R = 42
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen patterns for 2x2x2
    U_pat = np.zeros((2, 2, 7))
    V_pat = np.zeros((2, 2, 7))
    W_pat = np.zeros((2, 2, 7))
    
    # r=0 (M1)
    U_pat[0,0,0] = 1; U_pat[1,1,0] = 1
    V_pat[0,0,0] = 1; V_pat[1,1,0] = 1
    W_pat[0,0,0] = 1; W_pat[1,1,0] = 1
    
    # r=1 (M2)
    U_pat[1,0,1] = 1; U_pat[1,1,1] = 1
    V_pat[0,0,1] = 1
    W_pat[0,1,1] = 1; W_pat[1,1,1] = -1
    
    # r=2 (M3)
    U_pat[0,0,2] = 1
    V_pat[1,0,2] = 1; V_pat[1,1,2] = -1
    W_pat[1,0,2] = 1; W_pat[1,1,2] = 1
    
    # r=3 (M4)
    U_pat[1,1,3] = 1
    V_pat[0,1,3] = 1; V_pat[0,0,3] = -1
    W_pat[0,0,3] = 1; W_pat[0,1,3] = 1
    
    # r=4 (M5)
    U_pat[0,0,4] = 1; U_pat[0,1,4] = 1
    V_pat[1,1,4] = 1
    W_pat[0,0,4] = -1; W_pat[1,0,4] = 1
    
    # r=5 (M6)
    U_pat[1,0,5] = 1; U_pat[0,0,5] = -1
    V_pat[0,0,5] = 1; V_pat[1,0,5] = 1
    W_pat[1,1,5] = 1
    
    # r=6 (M7)
    U_pat[0,1,6] = 1; U_pat[1,1,6] = -1
    V_pat[1,1,6] = 1; V_pat[0,1,6] = 1
    W_pat[0,0,6] = 1; W_pat[1,0,6] = 1

    def add_strassen_block(j_start, k_start):
        nonlocal col
        for r_local in range(7):
            r_global = col
            col += 1
            for i in range(2):
                for j_local in range(2):
                    j_global = j_start + j_local
                    val = U_pat[i, j_local, r_local]
                    if val != 0:
                        U[i*m + j_global][r_global] = val
            for j_local in range(2):
                j_global = j_start + j_local
                for k_local in range(2):
                    k_global = k_start + k_local
                    val = V_pat[j_local, k_local, r_local]
                    if val != 0:
                        V[j_global*p + k_global][r_global] = val
            for k_local in range(2):
                k_global = k_start + k_local
                for i in range(2):
                    val = W_pat[k_local, i, r_local]
                    if val != 0:
                        W[k_global*n + i][r_global] = val

    def add_2x2x1_block(j_start, k_start):
        # Rank 2
        nonlocal col
        # Term 1: j=0
        r1 = col; col += 1
        for i in range(2):
            U[i*m + j_start][r1] = 1
        V[j_start*p + k_start][r1] = 1
        for i in range(2):
            W[k_start*n + i][r1] = 1
            
        # Term 2: j=1
        r2 = col; col += 1
        for i in range(2):
            U[i*m + j_start + 1][r2] = 1
        V[j_start*p + 1 + k_start][r2] = 1
        for i in range(2):
            W[k_start*n + i][r2] = 1

    def add_2x1x2_block(j_start, k_start):
        # Rank 2
        nonlocal col
        # Term 1: k=0
        r1 = col; col += 1
        for i in range(2):
            U[i*m + j_start][r1] = 1
        V[j_start*p + k_start][r1] = 1
        W[k_start*n + 0][r1] = 1
        W[k_start*n + 1][r1] = 1
        
        # Term 2: k=1
        r2 = col; col += 1
        for i in range(2):
            U[i*m + j_start][r2] = 1
        V[j_start*p + k_start + 1][r2] = 1
        W[(k_start+1)*n + 0][r2] = 1
        W[(k_start+1)*n + 1][r2] = 1

    def add_2x1x1_block(j_start, k_start):
        # Rank 1
        nonlocal col
        r1 = col; col += 1
        for i in range(2):
            U[i*m + j_start][r1] = 1
        V[j_start*p + k_start][r1] = 1
        for i in range(2):
            W[k_start*n + i][r1] = 1

    # Block 1: j in {0,1,2}, k in {0,1,2} -> 2x3x3
    # Split j into {0,1} and {2}, k into {0,1} and {2}
    # Sub-blocks:
    # 1.1 j{0,1}, k{0,1}: 2x2x2 (Rank 7)
    add_strassen_block(0, 0)
    # 1.2 j{0,1}, k{2}: 2x2x1 (Rank 2)
    add_2x2x1_block(0, 2)
    # 1.3 j{2}, k{0,1}: 2x1x2 (Rank 2)
    add_2x1x2_block(2, 0)
    # 1.4 j{2}, k{2}: 2x1x1 (Rank 1)
    add_2x1x1_block(2, 2)
    
    # Block 2: j in {0,1,2}, k in {3,4,5} -> 2x3x3
    # 2.1 j{0,1}, k{3,4}: 2x2x2 (Rank 7)
    add_strassen_block(0, 3)
    # 2.2 j{0,1}, k{5}: 2x2x1 (Rank 2)
    add_2x2x1_block(0, 5)
    # 2.3 j{2}, k{3,4}: 2x1x2 (Rank 2)
    add_2x1x2_block(2, 3)
    # 2.4 j{2}, k{5}: 2x1x1 (Rank 1)
    add_2x1x1_block(2, 5)
    
    # Block 3: j in {3,4}, k in {0,1,2} -> 2x2x3
    # 3.1 j{3,4}, k{0,1}: 2x2x2 (Rank 7)
    add_strassen_block(3, 0)
    # 3.2 j{3,4}, k{2}: 2x2x1 (Rank 2)
    add_2x2x1_block(3, 2)
    
    # Block 4: j in {3,4}, k in {3,4,5} -> 2x2x3
    # 4.1 j{3,4}, k{3,4}: 2x2x2 (Rank 7)
    add_strassen_block(3, 3)
    # 4.2 j{3,4}, k{5}: 2x2x1 (Rank 2)
    add_2x2x1_block(3, 5)

    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
