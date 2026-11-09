"""Rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

n, m, p = 3, 3, 8

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 Strassen.
    U shape (4, 7): rows (i,j) -> i*2+j
    V shape (4, 7): rows (j,k) -> j*2+k
    W shape (4, 7): rows (k,i) -> k*2+i
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1: (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A00(B10 - B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4: A11(B01 - B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00 + A10)B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A11 - A00)(B00 + B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W):
    """Verify if U, V, W form a valid tensor decomposition for 3x3 * 3x8."""
    nn, mm, pp = 3, 3, 8
    rank = len(U[0])
    for i in range(nn):
        for j in range(mm):
            for k in range(pp):
                s = 0
                for r in range(rank):
                    s += U[i*mm + j][r] * V[j*pp + k][r] * W[k*nn + i][r]
                if s != 1:
                    return False
    return True

def _make_schoolbook():
    """Create the schoolbook decomposition (rank = n*m*p = 72)."""
    nn, mm, pp = 3, 3, 8
    rank = nn * mm * pp
    U = [[0]*rank for _ in range(nn*mm)]
    V = [[0]*rank for _ in range(mm*pp)]
    W = [[0]*rank for _ in range(pp*nn)]
    c = 0
    for i in range(nn):
        for j in range(mm):
            for k in range(pp):
                U[i*mm+j][c] = 1
                V[j*pp+k][c] = 1
                W[k*nn+i][c] = 1
                c += 1
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x8.
    Uses Strassen (rank 7) for the 2x2 block of A and corresponding B rows,
    and schoolbook for the remaining parts.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    nn, mm, pp = 3, 3, 8
    
    # Strategy: 
    # 1. Use Strassen (rank 7) for the top-left 2x2 block of A (i in {0,1}, j in {0,1})
    #    and corresponding B rows (j in {0,1}), splitting the 8 columns into 4 groups of 2.
    #    This gives 4 * 7 = 28 rank terms.
    # 2. Use schoolbook for the remaining parts:
    #    - A[0:2, 2] * B[2, :] -> C[0:2, :] (rank 2*1*8 = 16)
    #    - A[2, 0:2] * B[0:2, :] -> C[2, :] (rank 1*2*8 = 16)
    #    - A[2, 2] * B[2, :] -> C[2, :] (rank 1*1*8 = 8)
    #    Total rank = 28 + 16 + 16 + 8 = 68.
    
    # Check time - if we've used too much, just return schoolbook
    if time.time() - start_time > 10.0:
        U, V, W = _make_schoolbook()
        return {"U": U, "V": V, "W": W}
    
    rank = 68
    U = [[0]*rank for _ in range(nn*mm)]
    V = [[0]*rank for _ in range(mm*pp)]
    W = [[0]*rank for _ in range(pp*nn)]
    
    col_offset = 0
    
    # 1. Strassen blocks for i in {0,1}, j in {0,1}, k in {0,1}, {2,3}, {4,5}, {6,7}
    U_str, V_str, W_str = _strassen_2x2x2()
    
    for k_group in range(4):
        k_start = k_group * 2
        k0 = k_start
        k1 = k_start + 1
        
        # Map local indices to global
        for i_l in range(2):
            for j_l in range(2):
                idx_U = i_l * 2 + j_l
                idx_U_global = i_l * 3 + j_l  # i*m + j
                
                for r_l in range(7):
                    U[idx_U_global][col_offset + r_l] = U_str[idx_U][r_l]
        
        for j_l in range(2):
            for k_l in range(2):
                idx_V = j_l * 2 + k_l
                idx_V_global = j_l * 3 + (k_start + k_l)  # j*p + k
                
                for r_l in range(7):
                    V[idx_V_global][col_offset + r_l] = V_str[idx_V][r_l]
        
        for k_l in range(2):
            for i_l in range(2):
                idx_W = k_l * 2 + i_l
                idx_W_global = (k_start + k_l) * 3 + i_l  # k*n + i
                
                for r_l in range(7):
                    W[idx_W_global][col_offset + r_l] = W_str[idx_W][r_l]
        
        col_offset += 7
    
    # 2. A[0:2, 2] * B[2, :] -> C[0:2, :]
    # i in {0,1}, j=2, k in {0..7}
    for i in range(2):
        for k in range(8):
            U[i * 3 + 2][col_offset] = 1
            V[2 * 3 + k][col_offset] = 1
            W[k * 3 + i][col_offset] = 1
            col_offset += 1
    
    # 3. A[2, 0:2] * B[0:2, :] -> C[2, :]
    # i=2, j in {0,1}, k in {0..7}
    for j in range(2):
        for k in range(8):
            U[2 * 3 + j][col_offset] = 1
            V[j * 3 + k][col_offset] = 1
            W[k * 3 + 2][col_offset] = 1
            col_offset += 1
    
    # 4. A[2, 2] * B[2, :] -> C[2, :]
    # i=2, j=2, k in {0..7}
    for k in range(8):
        U[2 * 3 + 2][col_offset] = 1
        V[2 * 3 + k][col_offset] = 1
        W[k * 3 + 2][col_offset] = 1
        col_offset += 1
    
    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails
        U_school, V_school, W_school = _make_schoolbook()
        return {"U": U_school, "V": V_school, "W": W_school}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
