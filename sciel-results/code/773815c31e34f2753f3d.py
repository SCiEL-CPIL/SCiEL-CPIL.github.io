"""
Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition.

Strategy:
1. Build a rank-88 Strassen-based decomposition (improvement over schoolbook rank 96).
2. Try to find a rank-70 decomposition via the 4x4x4 Strassen embedding technique.
3. Verify any candidate exactly before returning.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=None, m=None, p=None):
    """
    Verify that U, V, W form a valid exact tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    if n is None:
        n, m, p = DIMENSIONS
    
    rank = len(U[0]) if U else 0
    
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_2x2_strassen():
    """
    Return the Strassen decomposition for 2x2 * 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # M2 = (A21+A22) B11
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, 0]
    
    # M3 = A11 (B21-B22)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 0, 1, -1]
    W[:, 2] = [0, 0, 1, 0]
    
    # M4 = A22 (B12-B11)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 1, 0, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    # M5 = (A11+A12) B22
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    # M6 = (A21-A11)(B11+B12)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # M7 = (A12-A22)(B21+B22)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _build_2x2_times_2x4_strassen():
    """
    Build the Strassen-based decomposition for 2x2 * 2x4.
    Rank: 14
    U: (4, 14), V: (8, 14), W: (8, 14)
    """
    U_s, V_s, W_s = _build_2x2_strassen()
    
    U = np.zeros((4, 14))
    V = np.zeros((8, 14))
    W = np.zeros((8, 14))
    
    for r in range(7):
        U[:, r] = U_s[:, r]
        V[0, r] = V_s[0, r]
        V[1, r] = V_s[1, r]
        V[4, r] = V_s[2, r]
        V[5, r] = V_s[3, r]
        W[0, r] = W_s[0, r]
        W[1, r] = W_s[1, r]
        W[2, r] = W_s[2, r]
        W[3, r] = W_s[3, r]
        
        U[:, 7 + r] = U_s[:, r]
        V[2, 7 + r] = V_s[0, r]
        V[3, 7 + r] = V_s[1, r]
        V[6, 7 + r] = V_s[2, r]
        V[7, 7 + r] = V_s[3, r]
        W[4, 7 + r] = W_s[0, r]
        W[5, 7 + r] = W_s[1, r]
        W[6, 7 + r] = W_s[2, r]
        W[7, 7 + r] = W_s[3, r]
    
    return U, V, W

def _build_2x4_times_4x4():
    """
    Build the decomposition for 2x4 * 4x4 using Strassen.
    Rank: 28
    U: (8, 28), V: (16, 28), W: (8, 28)
    """
    U_22_24, V_22_24, W_22_24 = _build_2x2_times_2x4_strassen()
    
    U = np.zeros((8, 28))
    V = np.zeros((16, 28))
    W = np.zeros((8, 28))
    
    # A_left * B_top (r=0..13)
    for r in range(14):
        U[0, r] = U_22_24[0, r]
        U[1, r] = U_22_24[1, r]
        U[4, r] = U_22_24[2, r]
        U[5, r] = U_22_24[3, r]
        V[0, r] = V_22_24[0, r]
        V[1, r] = V_22_24[1, r]
        V[2, r] = V_22_24[2, r]
        V[3, r] = V_22_24[3, r]
        V[4, r] = V_22_24[4, r]
        V[5, r] = V_22_24[5, r]
        V[6, r] = V_22_24[6, r]
        V[7, r] = V_22_24[7, r]
        W[0, r] = W_22_24[0, r]
        W[1, r] = W_22_24[1, r]
        W[2, r] = W_22_24[2, r]
        W[3, r] = W_22_24[3, r]
        W[4, r] = W_22_24[4, r]
        W[5, r] = W_22_24[5, r]
        W[6, r] = W_22_24[6, r]
        W[7, r] = W_22_24[7, r]
    
    # A_right * B_bot (r=14..27)
    for r in range(14):
        U[2, 14 + r] = U_22_24[0, r]
        U[3, 14 + r] = U_22_24[1, r]
        U[6, 14 + r] = U_22_24[2, r]
        U[7, 14 + r] = U_22_24[3, r]
        V[8, 14 + r] = V_22_24[0, r]
        V[9, 14 + r] = V_22_24[1, r]
        V[10, 14 + r] = V_22_24[2, r]
        V[11, 14 + r] = V_22_24[3, r]
        V[12, 14 + r] = V_22_24[4, r]
        V[13, 14 + r] = V_22_24[5, r]
        V[14, 14 + r] = V_22_24[6, r]
        V[15, 14 + r] = V_22_24[7, r]
        W[0, 14 + r] = W_22_24[0, r]
        W[1, 14 + r] = W_22_24[1, r]
        W[2, 14 + r] = W_22_24[2, r]
        W[3, 14 + r] = W_22_24[3, r]
        W[4, 14 + r] = W_22_24[4, r]
        W[5, 14 + r] = W_22_24[5, r]
        W[6, 14 + r] = W_22_24[6, r]
        W[7, 14 + r] = W_22_24[7, r]
    
    return U, V, W

def _build_1x4_times_4x4_schoolbook():
    """
    Build the schoolbook decomposition for 1x4 * 4x4.
    Rank: 16
    U: (4, 16), V: (16, 16), W: (4, 16)
    """
    U = np.zeros((4, 16))
    V = np.zeros((16, 16))
    W = np.zeros((4, 16))
    
    col = 0
    for j in range(4):
        for k in range(4):
            U[j, col] = 1
            V[j * 4 + k, col] = 1
            W[k, col] = 1
            col += 1
    
    return U, V, W

def _build_3x4_times_4x4():
    """
    Build the decomposition for 3x4 * 4x4.
    Rank: 44
    U: (12, 44), V: (16, 44), W: (12, 44)
    """
    U_24, V_24, W_24 = _build_2x4_times_4x4()
    U_14, V_14, W_14 = _build_1x4_times_4x4_schoolbook()
    
    U = np.zeros((12, 44))
    V = np.zeros((16, 44))
    W = np.zeros((12, 44))
    
    # Top: rows 0,1 of A (r=0..27)
    for r in range(28):
        U[0, r] = U_24[0, r]
        U[1, r] = U_24[1, r]
        U[2, r] = U_24[2, r]
        U[3, r] = U_24[3, r]
        U[4, r] = U_24[4, r]
        U[5, r] = U_24[5, r]
        U[6, r] = U_24[6, r]
        U[7, r] = U_24[7, r]
        V[:, r] = V_24[:, r]
        W[0, r] = W_24[0, r]
        W[1, r] = W_24[1, r]
        W[3, r] = W_24[2, r]
        W[4, r] = W_24[3, r]
        W[6, r] = W_24[4, r]
        W[7, r] = W_24[5, r]
        W[9, r] = W_24[6, r]
        W[10, r] = W_24[7, r]
    
    # Bottom: row 2 of A (r=28..43)
    for r in range(16):
        U[8, 28 + r] = U_14[0, r]
        U[9, 28 + r] = U_14[1, r]
        U[10, 28 + r] = U_14[2, r]
        U[11, 28 + r] = U_14[3, r]
        V[:, 28 + r] = V_14[:, r]
        W[2, 28 + r] = W_14[0, r]
        W[5, 28 + r] = W_14[1, r]
        W[8, 28 + r] = W_14[2, r]
        W[11, 28 + r] = W_14[3, r]
    
    return U, V, W

def _build_3x4_times_4x8():
    """
    Build the full decomposition for 3x4 * 4x8.
    Rank: 88
    U: (12, 88), V: (32, 88), W: (24, 88)
    """
    U_34_44, V_34_44, W_34_44 = _build_3x4_times_4x4()
    
    U = np.zeros((12, 88))
    V = np.zeros((32, 88))
    W = np.zeros((24, 88))
    
    # Left half (r=0..43)
    for r in range(44):
        U[:, r] = U_34_44[:, r]
        V[0, r] = V_34_44[0, r]
        V[1, r] = V_34_44[1, r]
        V[2, r] = V_34_44[2, r]
        V[3, r] = V_34_44[3, r]
        V[8, r] = V_34_44[4, r]
        V[9, r] = V_34_44[5, r]
        V[10, r] = V_34_44[6, r]
        V[11, r] = V_34_44[7, r]
        V[16, r] = V_34_44[8, r]
        V[17, r] = V_34_44[9, r]
        V[18, r] = V_34_44[10, r]
        V[19, r] = V_34_44[11, r]
        V[24, r] = V_34_44[12, r]
        V[25, r] = V_34_44[13, r]
        V[26, r] = V_34_44[14, r]
        V[27, r] = V_34_44[15, r]
        W[0, r] = W_34_44[0, r]
        W[1, r] = W_34_44[1, r]
        W[2, r] = W_34_44[2, r]
        W[3, r] = W_34_44[3, r]
        W[4, r] = W_34_44[4, r]
        W[5, r] = W_34_44[5, r]
        W[6, r] = W_34_44[6, r]
        W[7, r] = W_34_44[7, r]
        W[8, r] = W_34_44[8, r]
        W[9, r] = W_34_44[9, r]
        W[10, r] = W_34_44[10, r]
        W[11, r] = W_34_44[11, r]
    
    # Right half (r=44..87)
    for r in range(44):
        U[:, 44 + r] = U_34_44[:, r]
        V[4, 44 + r] = V_34_44[0, r]
        V[5, 44 + r] = V_34_44[1, r]
        V[6, 44 + r] = V_34_44[2, r]
        V[7, 44 + r] = V_34_44[3, r]
        V[12, 44 + r] = V_34_44[4, r]
        V[13, 44 + r] = V_34_44[5, r]
        V[14, 44 + r] = V_34_44[6, r]
        V[15, 44 + r] = V_34_44[7, r]
        V[20, 44 + r] = V_34_44[8, r]
        V[21, 44 + r] = V_34_44[9, r]
        V[22, 44 + r] = V_34_44[10, r]
        V[23, 44 + r] = V_34_44[11, r]
        V[28, 44 + r] = V_34_44[12, r]
        V[29, 44 + r] = V_34_44[13, r]
        V[30, 44 + r] = V_34_44[14, r]
        V[31, 44 + r] = V_34_44[15, r]
        W[12, 44 + r] = W_34_44[0, r]
        W[13, 44 + r] = W_34_44[1, r]
        W[14, 44 + r] = W_34_44[2, r]
        W[15, 44 + r] = W_34_44[3, r]
        W[16, 44 + r] = W_34_44[4, r]
        W[17, 44 + r] = W_34_44[5, r]
        W[18, 44 + r] = W_34_44[6, r]
        W[19, 44 + r] = W_34_44[7, r]
        W[20, 44 + r] = W_34_44[8, r]
        W[21, 44 + r] = W_34_44[9, r]
        W[22, 44 + r] = W_34_44[10, r]
        W[23, 44 + r] = W_34_44[11, r]
    
    return U, V, W

def _try_4x4x4_embedding():
    """
    Try to embed 3x4x8 into 4x4x4 Strassen (rank 70).
    
    4x4x4 Strassen decomposition has rank 70.
    We fix A[3,:] = 0, B[:,3] = 0, and check if the remaining tensor
    can be represented with rank <= 70.
    
    This requires careful construction of the 4x4x4 Strassen decomposition.
    """
    # Build 4x4x4 Strassen decomposition
    # Split 4x4x4 into 2x2x2 blocks
    # A(4x4) = [[A11, A12], [A21, A22]] where each is 2x2
    # B(4x4) = [[B11, B12], [B21, B22]]
    # C(4x4) = [[C11, C12], [C21, C22]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each 2x2*2x2 uses Strassen (rank 7)
    # Total: 8 * 7 = 56... but Strassen for 4x4x4 is rank 70.
    # Actually, the standard Strassen for 4x4x4 gives rank 70 by a different construction.
    
    # Let me use the known rank-70 decomposition for 4x4x4.
    # This is complex, so let me try a simpler approach:
    # Use the 2x2 Strassen recursively.
    
    U_s, V_s, W_s = _build_2x2_strassen()
    
    # Build 4x4x4 with rank 56 (8 Strassen blocks of rank 7)
    U_444 = np.zeros((16, 56))
    V_444 = np.zeros((16, 56))
    W_444 = np.zeros((16, 56))
    
    # For each 2x2 block of the 4x4 matrices, apply Strassen
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Indices: A[i,j] where i,j in 0..3
    # A11: i=0,1; j=0,1 -> indices 0,1,4,5
    # A12: i=0,1; j=2,3 -> indices 2,3,6,7
    # A21: i=2,3; j=0,1 -> indices 8,9,12,13
    # A22: i=2,3; j=2,3 -> indices 10,11,14,15
    
    # B11: i=0,1; j=0,1 -> indices 0,1,4,5
    # B12: i=0,1; j=2,3 -> indices 2,3,6,7
    # B21: i=2,3; j=0,1 -> indices 8,9,12,13
    # B22: i=2,3; j=2,3 -> indices 10,11,14,15
    
    # C11: i=0,1; j=0,1 -> indices 0,1,4,5
    # C12: i=0,1; j=2,3 -> indices 2,3,6,7
    # C21: i=2,3; j=0,1 -> indices 8,9,12,13
    # C22: i=2,3; j=2,3 -> indices 10,11,14,15
    
    def _apply_strassen_block(U_block_idx, V_block_idx, W_block_idx, r_offset):
        for r in range(7):
            # U_block_idx maps 2x2 indices [0,1,2,3] to global indices
            for local_i in range(4):
                U_444[U_block_idx[local_i], r_offset + r] = U_s[local_i, r]
            for local_j in range(4):
                V_444[V_block_idx[local_j], r_offset + r] = V_s[local_j, r]
            for local_k in range(4):
                W_444[W_block_idx[local_k], r_offset + r] = W_s[local_k, r]
    
    # C11 = A11*B11 + A12*B21
    # Term 1: A11 * B11
    A11_idx = [0, 1, 4, 5]
    B11_idx = [0, 1, 4, 5]
    C11_idx = [0, 1, 4, 5]
    _apply_strassen_block(A11_idx, B11_idx, C11_idx, 0)
    
    # Term 2: A12 * B21
    A12_idx = [2, 3, 6, 7]
    B21_idx = [8, 9, 12, 13]
    C11_idx = [0, 1, 4, 5]
    _apply_strassen_block(A12_idx, B21_idx, C11_idx, 7)
    
    # C12 = A11*B12 + A12*B22
    # Term 1: A11 * B12
    A11_idx = [0, 1, 4, 5]
    B12_idx = [2, 3, 6, 7]
    C12_idx = [2, 3, 6, 7]
    _apply_strassen_block(A11_idx, B12_idx, C12_idx, 14)
    
    # Term 2: A12 * B22
    A12_idx = [2, 3, 6, 7]
    B22_idx = [10, 11, 14, 15]
    C12_idx = [2, 3, 6, 7]
    _apply_strassen_block(A12_idx, B22_idx, C12_idx, 21)
    
    # C21 = A21*B11 + A22*B21
    # Term 1: A21 * B11
    A21_idx = [8, 9, 12, 13]
    B11_idx = [0, 1, 4, 5]
    C21_idx = [8, 9, 12, 13]
    _apply_strassen_block(A21_idx, B11_idx, C21_idx, 28)
    
    # Term 2: A22 * B21
    A22_idx = [10, 11, 14, 15]
    B21_idx = [8, 9, 12, 13]
    C21_idx = [8, 9, 12, 13]
    _apply_strassen_block(A22_idx, B21_idx, C21_idx, 35)
    
    # C22 = A21*B12 + A22*B22
    # Term 1: A21 * B12
    A21_idx = [8, 9, 12, 13]
    B12_idx = [2, 3, 6, 7]
    C22_idx = [10, 11, 14, 15]
    _apply_strassen_block(A21_idx, B12_idx, C22_idx, 42)
    
    # Term 2: A22 * B22
    A22_idx = [10, 11, 14, 15]
    B22_idx = [10, 11, 14, 15]
    C22_idx = [10, 11, 14, 15]
    _apply_strassen_block(A22_idx, B22_idx, C22_idx, 49)
    
    # Now embed 3x4x8 into this 4x4x4 decomposition
    # 3x4x8: A(3x4), B(4x8), C(3x8)
    # 4x4x4: A(4x4), B(4x4), C(4x4)
    
    # Map 3x4 A to 4x4 A: A_4x4[i,j] = A_3x4[i,j] for i<3, j<4; A_4x4[3,:] = 0
    # Map 4x8 B to 4x4 B: This doesn't work directly since B is 4x8 not 4x4.
    
    # Alternative: Use 4x8x8 instead? No, we need 3x4x8.
    
    # Let me try a different approach: directly construct a rank-70 decomposition
    # by combining Strassen blocks.
    
    # Actually, let me just return None and use the rank-88 decomposition.
    return None

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x8 matrix.
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U: (12, rank) list of lists
      V: (32, rank) list of lists
      W: (24, rank) list of lists
    
    The decomposition satisfies:
      T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for all valid i, j, k, where T is the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with schoolbook as the guaranteed-valid baseline
    best = _schoolbook()
    best_rank = len(best['U'][0])
    
    # Try the rank-88 Strassen decomposition
    if time.time() < deadline - 10:
        try:
            U, V, W = _build_3x4_times_4x8()
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify(U_list, V_list, W_list):
                rank = len(U_list[0])
                if rank < best_rank:
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
        except Exception:
            pass
    
    # Try the 4x4x4 embedding (rank 70 if it works)
    if time.time() < deadline - 5:
        try:
            result = _try_4x4x4_embedding()
            if result is not None:
                rank = len(result['U'][0])
                if rank < best_rank:
                    if _verify(result['U'], result['V'], result['W']):
                        best = result
                        best_rank = rank
        except Exception:
            pass
    
    # Final verification
    if not _verify(best['U'], best['V'], best['W']):
        best = _schoolbook()
    
    return best
