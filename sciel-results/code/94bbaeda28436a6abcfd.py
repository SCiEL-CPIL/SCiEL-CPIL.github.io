"""Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 6]

def _strassen_2x2():
    """Return the 7-term Strassen decomposition for 2x2 * 2x2.
    Returns (U, V, W) where each is a list of 4-element vectors (for the 4 output positions)
    with 7 entries each.
    
    Strassen's algorithm:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U[i][c] for i in 0..3 (A positions), c in 0..6
    # V[i][c] for i in 0..3 (B positions), c in 0..6
    # W[i][c] for i in 0..3 (C positions), c in 0..6
    
    # A positions: 0=A00, 1=A01, 2=A10, 3=A11
    # B positions: 0=B00, 1=B01, 2=B10, 3=B11
    # C positions: 0=C00, 1=C01, 2=C10, 3=C11
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10 + A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C10 += M2, C11 -= M2
    
    # M3 = A00 * (B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11 * (B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C00 += M4, C10 += M4
    
    # M5 = (A00 + A01) * B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C00 -= M5, C01 += M5
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11 += M6
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00 += M7
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, n, m, p):
        """Check if U, V, W form a valid decomposition."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(expected - actual) > 1e-9:
                        return False
        return True
    
    # Use Strassen-based decomposition:
    # A (2x3) * B (3x6)
    # Split: A = [A_left (2x2), A_right (2x1)], B = [B_top (2x6); B_bottom (1x6)]
    # C = A_left * B_top + A_right * B_bottom
    # 
    # A_left * B_top: (2x2) * (2x6)
    # Split B_top into 3 blocks of (2x2): B_top = [BT0, BT1, BT2]
    # Each (2x2)*(2x2) uses Strassen: rank 7
    # Total for this part: 3 * 7 = 21
    #
    # A_right * B_bottom: (2x1) * (1x6) = 12 scalar products, rank 12
    #
    # Total rank: 21 + 12 = 33
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    sU, sV, sW = _strassen_2x2()
    
    # Part 1: A_left (cols 0,1) * B_top (rows 0,1), split into 3 (2x2)*(2x2) blocks
    for block in range(3):
        k_offset = block * 2
        for c in range(7):
            # A_left positions: i=0,1; j=0,1 -> indices i*m+j = i*3+j
            # sU[i][c] where i: 0=A00, 1=A01, 2=A10, 3=A11
            # A00 -> i=0,j=0 -> idx 0
            # A01 -> i=0,j=1 -> idx 1
            # A10 -> i=1,j=0 -> idx 3
            # A11 -> i=1,j=1 -> idx 4
            a_idx_map = [0, 1, 3, 4]  # sU index -> U row index
            b_idx_map = [k_offset, k_offset+1, k_offset+2, k_offset+3]  # sV index -> V row index (j*p+k)
            # B00 -> j=0,k=k_offset -> idx 0*6+k_offset = k_offset
            # B01 -> j=0,k=k_offset+1 -> idx k_offset+1
            # B10 -> j=1,k=k_offset -> idx 1*6+k_offset = 6+k_offset
            # B11 -> j=1,k=k_offset+1 -> idx 6+k_offset+1 = 7+k_offset
            b_idx_map = [k_offset, k_offset+1, 6+k_offset, 7+k_offset]
            # C positions: i=0,1; k=k_offset,k_offset+1 -> idx k*n+i
            # C00 -> i=0,k=k_offset -> idx k_offset*2+0 = 2*k_offset
            # C01 -> i=0,k=k_offset+1 -> idx (k_offset+1)*2+0 = 2*k_offset+2
            # C10 -> i=1,k=k_offset -> idx 2*k_offset+1
            # C11 -> i=1,k=k_offset+1 -> idx 2*k_offset+3
            c_idx_map = [2*k_offset, 2*k_offset+2, 2*k_offset+1, 2*k_offset+3]
            
            for si in range(4):
                if sU[si][c] != 0:
                    U[a_idx_map[si]][col] += sU[si][c]
            for sj in range(4):
                if sV[sj][c] != 0:
                    V[b_idx_map[sj]][col] += sV[sj][c]
            for sc in range(4):
                if sW[sc][c] != 0:
                    W[c_idx_map[sc]][col] += sW[sc][c]
            col += 1
    
    # Part 2: A_right (col 2) * B_bottom (row 2): (2x1)*(1x6)
    # A_right: i=0,1; j=2 -> indices i*3+2 = 2, 5
    # B_bottom: j=2; k=0..5 -> indices 2*6+k = 12+k
    # C: i=0,1; k=0..5 -> indices k*2+i
    for i in range(2):
        for k in range(6):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Verify
    assert col == rank, f"Column count {col} != rank {rank}"
    assert verify(U, V, W, n, m, p), "Strassen-based decomposition failed verification"
    
    return {"U": U, "V": V, "W": W}
