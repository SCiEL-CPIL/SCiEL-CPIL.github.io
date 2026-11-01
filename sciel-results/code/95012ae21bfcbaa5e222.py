"""Matrix multiplication 2x2, 2x7 exact tensor decomposition search."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def _build_schoolbook():
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

def _verify(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _get_strassen_decomp():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U rows: 0:A00, 1:A01, 2:A10, 3:A11
    V rows: 0:B00, 1:B01, 2:B10, 3:B11
    W rows: 0:C00, 1:C01, 2:C10, 3:C11
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, -1],  # A01
        [0, 1, 0, 0, 0, -1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    return U, V, W

def _get_schoolbook_1col_decomp():
    """
    Returns U, V, W for 2x2x1 Schoolbook decomposition.
    U: 4x4, V: 2x4, W: 2x4
    Indices:
    U rows: 0:A00, 1:A01, 2:A10, 3:A11
    V rows: 0:B00, 1:B10 (Note: local index for col 0)
    W rows: 0:C0, 1:C1
    """
    U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V = [
        [1, 0, 1, 0], # B00
        [0, 1, 0, 1]  # B10
    ]
    W = [
        [1, 1, 0, 0], # C0
        [0, 0, 1, 1]  # C1
    ]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Rank 25 Construction
    # 3 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x1 (Schoolbook, rank 4)
    # Total Rank = 7 * 3 + 4 = 25
    
    R = 25
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    StrU, StrV, StrW = _get_strassen_decomp()
    SchU, SchV, SchW = _get_schoolbook_1col_decomp()
    
    # Block 1: Cols 0, 1 (k=0,1)
    # U rows: 0..3 (same for all blocks)
    # V rows: j*p+k. j=0,1. k=0,1.
    #   j=0, k=0 -> 0
    #   j=0, k=1 -> 1
    #   j=1, k=0 -> 7
    #   j=1, k=1 -> 8
    # W rows: k*n+i. k=0,1. i=0,1.
    #   k=0, i=0 -> 0
    #   k=0, i=1 -> 1
    #   k=1, i=0 -> 2
    #   k=1, i=1 -> 3
    
    col_offset = 0
    for r in range(7):
        for i in range(4):
            U[i][col_offset + r] = StrU[i][r]
        # Map StrV local rows (0:B00, 1:B01, 2:B10, 3:B11) to global V rows
        V[0][col_offset + r] = StrV[0][r]   # B00
        V[1][col_offset + r] = StrV[1][r]   # B01
        V[7][col_offset + r] = StrV[2][r]   # B10
        V[8][col_offset + r] = StrV[3][r]   # B11
        
        # Map StrW local rows (0:C00, 1:C01, 2:C10, 3:C11) to global W rows
        W[0][col_offset + r] = StrW[0][r]   # C00
        W[1][col_offset + r] = StrW[1][r]   # C01
        W[2][col_offset + r] = StrW[2][r]   # C10
        W[3][col_offset + r] = StrW[3][r]   # C11
    col_offset += 7

    # Block 2: Cols 2, 3 (k=2,3)
    # V rows:
    #   j=0, k=2 -> 2
    #   j=0, k=3 -> 3
    #   j=1, k=2 -> 9
    #   j=1, k=3 -> 10
    # W rows:
    #   k=2, i=0 -> 4
    #   k=2, i=1 -> 5
    #   k=3, i=0 -> 6
    #   k=3, i=1 -> 7
    
    for r in range(7):
        for i in range(4):
            U[i][col_offset + r] = StrU[i][r]
        
        V[2][col_offset + r] = StrV[0][r]   # B02
        V[3][col_offset + r] = StrV[1][r]   # B03
        V[9][col_offset + r] = StrV[2][r]   # B12
        V[10][col_offset + r] = StrV[3][r]  # B13
        
        W[4][col_offset + r] = StrW[0][r]   # C20
        W[5][col_offset + r] = StrW[1][r]   # C21
        W[6][col_offset + r] = StrW[2][r]   # C30
        W[7][col_offset + r] = StrW[3][r]   # C31
    col_offset += 7

    # Block 3: Cols 4, 5 (k=4,5)
    # V rows:
    #   j=0, k=4 -> 4
    #   j=0, k=5 -> 5
    #   j=1, k=4 -> 11
    #   j=1, k=5 -> 12
    # W rows:
    #   k=4, i=0 -> 8
    #   k=4, i=1 -> 9
    #   k=5, i=0 -> 10
    #   k=5, i=1 -> 11
    
    for r in range(7):
        for i in range(4):
            U[i][col_offset + r] = StrU[i][r]
        
        V[4][col_offset + r] = StrV[0][r]   # B04
        V[5][col_offset + r] = StrV[1][r]   # B05
        V[11][col_offset + r] = StrV[2][r]  # B14
        V[12][col_offset + r] = StrV[3][r]  # B15
        
        W[8][col_offset + r] = StrW[0][r]   # C40
        W[9][col_offset + r] = StrW[1][r]   # C41
        W[10][col_offset + r] = StrW[2][r]  # C50
        W[11][col_offset + r] = StrW[3][r]  # C51
    col_offset += 7

    # Block 4: Col 6 (k=6)
    # V rows:
    #   j=0, k=6 -> 6
    #   j=1, k=6 -> 13
    # W rows:
    #   k=6, i=0 -> 12
    #   k=6, i=1 -> 13
    
    for r in range(4):
        for i in range(4):
            U[i][col_offset + r] = SchU[i][r]
        
        V[6][col_offset + r] = SchV[0][r]   # B06
        V[13][col_offset + r] = SchV[1][r]  # B16
        
        W[12][col_offset + r] = SchW[0][r]  # C60
        W[13][col_offset + r] = SchW[1][r]  # C61
    col_offset += 4

    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return _build_schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
