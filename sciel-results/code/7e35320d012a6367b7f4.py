"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block-structured decomposition:
    # 6 = 3 * 2, so partition into 3x3 blocks of 2x2 submatrices.
    # Outer: 3x3x3 matrix multiplication (rank 23, Bini et al.)
    # Inner: 2x2x2 matrix multiplication (rank 7, Strassen)
    # Total rank: 23 * 7 = 161
    
    # Strassen decomposition for 2x2x2 (rank 7):
    # Indices: U row = i*2+j, V row = j*2+k, W row = k*2+i, for i,j,k in {0,1}
    # The 7 Strassen factors:
    strassen_U = [
        # r=0: (A00+A11)
        [1, 0, 0, 1],
        # r=1: (A10+A11)
        [0, 0, 1, 1],
        # r=2: A00
        [1, 0, 0, 0],
        # r=3: A11
        [0, 0, 0, 1],
        # r=4: (A00+A01)
        [1, 1, 0, 0],
        # r=5: (A10-A00)
        [-1, 0, 1, 0],
        # r=6: (A01-A11)
        [0, 1, 0, -1],
    ]
    strassen_V = [
        # r=0: (B00+B11)
        [1, 0, 0, 1],
        # r=1: B00
        [1, 0, 0, 0],
        # r=2: (B10-B11)
        [0, 0, 1, -1],
        # r=3: (B01-B00)
        [1, -1, 0, 0],
        # r=4: B11
        [0, 0, 0, 1],
        # r=5: (B00+B01)
        [1, 1, 0, 0],
        # r=6: (B10+B11)
        [0, 0, 1, 1],
    ]
    strassen_W = [
        # r=0: C00 += 1, C11 += 1
        [1, 0, 0, 1],
        # r=1: C10 += 1
        [0, 0, 1, 0],
        # r=2: C00 += 1
        [1, 0, 0, 0],
        # r=3: C11 += 1
        [0, 0, 0, 1],
        # r=4: C00 -= 1, C11 -= 1  [Wait, let me recheck]
        # M5 = (A00+A01)B11, C00 = M1+M4-M5+M7, C12 = M3+M5
        # So M5 contributes -1 to C00 and +1 to C12
        # C12 is k=1, i=0 -> W[1*2+0] = W[2]
        [0, 0, 1, -1],  # C12 += 1, C00 -= 1
        # r=5: M6 = (A10-A00)(B00+B01), C22 = M1-M2+M3+M6
        # C22 is k=1, i=1 -> W[1*2+1] = W[3]
        [0, 0, 0, 1],  # C22 += 1
        # r=6: M7 = (A01-A11)(B10+B11), C00 = M1+M4-M5+M7
        # C00 is k=0, i=0 -> W[0*2+0] = W[0]
        [1, 0, 0, 0],  # C00 += 1
    ]
    
    # Verify Strassen decomposition
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += strassen_U[i*2+j][r] * strassen_V[j*2+k][r] * strassen_W[k*2+i][r]
                    if s != 1:
                        return False
        return True
    
    # Bini decomposition for 3x3x3 (rank 23):
    # This is complex. Let me use a simpler approach: use the schoolbook for the outer
    # but that gives 27*7 = 189, still better than 216.
    # Or use a known rank-23 decomposition.
    
    # For now, let's use the 3x3x3 schoolbook (rank 27) composed with Strassen (rank 7):
    # Total rank = 27 * 7 = 189
    
    outer_rank = 9  # 3*3*3 schoolbook
    inner_rank = 7  # Strassen
    
    total_rank = outer_rank * inner_rank  # 189
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block indices: for 6x6 with 3x3 blocks of 2x2:
    # A[i,j] where i = bi*2 + si, j = bj*2 + sj, bi,bj in {0,1,2}, si,sj in {0,1}
    # U row index: i*m + j = (bi*2+si)*6 + (bj*2+sj)
    # V row index: j*p + k = (bj*2+sj)*6 + (bk*2+sk)
    # W row index: k*n + i = (bk*2+sk)*6 + (bi*2+si)
    
    # For each outer factor r_out (0..8), inner factor r_in (0..6):
    # r = r_out * 7 + r_in
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                r_out = bi * 3 + bj  # outer factor index for (bi,bj,bk)
                # Wait, the outer tensor factor for 3x3x3 schoolbook:
                # U_outer[bi*3+bj, r_out], V_outer[bj*3+bk, r_out], W_outer[bk*3+bi, r_out]
                # For schoolbook, r_out = bi*3+bj*3+bk... no, r_out = bi*3*3 + bj*3 + bk = bi*9+bj*3+bk
                # Actually for schoolbook 3x3x3: rank = 27, indexed by (bi,bj,bk)
                r_out = bi * 9 + bj * 3 + bk
                
                for r_in in range(7):
                    r = r_out * 7 + r_in
                    
                    # U contribution: for each (si, sj) in {0,1}^2:
                    for si in range(2):
                        for sj in range(2):
                            i = bi * 2 + si
                            j = bj * 2 + sj
                            U[i * m + j][r] = strassen_U[si * 2 + sj][r_in]
                    
                    # V contribution: for each (sj, sk) in {0,1}^2:
                    for sj in range(2):
                        for sk in range(2):
                            j = bj * 2 + sj
                            k = bk * 2 + sk
                            V[j * p + k][r] = strassen_V[sj * 2 + sk][r_in]
                    
                    # W contribution: for each (sk, si) in {0,1}^2:
                    for sk in range(2):
                        for si in range(2):
                            k = bk * 2 + sk
                            i = bi * 2 + si
                            W[k * n + i][r] = strassen_W[sk * 2 + si][r_in]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
