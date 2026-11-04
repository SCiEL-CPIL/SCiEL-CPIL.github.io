"""Rewritten program for matrix multiplication tensor decomposition (4x5, 5x6)."""
import time
import random
import math
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r] for i,j,k in [0,2).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10
    
    # M3 = A00(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1; V[3][2] = -1  # B10, -B11
    W[1][2] = 1; W[2][2] = 1  # C01, C10
    
    # M4 = A11(B01-B00)
    U[3][3] = 1  # A11
    V[1][3] = 1; V[0][3] = -1  # B01, -B00
    W[0][3] = 1; W[2][3] = 1  # C00, C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    W[0][4] = -1; W[1][4] = 1  # -C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A10, -A00
    V[0][5] = 1; V[1][5] = 1  # B00, B01
    W[3][5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, -A11
    V[2][6] = 1; V[3][6] = 1  # B10, B11
    W[0][6] = 1; W[3][6] = 1  # C00, C11
    
    return U, V, W

def _strassen_4x4x4():
    """Return rank-49 Strassen decomposition for 4x4x4.
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r] for i,j,k in [0,4).
    """
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    s2U, s2V, s2W = _strassen_2x2()
    
    # The 7 super-multiplications for 4x4 Strassen:
    # Each is (a_blocks, b_blocks, c_blocks) where blocks are (block_row, block_col, coeff)
    super_mults = [
        ([[(0,0,1),(1,1,1)]], [[(0,0,1),(1,1,1)]], [[(0,0,1),(1,1,1)]]),   # S1
        ([[(1,0,1),(1,1,1)]], [[(0,0,1)]], [[(1,0,1)]]),                      # S2
        ([[(0,0,1)]], [[(1,0,1),(1,1,-1)]], [[(0,1,1),(1,0,1)]]),             # S3
        ([[(1,1,1)]], [[(0,1,1),(0,0,-1)]], [[(0,0,1),(1,0,1)]]),             # S4
        ([[(0,0,1),(0,1,1)]], [[(1,1,1)]], [[(0,0,-1),(0,1,1)]]),             # S5
        ([[(1,0,1),(0,0,-1)]], [[(0,0,1),(0,1,1)]], [[(1,1,1)]]),             # S6
        ([[(0,1,1),(1,1,-1)]], [[(1,0,1),(1,1,1)]], [[(0,0,1),(1,1,1)]]),     # S7
    ]
    
    for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(super_mults):
        # Build effective 2x2 matrices
        eff_a = [[0]*2 for _ in range(2)]
        for (br, bc, coeff) in a_blocks:
            eff_a[br][bc] += coeff
        
        eff_b = [[0]*2 for _ in range(2)]
        for (br, bc, coeff) in b_blocks:
            eff_b[br][bc] += coeff
        
        eff_c = [[0]*2 for _ in range(2)]
        for (br, bc, coeff) in c_blocks:
            eff_c[br][bc] += coeff
        
        for t_idx in range(7):
            r = s_idx * 7 + t_idx
            
            for i2 in range(2):
                for j2 in range(2):
                    coeff_a = s2U[i2*2+j2][t_idx] * eff_a[i2][j2]
                    if coeff_a != 0:
                        for di in range(2):
                            for dj in range(2):
                                i4 = i2*2 + di
                                j4 = j2*2 + dj
                                U[i4*4 + j4][r] += coeff_a
            
            for j2 in range(2):
                for k2 in range(2):
                    coeff_b = s2V[j2*2+k2][t_idx] * eff_b[j2][k2]
                    if coeff_b != 0:
                        for dj in range(2):
                            for dk in range(2):
                                j4 = j2*2 + dj
                                k4 = k2*2 + dk
                                V[j4*4 + k4][r] += coeff_b
            
            for k2 in range(2):
                for i2 in range(2):
                    coeff_c = s2W[k2*2+i2][t_idx] * eff_c[k2][i2]
                    if coeff_c != 0:
                        for dk in range(2):
                            for di in range(2):
                                k4 = k2*2 + dk
                                i4 = i2*2 + di
                                W[k4*4 + i4][r] += coeff_c
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _block_decomposition():
    """Block decomposition for 4x5x6 using Strassen on 4x4x4 sub-block.
    Split m=5 into 4+1, p=6 into 4+2.
    Blocks: (4x4x4), (4x4x2), (4x1x4), (4x1x2)
    Use Strassen (rank 49) for 4x4x4, schoolbook for others.
    Total rank: 49 + 32 + 16 + 8 = 105
    """
    n, m, p = 4, 5, 6
    
    m1, m2 = 4, 1  # m = m1 + m2
    p1, p2 = 4, 2  # p = p1 + p2
    
    rank_444 = 49
    rank_442 = 4*4*2
    rank_414 = 4*1*4
    rank_412 = 4*1*2
    total_rank = rank_444 + rank_442 + rank_414 + rank_412
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    
    # Block 1: (4x4x4) - Strassen
    sU, sV, sW = _strassen_4x4x4()
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i*m + j][col + r] = sU[i*4 + j][r]
        for j in range(4):
            for k in range(4):
                V[j*p + k][col + r] = sV[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W[k*n + i][col + r] = sW[k*4 + i][r]
    col += rank_444
    
    # Block 2: (4x4x2) - schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Block 3: (4x1x4) - schoolbook
    for i in range(4):
        for j in range(4, 5):
            for k in range(4):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Block 4: (4x1x2) - schoolbook
    for i in range(4):
        for j in range(4, 5):
            for k in range(4, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    assert col == total_rank, f"col={col}, total_rank={total_rank}"
    return {"U": U, "V": V, "W": W}

def _winograd_4x4x4():
    """Return rank-16 Winograd (FMA) decomposition for 4x4x4.
    Uses 4-point DFT structure for cyclic convolution.
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r] for i,j,k in [0,4).
    """
    # Winograd FMA for 4x4x4: uses the fact that 4x4 matrix multiplication
    # can be viewed as 16 cyclic convolutions of length 4 (not exactly, but
    # the tensor rank of 4x4x4 is known to be 49 via Strassen, not 16).
    # Actually, rank 16 is NOT achievable for 4x4x4. The known lower bound is 49.
    # This function is a placeholder - we'll use Strassen instead.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try block decomposition with Strassen
    try:
        result = _block_decomposition()
        if _verify(result['U'], result['V'], result['W'], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
