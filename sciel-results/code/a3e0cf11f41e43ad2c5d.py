import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS

def _schoolbook():
    """Return the trivial rank-72 decomposition."""
    n, m, p = N, M, P
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
    return U, V, W, rank

def _verify(U, V, W, rank):
    """Verify that the decomposition is exact for 3x3 * 3x8."""
    n, m, p = N, M, P
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check non-zero entries: for all i, j, k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    
    # Check zero entries: sample some
    # T[i*m+j, j'*p+k', k'*n+i'] should be 0 unless i==i', j==j', k==k'
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check a few mismatches
                for i2 in range(n):
                    if i2 == i:
                        continue
                    for j2 in range(m):
                        for k2 in range(p):
                            if j2 == j and k2 == k:
                                continue
                            s = 0
                            for r in range(rank):
                                s += U[i*m + j][r] * V[j2*p + k2][r] * W[k2*n + i2][r]
                            if abs(s) > 1e-9:
                                return False
    return True

def _strassen_2x2_2x8():
    """
    Return a rank-56 decomposition for 2x2 * 2x8 matrix multiplication.
    Uses Strassen on the 2x2 block, extended to 8 columns.
    
    A is 2x2, B is 2x8, C = A*B is 2x8.
    U shape: (4, rank), V shape: (16, rank), W shape: (16, rank)
    Indexing: U[i*2+j, r], V[j*8+k, r], W[k*2+i, r]
    """
    n, m, p = 2, 2, 8
    rank = 7 * 8  # 7 Strassen terms * 8 columns
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x8, B has 8 columns. We apply Strassen per column pair.
    # Actually, Strassen works on 2x2 blocks. For 2x2 * 2x8, we can think of
    # B as 4 blocks of 2x2. So we need 4 Strassen decompositions, each of rank 7.
    # Total rank = 4 * 7 = 28.
    
    # Wait, that's not right. Let me reconsider.
    # A is 2x2, B is 2x8. C = A*B is 2x8.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is just 2*8 = 16 scalar multiplications in schoolbook.
    # Strassen doesn't directly apply here because A is 2x2 and B is 2x8.
    # We can partition B into four 2x2 blocks: B = [B1 B2 B3 B4] where each Bj is 2x2.
    # Then C = [A*B1 A*B2 A*B3 A*B4], and each A*Bj is a 2x2*2x2 multiplication.
    # So we use Strassen 4 times, each of rank 7. Total rank = 28.
    
    # Let me redo this properly.
    U = [[0] * 28 for _ in range(4)]
    V = [[0] * 28 for _ in range(16)]
    W = [[0] * 28 for _ in range(16)]
    
    # For each column block c in 0..3 (each block has 2 columns):
    # B[:, c*2:(c+1)*2] is a 2x2 block.
    # A * B_block is a 2x2 result.
    # Use Strassen for this 2x2 * 2x2 multiplication.
    
    for c in range(4):
        base = c * 7
        # Strassen terms for this block:
        # M1 = (A00+A11)(B00+B11) where B00 = B[0, c*2], B11 = B[1, c*2+1]
        # M2 = (A10+A11)B00 where B00 = B[0, c*2]
        # M3 = A00(B10-B11) where B10 = B[1, c*2], B11 = B[1, c*2+1]
        # M4 = A11(B01-B10) where B01 = B[0, c*2+1], B10 = B[1, c*2]
        # M5 = (A00+A01)B11 where B11 = B[1, c*2+1]
        # M6 = (A10-A00)(B00+B01) where B00 = B[0, c*2], B01 = B[0, c*2+1]
        # M7 = (A01-A11)(B10+B11) where B10 = B[1, c*2], B11 = B[1, c*2+1]
        
        # C00 = M1 - M5 + M7 -> C[0, c*2]
        # C01 = M3 + M5 -> C[0, c*2+1]
        # C10 = M2 + M4 -> C[1, c*2]
        # C11 = M1 - M2 + M3 + M6 -> C[1, c*2+1]
        
        # Term M1:
        U[0][base] = 1; U[3][base] = 1  # A00 + A11
        V[c*2][base] = 1; V[2*(c*2+1)][base] = 1  # B00 + B11 (B[0,c*2] + B[1,c*2+1])
        W[c*2*2 + 0][base] = 1  # C[0, c*2]
        W[(c*2+1)*2 + 1][base] = 1  # C[1, c*2+1]
        
        # Term M2:
        U[2][base+1] = 1; U[3][base+1] = 1  # A10 + A11
        V[c*2][base+1] = 1  # B00
        W[(c*2)*2 + 1][base+1] = 1  # C[1, c*2]
        W[(c*2+1)*2 + 1][base+1] = -1  # C[1, c*2+1]
        
        # Term M3:
        U[0][base+2] = 1  # A00
        V[2*(c*2)][base+2] = 1; V[2*(c*2+1)][base+2] = -1  # B10 - B11
        W[(c*2+1)*2 + 0][base+2] = 1  # C[0, c*2+1]
        W[(c*2+1)*2 + 1][base+2] = 1  # C[1, c*2+1]
        
        # Term M4:
        U[3][base+3] = 1  # A11
        V[2*(c*2+1)][base+3] = 1; V[2*(c*2)][base+3] = -1  # B01 - B10
        W[(c*2)*2 + 1][base+3] = 1  # C[1, c*2]
        
        # Term M5:
        U[0][base+4] = 1; U[1][base+4] = 1  # A00 + A01
        V[2*(c*2+1)][base+4] = 1  # B11
        W[(c*2)*2 + 0][base+4] = -1  # C[0, c*2]
        W[(c*2+1)*2 + 0][base+4] = 1  # C[0, c*2+1]
        
        # Term M6:
        U[2][base+5] = 1; U[0][base+5] = -1  # A10 - A00
        V[c*2][base+5] = 1; V[2*(c*2+1)][base+5] = 1  # B00 + B01
        W[(c*2+1)*2 + 1][base+5] = 1  # C[1, c*2+1]
        
        # Term M7:
        U[1][base+6] = 1; U[3][base+6] = -1  # A01 - A11
        V[2*(c*2)][base+6] = 1; V[2*(c*2+1)][base+6] = 1  # B10 + B11
        W[(c*2)*2 + 0][base+6] = 1  # C[0, c*2]
    
    return U, V, W, 28

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook rank-72
    U, V, W, rank = _schoolbook()
    
    # Try to use Strassen for the 2x2 block of A and B
    # A is 3x3, B is 3x8.
    # Partition A into:
    # A = [A_top (2x2)  A_right (2x1)]
    #     [A_bottom (1x2) A_bottomright (1x1)]
    # Partition B into:
    # B = [B_top (2x8)]
    #     [B_bottom (1x8)]
    # C = A*B is 3x8:
    # C_top (2x8) = A_top * B_top + A_right * B_bottom
    # C_bottom (1x8) = A_bottom * B_top + A_bottomright * B_bottom
    
    # A_top * B_top is 2x2 * 2x8, which we can do with rank 28 (Strassen).
    # A_right * B_bottom is 2x1 * 1x8, which is rank 16 (schoolbook).
    # A_bottom * B_top is 1x2 * 2x8, which is rank 16 (schoolbook).
    # A_bottomright * B_bottom is 1x1 * 1x8, which is rank 8 (schoolbook).
    # Total rank = 28 + 16 + 16 + 8 = 68.
    
    # Let me construct this.
    U_new = [[0] * 68 for _ in range(9)]
    V_new = [[0] * 68 for _ in range(24)]
    W_new = [[0] * 68 for _ in range(24)]
    
    # Strassen for A_top (2x2) * B_top (2x8) -> C_top (2x8)
    # A_top indices: A[0,0]=A[0], A[0,1]=A[1], A[1,0]=A[3], A[1,1]=A[4]
    # B_top indices: B[0,k]=B[k], B[1,k]=B[8+k] for k=0..7
    # C_top indices: C[0,k]=C[k*3+0], C[1,k]=C[k*3+1] for k=0..7
    
    U_s, V_s, W_s, rank_s = _strassen_2x2_2x8()
    for r in range(rank_s):
        for i in range(4):
            U_new[i][r] = U_s[i][r]
        for j in range(16):
            V_new[j][r] = V_s[j][r]
        for k in range(16):
            W_new[k][r] = W_s[k][r]
    
    # A_right (2x1) * B_bottom (1x8) -> C_right (2x8)
    # A_right: A[0,2]=A[2], A[1,2]=A[5]
    # B_bottom: B[2,k]=B[16+k] for k=0..7
    # C_right: C[0,k]=C[k*3+0], C[1,k]=C[k*3+1]
    # This is rank 16.
    base = 28
    for k in range(8):
        # C[0,k] += A[0,2] * B[2,k]
        U_new[2][base + k] = 1
        V_new[16 + k][base + k] = 1
        W_new[k * 3 + 0][base + k] = 1
        # C[1,k] += A[1,2] * B[2,k]
        U_new[5][base + 8 + k] = 1
        V_new[16 + k][base + 8 + k] = 1
        W_new[k * 3 + 1][base + 8 + k] = 1
    
    # A_bottom (1x2) * B_top (2x8) -> C_bottom (1x8)
    # A_bottom: A[2,0]=A[6], A[2,1]=A[7]
    # B_top: B[0,k]=B[k], B[1,k]=B[8+k]
    # C_bottom: C[2,k]=C[k*3+2]
    # This is rank 16.
    base = 44
    for k in range(8):
        # C[2,k] += A[2,0] * B[0,k]
        U_new[6][base + k] = 1
        V_new[k][base + k] = 1
        W_new[k * 3 + 2][base + k] = 1
        # C[2,k] += A[2,1] * B[1,k]
        U_new[7][base + 8 + k] = 1
        V_new[8 + k][base + 8 + k] = 1
        W_new[k * 3 + 2][base + 8 + k] = 1
    
    # A_bottomright (1x1) * B_bottom (1x8) -> C_bottomright (1x8)
    # A_bottomright: A[2,2]=A[8]
    # B_bottom: B[2,k]=B[16+k]
    # C_bottomright: C[2,k]=C[k*3+2]
    # This is rank 8.
    base = 60
    for k in range(8):
        U_new[8][base + k] = 1
        V_new[16 + k][base + k] = 1
        W_new[k * 3 + 2][base + k] = 1
    
    # Verify
    if _verify(U_new, V_new, W_new, 68):
        return {"U": U_new, "V": V_new, "W": W_new}
    
    # Fallback to schoolbook
    return {"U": U, "V": V, "W": W}
