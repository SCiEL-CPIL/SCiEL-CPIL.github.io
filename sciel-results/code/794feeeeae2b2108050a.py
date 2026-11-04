import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, max_samples=50, tol=1e-10):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for _ in range(max_samples):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            C_decomp += (u @ v) * w
        if not np.allclose(C_decomp, C_expected, atol=tol):
            return False
    return True

def _strassen_2x2x2(U, V, W):
    """Add 7 Strassen terms for 2x2x2 multiplication starting at column offset."""
    # Returns the rank used (7)
    return 7

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try block Strassen approach
    # Split 4x4x7 into four 2x2x7 blocks
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B is 4x7, split into B_top (2x7) and B_bot (2x7)
    # C = [[A11*B_top + A12*B_bot, ...], ...]
    
    # For each 2x2x7 multiplication, we can use Strassen for the 2x2x2 parts
    # and schoolbook for the 2x2x1 part
    # 7 = 2 + 2 + 2 + 1
    # So 2x2x7 = 3*(2x2x2) + 1*(2x2x1)
    # 2x2x2 Strassen rank 7, 2x2x1 schoolbook rank 4
    # Total per 2x2x7: 3*7 + 4 = 25
    # Total for 4x4x7: 4 * 25 = 100
    
    def build_2x2x7_decomposition():
        """Build a 2x2x7 decomposition using 3 Strassen + 1 schoolbook."""
        rank = 25
        U = [[0] * rank for _ in range(4)]  # 2*2 = 4
        V = [[0] * rank for _ in range(14)]  # 2*7 = 14
        W = [[0] * rank for _ in range(14)]  # 7*2 = 14
        
        col = 0
        # Strassen for columns 0-1 (B[:, 0:2])
        # M1 = (A00+A11)(B00+B11) -> W00 += M1, W01 += M1
        # M2 = A11*(B01-B11) -> W01 += M2
        # M3 = (A00-A11)*B11 -> W11 += M3
        # M4 = A00*(B10-B00) -> W10 += M4
        # M5 = (A00+A10)*B00 -> W00 += M5
        # M6 = (A11+A10)*(B10-B00) -> W10 += M6
        # M7 = (A00+A11)*(B01-B11) -> W01 += M7
        
        # U indices: A00=0, A01=1, A10=2, A11=3
        # V indices: B00=0, B01=1, B10=4, B11=5 (for cols 0,1)
        # W indices: W00=0, W01=1, W10=2, W11=3 (for cols 0,1)
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col] += 1; U[3][col] += 1
        V[0][col] += 1; V[5][col] += 1
        W[0][col] += 1; W[1][col] += 1
        col += 1
        
        # M2 = A11*(B01-B11)
        U[3][col] += 1
        V[1][col] += 1; V[5][col] -= 1
        W[1][col] += 1
        col += 1
        
        # M3 = (A00-A11)*B11
        U[0][col] += 1; U[3][col] -= 1
        V[5][col] += 1
        W[3][col] += 1
        col += 1
        
        # M4 = A00*(B10-B00)
        U[0][col] += 1
        V[4][col] += 1; V[0][col] -= 1
        W[2][col] += 1
        col += 1
        
        # M5 = (A00+A10)*B00
        U[0][col] += 1; U[2][col] += 1
        V[0][col] += 1
        W[0][col] += 1
        col += 1
        
        # M6 = (A11+A10)*(B10-B00)
        U[3][col] += 1; U[2][col] += 1
        V[4][col] += 1; V[0][col] -= 1
        W[2][col] += 1
        col += 1
        
        # M7 = (A00+A11)*(B01-B11)
        U[0][col] += 1; U[3][col] += 1
        V[1][col] += 1; V[5][col] -= 1
        W[1][col] += 1
        col += 1
        
        # Strassen for columns 2-3 (B[:, 2:4])
        # V indices: B00=2, B01=3, B10=6, B11=7
        # W indices: W00=4, W01=5, W10=6, W11=7
        
        # M1 = (A00+A11)(B02+B12)
        U[0][col] += 1; U[3][col] += 1
        V[2][col] += 1; V[7][col] += 1
        W[4][col] += 1; W[5][col] += 1
        col += 1
        
        # M2 = A11*(B03-B13)
        U[3][col] += 1
        V[3][col] += 1; V[7][col] -= 1
        W[5][col] += 1
        col += 1
        
        # M3 = (A00-A11)*B12
        U[0][col] += 1; U[3][col] -= 1
        V[7][col] += 1
        W[7][col] += 1
        col += 1
        
        # M4 = A00*(B12-B02)
        U[0][col] += 1
        V[6][col] += 1; V[2][col] -= 1
        W[6][col] += 1
        col += 1
        
        # M5 = (A00+A10)*B02
        U[0][col] += 1; U[2][col] += 1
        V[2][col] += 1
        W[4][col] += 1
        col += 1
        
        # M6 = (A11+A10)*(B12-B02)
        U[3][col] += 1; U[2][col] += 1
        V[6][col] += 1; V[2][col] -= 1
        W[6][col] += 1
        col += 1
        
        # M7 = (A00+A11)*(B03-B13)
        U[0][col] += 1; U[3][col] += 1
        V[3][col] += 1; V[7][col] -= 1
        W[5][col] += 1
        col += 1
        
        # Strassen for columns 4-5 (B[:, 4:6])
        # V indices: B00=4, B01=5, B10=8, B11=9
        # W indices: W00=8, W01=9, W10=10, W11=11
        
        # M1 = (A00+A11)(B04+B14)
        U[0][col] += 1; U[3][col] += 1
        V[4][col] += 1; V[9][col] += 1
        W[8][col] += 1; W[9][col] += 1
        col += 1
        
        # M2 = A11*(B05-B15)
        U[3][col] += 1
        V[5][col] += 1; V[9][col] -= 1
        W[9][col] += 1
        col += 1
        
        # M3 = (A00-A11)*B14
        U[0][col] += 1; U[3][col] -= 1
        V[9][col] += 1
        W[11][col] += 1
        col += 1
        
        # M4 = A00*(B14-B04)
        U[0][col] += 1
        V[8][col] += 1; V[4][col] -= 1
        W[10][col] += 1
        col += 1
        
        # M5 = (A00+A10)*B04
        U[0][col] += 1; U[2][col] += 1
        V[4][col] += 1
        W[8][col] += 1
        col += 1
        
        # M6 = (A11+A10)*(B14-B04)
        U[3][col] += 1; U[2][col] += 1
        V[8][col] += 1; V[4][col] -= 1
        W[10][col] += 1
        col += 1
        
        # M7 = (A00+A11)*(B05-B15)
        U[0][col] += 1; U[3][col] += 1
        V[5][col] += 1; V[9][col] -= 1
        W[9][col] += 1
        col += 1
        
        # Schoolbook for column 6 (B[:, 6])
        # V indices: B00=6, B10=10
        # W indices: W00=12, W10=14
        
        # A00*B06
        U[0][col] += 1
        V[6][col] += 1
        W[12][col] += 1
        col += 1
        
        # A01*B16
        U[1][col] += 1
        V[10][col] += 1
        W[13][col] += 1
        col += 1
        
        # A10*B06
        U[2][col] += 1
        V[6][col] += 1
        W[14][col] += 1
        col += 1
        
        # A11*B16
        U[3][col] += 1
        V[10][col] += 1
        W[15][col] += 1
        col += 1
        
        return U, V, W, rank
    
    # Build the full 4x4x7 decomposition
    def build_4x4x7_decomposition():
        """Build a 4x4x7 decomposition using block Strassen."""
        rank = 100
        U = [[0] * rank for _ in range(16)]  # 4*4 = 16
        V = [[0] * rank for _ in range(28)]  # 4*7 = 28
        W = [[0] * rank for _ in range(28)]  # 7*4 = 28
        
        col = 0
        # Top-left 2x2x7: A11 * B_top
        U11, V11, W11, r11 = build_2x2x7_decomposition()
        # Map U11 indices 0,1,2,3 to U indices 0,1,4,5 (A11 is rows 0,1 cols 0,1)
        # Map V11 indices 0-13 to V indices 0-13 (B_top is rows 0,1 cols 0-6)
        # Map W11 indices 0-13 to W indices 0-13 (C_top is rows 0,1 cols 0-6)
        for i in range(4):
            for j in range(r11):
                U[i * 4 + (i % 2)][col + j] += U11[i][j]
        for i in range(14):
            for j in range(r11):
                V[i][col + j] += V11[i][j]
        for i in range(14):
            for j in range(r11):
                W[i][col + j] += W11[i][j]
        col += r11
        
        # Top-right 2x2x7: A12 * B_bot
        U12, V12, W12, r12 = build_2x2x7_decomposition()
        # Map U12 indices 0,1,2,3 to U indices 2,3,6,7 (A12 is rows 0,1 cols 2,3)
        # Map V12 indices 0-13 to V indices 14-27 (B_bot is rows 2,3 cols 0-6)
        # Map W12 indices 0-13 to W indices 0-13 (C_top is rows 0,1 cols 0-6)
        for i in range(4):
            for j in range(r12):
                U[i * 4 + 2 + (i % 2)][col + j] += U12[i][j]
        for i in range(14):
            for j in range(r12):
                V[14 + i][col + j] += V12[i][j]
        for i in range(14):
            for j in range(r12):
                W[i][col + j] += W12[i][j]
        col += r12
        
        # Bottom-left 2x2x7: A21 * B_top
        U21, V21, W21, r21 = build_2x2x7_decomposition()
        # Map U21 indices 0,1,2,3 to U indices 8,9,12,13 (A21 is rows 2,3 cols 0,1)
        # Map V21 indices 0-13 to V indices 0-13 (B_top is rows 0,1 cols 0-6)
        # Map W21 indices 0-13 to W indices 14-27 (C_bot is rows 2,3 cols 0-6)
        for i in range(4):
            for j in range(r21):
                U[8 + i * 4 + (i % 2)][col + j] += U21[i][j]
        for i in range(14):
            for j in range(r21):
                V[i][col + j] += V21[i][j]
        for i in range(14):
            for j in range(r21):
                W[14 + i][col + j] += W21[i][j]
        col += r21
        
        # Bottom-right 2x2x7: A22 * B_bot
        U22, V22, W22, r22 = build_2x2x7_decomposition()
        # Map U22 indices 0,1,2,3 to U indices 10,11,14,15 (A22 is rows 2,3 cols 2,3)
        # Map V22 indices 0-13 to V indices 14-27 (B_bot is rows 2,3 cols 0-6)
        # Map W22 indices 0-13 to W indices 14-27 (C_bot is rows 2,3 cols 0-6)
        for i in range(4):
            for j in range(r22):
                U[8 + i * 4 + 2 + (i % 2)][col + j] += U22[i][j]
        for i in range(14):
            for j in range(r22):
                V[14 + i][col + j] += V22[i][j]
        for i in range(14):
            for j in range(r22):
                W[14 + i][col + j] += W22[i][j]
        col += r22
        
        return U, V, W, rank
    
    # Try the block Strassen approach
    try:
        if time.time() < deadline - 10:
            U_new, V_new, W_new, rank_new = build_4x4x7_decomposition()
            # Verify
            if time.time() < deadline - 5:
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, max_samples=20):
                    return {"U": U_new, "V": V_new, "W": W_new}
    except Exception:
        pass
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
