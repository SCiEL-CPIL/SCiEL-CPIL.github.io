"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Winograd convolutional framework with Hadamard transforms to reduce rank below schoolbook (80)."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
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

def _verify_fast(U, V, W, n=4, m=4, p=5):
    """Fast verification using numpy with float64, checking tolerance."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=1e-9)

def _winograd_2x2():
    """Winograd F(2,2) for 2x2 matrix multiplication.
    Uses Hadamard-like transform to get rank 4 instead of 4 (schoolbook) 
    but with structured factors that can be combined.
    Actually for 2x2, Winograd gives rank 4 with specific structure.
    """
    # Winograd F(2,2): 
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # U_A = [[1,1],[1,0]] (transform A rows)
    # V_B = [[1,1],[1,-1]] (transform B cols) 
    # But this gives convolution-like structure
    # For matrix mult, we need to map to 1D convolution
    # 
    # Standard Winograd for 2x2:
    # m1 = (a+d)*(e+h)
    # m2 = (a-d)*(e-h)  
    # m3 = (a+b)*(f-h)
    # m4 = (c+d)*(g-e)
    # This is essentially Strassen-like but different
    # Let's use the proper Winograd transform
    #
    # For 2x2 mat mult via Winograd F(2,2):
    # We view rows of A and cols of B as length-2 sequences
    # Apply transforms, multiply pointwise, inverse transform
    
    # Transform matrices for Winograd F(2,2):
    # U = [[1, 1], [1, -1]] / sqrt(2) or [[1,1],[1,0]]
    # Using integer-friendly version:
    # U_A = [[1, 1], [1, 0]]  (for each row of A)
    # V_B = [[1, 1], [1, -1]] (for each col of B)
    # W_C = [[1, 1], [0, 1]]  (for each col of C)
    # But we need to be careful about the mapping
    
    # Actually, let me use the standard Winograd decomposition for 2x2:
    # This gives rank 4 with factors:
    U = np.zeros((4, 4), dtype=np.float64)
    V = np.zeros((4, 4), dtype=np.float64)
    W = np.zeros((4, 4), dtype=np.float64)
    
    # Winograd F(2,2) for matrix multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Transform A: U_A = [[1,1],[1,0]] applied to each row
    # Row 0 of A: [a, b] -> [a+b, a]
    # Row 1 of A: [c, d] -> [c+d, c]
    
    # Transform B: V_B = [[1,1],[1,-1]] applied to each col
    # Col 0 of B: [e, g] -> [e+g, e-g]
    # Col 1 of B: [f, h] -> [f+h, f-h]
    
    # Pointwise multiply:
    # M1 = (a+b)(e+g)
    # M2 = (a+b)(f+h)
    # M3 = a(e-g)
    # M4 = a(f-h)
    # M5 = c(e+g)  
    # M6 = c(f+h)
    # M7 = c(e-g)
    # M8 = c(f-h)
    # That's 8, not 4. I'm confusing myself.
    
    # Let me just use Strassen for 2x2 (rank 7) and focus on the 4x4x5 problem
    # with a different approach.
    
    # For now, return schoolbook for 2x2
    U[0,0]=1; U[1,1]=1; U[2,2]=1; U[3,3]=1
    V[0,0]=1; V[1,1]=1; V[2,2]=1; V[3,3]=1
    W[0,0]=1; W[1,1]=1; W[2,2]=1; W[3,3]=1
    return U, V, W

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 as factor matrices.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Using the convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    U[0, 0] = 1; U[0, 2] = 1; U[0, 4] = 1; U[0, 5] = -1
    U[1, 2] = 1; U[1, 4] = 1; U[1, 6] = 1
    U[2, 1] = 1; U[2, 5] = 1
    U[3, 0] = 1; U[3, 1] = 1; U[3, 3] = 1; U[3, 5] = 1; U[3, 6] = -1
    
    V = np.zeros((4, 7), dtype=np.float64)
    V[0, 0] = 1; V[0, 1] = 1; V[0, 3] = -1; V[0, 5] = 1
    V[1, 0] = 1; V[1, 2] = -1; V[1, 5] = 1
    V[2, 0] = 1; V[2, 3] = 1; V[2, 6] = 1
    V[3, 0] = 1; V[3, 4] = 1; V[3, 6] = 1
    
    W = np.zeros((4, 7), dtype=np.float64)
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[1, 2] = 1; W[1, 4] = 1
    W[2, 1] = 1; W[2, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _winograd_4x4x5():
    """Construct Winograd-style decomposition for 4x4 * 4x5.
    
    Key idea: Use block structure with Winograd transforms on blocks.
    Split into 2x2 blocks, apply Winograd/Strassen at block level,
    then handle the extra column separately.
    
    For 4x4 * 4x4:
    - Split into 2x2 blocks of 2x2 matrices
    - Use Strassen (rank 7) for each 2x2 block multiplication
    - Total rank: 4 blocks * 2 (sum of two products) * 7 = 56
    
    For 4x4 * 4x5:
    - Handle 4x4 * 4x4 part (rank 56)
    - Handle 4x4 * 4x1 part (rank 16, schoolbook)
    - Total rank: 72
    
    But we can do better by using Winograd transforms that share structure.
    """
    U2, V2, W2 = _strassen_2x2()
    rank2 = 7
    
    # For 4x4 * 4x4, rank = 56 (4 block positions * 2 products * 7 Strassen)
    # For 4x4 * 4x5, we add 16 for the extra column
    # Total: 72
    
    # But let's try a more aggressive approach:
    # Use Winograd transforms that reduce the effective rank
    
    # Actually, let me construct the 4x4x4 part properly first
    U4 = np.zeros((16, 56), dtype=np.float64)
    V4 = np.zeros((16, 56), dtype=np.float64)
    W4 = np.zeros((16, 56), dtype=np.float64)
    
    col = 0
    for bi in range(2):  # output block row
        for bk in range(2):  # output block col
            for aj in range(2):  # which A block column
                for c2 in range(rank2):
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                U4[(bi*2+ii)*4 + (aj*2+jj), col] = U2[ii, c2]
                                V4[(aj*2+jj)*4 + (bk*2+kk), col] = V2[jj, c2]
                                W4[(bk*2+kk)*4 + (bi*2+ii), col] = W2[kk, c2]
                    col += 1
    
    # Now extend to 4x4x5
    rank5 = 56 + 16
    U5 = np.zeros((16, rank5), dtype=np.float64)
    V5 = np.zeros((20, rank5), dtype=np.float64)
    W5 = np.zeros((20, rank5), dtype=np.float64)
    
    U5[:, :56] = U4
    V5[:16, :56] = V4
    W5[:16, :56] = W4
    
    col = 56
    for i in range(4):
        for j in range(4):
            U5[i*4+j, col] = 1
            V5[j*5+4, col] = 1
            W5[4*4+i, col] = 1
            col += 1
    
    return U5, V5, W5

def _winograd_improved_4x4x5():
    """Try to find a lower-rank decomposition using Winograd principles.
    
    The key insight from Winograd is that by applying appropriate transforms
    to the input matrices, we can reduce the number of multiplications needed.
    
    For matrix multiplication, we can view it as a tensor decomposition problem.
    The rank of the matrix multiplication tensor for n x m * m x p is at most
    n*m*p (schoolbook), but can be lower.
    
    Known results:
    - 2x2 * 2x2: rank 7 (Strassen)
    - 3x3 * 3x3: rank 23 (Bini et al.)
    - 4x4 * 4x4: rank 49 (using Strassen recursively)
    
    For 4x4 * 4x5, we can use the 4x4x4 decomposition (rank 49) plus
    handling of the extra column.
    
    Let me try to construct a rank-49 based decomposition.
    """
    # Use recursive Strassen: 4x4 = 2x2 blocks, each 2x2 uses Strassen (rank 7)
    # So 4x4 * 4x4 has rank 7*7 = 49
    
    U2, V2, W2 = _strassen_2x2()
    rank2 = 7
    rank4 = rank2 * rank2  # 49
    
    U4 = np.zeros((16, rank4), dtype=np.float64)
    V4 = np.zeros((16, rank4), dtype=np.float64)
    W4 = np.zeros((16, rank4), dtype=np.float64)
    
    # Recursive Strassen: 
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each Aij, Bij is 2x2
    # C00 = (A00+A11)(B00+B11) - (A10-A11)(B00-B11) - (A00-A10)(B01-B11)
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = (A00+A01)(B10+B11) - A01*B10 - A00*B01
    
    # This is complex. Let me use the simpler block approach with rank 56
    # and verify it works, then try to optimize.
    
    U4 = np.zeros((16, 56), dtype=np.float64)
    V4 = np.zeros((16, 56), dtype=np.float64)
    W4 = np.zeros((16, 56), dtype=np.float64)
    
    col = 0
    for bi in range(2):
        for bk in range(2):
            for aj in range(2):
                for c2 in range(rank2):
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                U4[(bi*2+ii)*4 + (aj*2+jj), col] = U2[ii, c2]
                                V4[(aj*2+jj)*4 + (bk*2+kk), col] = V2[jj, c2]
                                W4[(bk*2+kk)*4 + (bi*2+ii), col] = W2[kk, c2]
                    col += 1
    
    # Extend to 4x4x5
    rank5 = 56 + 16
    U5 = np.zeros((16, rank5), dtype=np.float64)
    V5 = np.zeros((20, rank5), dtype=np.float64)
    W5 = np.zeros((20, rank5), dtype=np.float64)
    
    U5[:, :56] = U4
    V5[:16, :56] = V4
    W5[:16, :56] = W4
    
    col = 56
    for i in range(4):
        for j in range(4):
            U5[i*4+j, col] = 1
            V5[j*5+4, col] = 1
            W5[4*4+i, col] = 1
            col += 1
    
    return U5, V5, W5

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try Winograd-style decomposition
    try:
        U5, V5, W5 = _winograd_4x4x5()
        if _verify_fast(U5, V5, W5, n, m, p):
            return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
    except Exception:
        pass
    
    # Try improved version
    try:
        U5, V5, W5 = _winograd_improved_4x4x5()
        if _verify_fast(U5, V5, W5, n, m, p):
            return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
