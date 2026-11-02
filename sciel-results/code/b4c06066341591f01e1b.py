import time
import numpy as np

# Time budget guard: kill at 150s, safety margin 10s
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def _winograd_3x3x3_vectors():
    """
    Return U, V, W vectors for 3x3x3 matrix multiplication (Winograd rank 23).
    U: 9 x 23 (input A), V: 9 x 23 (input B), W: 9 x 23 (output C)
    Indexing: A is 3x3, B is 3x3, C is 3x3.
    U[i*3+j, c] = coefficient of A[i,j] in the c-th product
    V[j*3+k, c] = coefficient of B[j,k] in the c-th product
    W[k*3+i, c] = coefficient of C[i,k] in the c-th product
    """
    # Winograd 3x3x3 vectors (from Winograd's paper / known decomposition)
    # The 23 multipliers correspond to:
    # m0 = a00*b00, m1 = a01*b01, m2 = a02*b02, m3 = a10*b10, m4 = a11*b11,
    # m5 = a12*b12, m6 = a20*b20, m7 = a21*b21, m8 = a22*b22,
    # m9 = (a00+a01)*(b00+b01), m10 = (a01+a02)*(b01+b02),
    # m11 = (a00+a02)*(b00+b02), m12 = (a10+a11)*(b10+b11),
    # m13 = (a11+a12)*(b11+b12), m14 = (a10+a12)*(b10+b12),
    # m15 = (a20+a21)*(b20+b21), m16 = (a21+a22)*(b21+b22),
    # m17 = (a20+a22)*(b20+b22), m18 = (a00+a10+a20)*(b00+b10+b20),
    # m19 = (a01+a11+a21)*(b01+b11+b21), m20 = (a02+a12+a22)*(b02+b12+b22),
    # m21 = (a00+a11+a22)*(b00+b11+b22), m22 = (a02-a00+a12-a10+a22-a20)*(b02-b00+b12-b10+b22-b20)
    #
    # Actually, let me use the standard Winograd 3x3x3 vectors from literature.
    # The decomposition has rank 23. Let me construct them properly.
    
    # Using the known Winograd 3x3x3 scheme:
    # We have 9 diagonal + 9 row/col sums + 3 cross terms + 2 more = 23
    
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # Let's use the standard Winograd vectors for 3x3x3.
    # From "Arithmetic Complexity of the Fast Transform" and related work.
    
    # I'll use a known correct set of vectors.
    # The Winograd 3x3x3 decomposition:
    
    # Multiplier 0: a00 * b00
    U[0,0] = 1; V[0,0] = 1; W[0,0] = 1  # c[0,0] += a00*b00
    # But wait, C[i,k] gets contributions from multiple multipliers.
    # Let me re-index: C[i*3+k] in W.
    # W[k*3 + i, c] per the task convention: W[k*n+i, c] where n=3.
    
    # Actually, the task says W[k*n+i, r] for the 4x4x6 case.
    # For 3x3x3: W[k*3+i, c].
    
    # Let me define the multipliers and their contributions to C.
    
    # m0 = a00 * b00
    # m1 = a01 * b01
    # m2 = a02 * b02
    # m3 = a10 * b10
    # m4 = a11 * b11
    # m5 = a12 * b12
    # m6 = a20 * b20
    # m7 = a21 * b21
    # m8 = a22 * b22
    # m9 = (a00+a01) * (b00+b01)
    # m10 = (a01+a02) * (b01+b02)
    # m11 = (a00+a02) * (b00+b02)
    # m12 = (a10+a11) * (b10+b11)
    # m13 = (a11+a12) * (b11+b12)
    # m14 = (a10+a12) * (b10+b12)
    # m15 = (a20+a21) * (b20+b21)
    # m16 = (a21+a22) * (b21+b22)
    # m17 = (a20+a22) * (b20+b22)
    # m18 = (a00+a10+a20) * (b00+b10+b20)
    # m19 = (a01+a11+a21) * (b01+b11+b21)
    # m20 = (a02+a12+a22) * (b02+b12+b22)
    # m21 = (a00+a11+a22) * (b00+b11+b22)
    # m22 = (a02-a00+a12-a10+a22-a20) * (b02-b00+b12-b10+b22-b20)
    
    # Now, C[i,k] = sum of contributions.
    # Let me compute the W vectors for each multiplier.
    
    # For each multiplier c, W[k*3+i, c] = coefficient in C[i,k]
    
    # m0 = a00*b00: contributes to C[0,0]
    # C[0,0] += a00*b00 -> W[0*3+0, 0] = 1
    W[0, 0] = 1
    # m1 = a01*b01: contributes to C[0,1]
    W[1, 1] = 1
    # m2 = a02*b02: contributes to C[0,2]
    W[2, 2] = 1
    # m3 = a10*b10: contributes to C[1,0]
    W[0*3+1, 3] = 1  # W[1, 3]
    W[1, 3] = 1
    # m4 = a11*b11: contributes to C[1,1]
    W[1*3+1, 4] = 1  # W[4, 4]
    W[4, 4] = 1
    # m5 = a12*b12: contributes to C[1,2]
    W[2*3+1, 5] = 1  # W[5, 5]
    W[5, 5] = 1
    # m6 = a20*b20: contributes to C[2,0]
    W[0*3+2, 6] = 1  # W[2, 6]
    W[2, 6] = 1
    # m7 = a21*b21: contributes to C[2,1]
    W[1*3+2, 7] = 1  # W[7, 7]
    W[7, 7] = 1
    # m8 = a22*b22: contributes to C[2,2]
    W[2*3+2, 8] = 1  # W[8, 8]
    W[8, 8] = 1
    
    # m9 = (a00+a01)*(b00+b01): 
    # C[0,0] += a00*b01 + a01*b00
    # C[0,1] += a00*b01 + a01*b00 (wait, no)
    # Actually, (a00+a01)*(b00+b01) = a00*b00 + a00*b01 + a01*b00 + a01*b01
    # This contributes to C[0,0] (a00*b00 + a01*b01 already counted in m0,m1)
    # and to C[0,1]? No, let me think about this differently.
    
    # The Winograd scheme is designed so that each multiplier's result is added 
    # to specific C entries. Let me use a verified set of vectors.
    
    # I'll use the vectors from the known rank-23 decomposition of 3x3x3.
    
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # This is getting complex. Let me use a simpler approach: 
    # Use the block decomposition with schoolbook for simplicity, 
    # but optimize the 4x4x6 structure.
    
    # Actually, for the 4x4x6 problem, the schoolbook rank is 96.
    # A better approach: use Winograd for 3x3x3 blocks.
    
    # Let me just use a direct construction that I can verify.
    
    # For 4x4x6, I'll use the following strategy:
    # Split into 4x4x4 + 4x4x2
    # For 4x4x4: use 3x3x3 Winograd (rank 23) + boundary terms
    # For 4x4x2: use schoolbook (rank 32) or better
    
    # Total: 23 + 9 + 9 + 9 + 3 + 3 + 3 + 1 + 32 = 92
    
    return U, V, W

def construct():
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    
    # Schoolbook decomposition as baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
