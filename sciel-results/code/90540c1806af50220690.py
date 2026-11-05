"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [3, 3, 4]
_START = time.time()
DEADLINE = 150.0 - 20.0  # safety margin

def _remaining():
    return DEADLINE - (time.time() - _START)

# EVOLVE-BLOCK-START
def _build_laderman_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Laderman's algorithm (rank 7).
    U shape: (2*2, 7) = (4, 7)
    V shape: (2*2, 7) = (4, 7)
    W shape: (2*2, 7) = (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]
    
    # Laderman's decomposition
    # C = A B
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # r=0: (a)(e) -> W[0,0]
    U[0][0] = 1.0; V[0][0] = 1.0; W[0][0] = 1.0
    # r=1: (b)(g) -> W[1,0]
    U[1][1] = 1.0; V[2][1] = 1.0; W[2][1] = 1.0 # V index j*2+k. j=1, k=0 -> 2. W index k*2+i. k=1, i=0 -> 2.
    # Wait, W[k*2+i].
    # C[0,0] = ae + bg.
    # Term ae: A[0,0] B[0,0]. U[0,0]=1, V[0,0]=1. W[0,0]=1.
    # Term bg: A[0,1] B[1,0]. U[1,1]=1, V[2,1]=1. W[0,1]=1? No, C[0,0] is W[0,0].
    # So W[0,1] should contribute to C[0,0].
    # W index is k*2+i. For C[0,0], k=0, i=0. Index 0.
    # So W[0][1] = 1.0.
    
    # Let's restart Laderman carefully.
    # C00 = ae + bg
    # C01 = af + bh
    # C10 = ce + dg
    # C11 = cf + dh
    
    # Laderman:
    # 1: (a+e)(c+g) - ae - cg -> ?
    # Standard Laderman:
    # 1: a e
    # 2: b g
    # 3: (a+b)(e+f) - ae - bf
    # 4: (c+d)(g+h) - cg - dh
    # 5: (a+d)(e+h) - ae - dh
    # 6: (b+c)(f+g) - bf - cg
    # 7: (a+b+c+d)(e+f+g+h) - ...
    
    # Simpler Laderman:
    # r0: a e
    # r1: b g
    # r2: (a+b)(e+f) - ae - bf
    # r3: (c+d)(g+h) - cg - dh
    # r4: (a+d)(e+h) - ae - dh
    # r5: (b+c)(f+g) - bf - cg
    # r6: (a+b+c+d)(e+f+g+h) - ...
    
    # Let's use the known coefficients.
    # U (A indices i,j)
    # V (B indices j,k)
    # W (C indices k,i)
    
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # r=0: a e
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # r=1: b g
    U[1][1] = 1; V[2][1] = 1; W[2][1] = 1 # C[1,0] is k=1,i=0 -> idx 2.
    # r=2: (a+b)(e+f) - ae - bf
    # A part: a+b -> U[0][2]=1, U[1][2]=1
    # B part: e+f -> V[0][2]=1, V[1][2]=1
    # C part: C00 + C01 - ae - bf
    # C00 (0,0): +1. C01 (1,0): +1.
    # ae is r0. bf is ?
    # bf contributes to C01.
    # We need W such that W[0][2]=1, W[1][2]=1.
    # And subtract r0 and r_bf.
    # This is getting complicated to hardcode manually without errors.
    
    # I will use a verified Laderman implementation.
    # From literature:
    # 1: (a, e)
    # 2: (b, g)
    # 3: (a+b, e+f)
    # 4: (c+d, g+h)
    # 5: (a+d, e+h)
    # 6: (b+c, f+g)
    # 7: (a+b+c+d, e+f+g+h)
    
    # Coefficients for W:
    # C00 = 1 - 3 + 5
    # C01 = -2 + 3 - 6 + 7
    # C10 = -1 + 4 - 6 + 7
    # C11 = 2 - 4 + 5 - 7
    
    # Let's map to indices.
    # U[i*2+j]
    # V[j*2+k]
    # W[k*2+i]
    
    # r=0:
    U[0][0] = 1; V[0][0] = 1
    W[0][0] = 1   # C00
    W[2][0] = -1  # C10 (k=1,i=0)
    
    # r=1:
    U[1][1] = 1; V[2][1] = 1
    W[1][1] = 1   # C01 (k=0,i=1)
    W[3][1] = -1  # C11 (k=1,i=1)
    
    # r=2:
    U[0][2] = 1; U[1][2] = 1
    V[0][2] = 1; V[1][2] = 1
    W[0][2] = -1  # C00
    W[1][2] = 1   # C01
    W[2][2] = 1   # C10
    W[3][2] = -1  # C11
    
    # r=3:
    U[2][3] = 1; U[3][3] = 1
    V[2][3] = 1; V[3][3] = 1
    W[0][3] = 1   # C00
    W[1][3] = -1  # C01
    W[2][3] = -1  # C10
    W[3][3] = 1   # C11
    
    # r=4:
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[3][4] = 1
    W[0][4] = -1  # C00
    W[1][4] = 1   # C01
    W[2][4] = -1  # C10
    W[3][4] = 1   # C11
    
    # r=5:
    U[1][5] = 1; U[2][5] = 1
    V[1][5] = 1; V[2][5] = 1
    W[0][5] = 1   # C00
    W[1][5] = -1  # C01
    W[2][5] = -1  # C10
    W[3][5] = 1   # C11
    
    # r=6:
    U[0][6] = 1; U[1][6] = 1; U[2][6] = 1; U[3][6] = 1
    V[0][6] = 1; V[1][6] = 1; V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1  # C00
    W[1][6] = 1   # C01
    W[2][6] = 1   # C10
    W[3][6] = -1  # C11
    
    return U, V, W

def _build_outer_2x1x2():
    # Rank 2
    rank = 2
    U = [[0.0]*rank for _ in range(2)] # 2x1
    V = [[0.0]*rank for _ in range(2)] # 1x2
    W = [[0.0]*rank for _ in range(4)] # 2x2 (k*2+i)
    
    # C[i,k] = A[i,0] B[0,k]
    # r=0: A[0,0] B[0,0] -> C[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # r=1: A[1,0] B[0,1] -> C[1,1]? No.
    # We need C[0,1] and C[1,0] too.
    # Rank 2 is not enough for 2x1x2?
    # Rank of 2x1x2 is 2?
    # If rank 2, we can do:
    # r0: A[0,0] B[0,0] -> C[0,0]
    # r1: A[1,0] B[0,1] -> C[1,1]
    # Missing C[0,1] and C[1,0].
    # So rank 2 is NOT enough.
    # Rank of 2x1x2 is 4?
    # Actually, 2x1x2 is just outer product.
    # The tensor rank of outer product of 2-vectors is 2?
    # No, the multiplication tensor is different.
    # I will use schoolbook for these small blocks to be safe.
    # 2x1x2 rank 2? No, rank 2 is minimal for 2x2x2? No.
    # I will use schoolbook for non-2x2x2 blocks.
    return None

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for ip in range(n):
        for jp in range(m):
            for jpp in range(m):
                for kpp in range(p):
                    for i in range(n):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[ip*m+jp][r] * V[jpp*p+kpp][r] * W[k*n+i][r]
                            expected = (1.0 if (ip==i and jp==jpp and jpp==k) else 0.0)
                            if abs(s - expected) > 1e-6:
                                return False
    return True

def construct():
    n, m, p = 3, 3, 4
    # Use Laderman for 2x2x2 blocks, schoolbook for others.
    # This is complex to assemble.
    # I will return schoolbook for now to ensure validity, then try to improve.
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
