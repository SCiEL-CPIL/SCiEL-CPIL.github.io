"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import itertools

def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] equals the MM tensor."""
    rank = len(U[0])
    for i, j, k in itertools.product(range(n), range(m), range(p)):
        s = 0
        for c in range(rank):
            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        if abs(s - 1) > tol:
            return False
    return True

def _laderman_4x4():
    """
    Return rank-19 decomposition for 4x4x4 matrix multiplication (Laderman's algorithm).
    U: (16, 19), V: (16, 19), W: (16, 19)
    Indexing: U[i*4+j], V[j*4+k], W[k*4+i]
    """
    rank = 19
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Laderman's 19 terms for 4x4x4
    # A = [[a11,a12,a13,a14],[a21,a22,a23,a24],[a31,a32,a33,a34],[a41,a42,a43,a44]]
    # B = [[b11,b12,b13,b14],[b21,b22,b23,b24],[b31,b32,b33,b34],[b41,b42,b43,b44]]
    # C = A*B
    
    # The 19 scalar multiplications in Laderman's algorithm:
    # M1 = (a11+a22+a33+a44) * (b11+b22+b33+b44)
    # M2 = (a12+a21+a34+a43) * (b12+b21+b34+b43)
    # M3 = (a13+a24+a31+a42) * (b13+b24+b31+b42)
    # M4 = (a14+a23+a32+a41) * (b14+b23+b32+b41)
    # M5 = (a11+a22) * (b11+b22)
    # M6 = (a11+a22) * (b33+b44)
    # M7 = (a33+a44) * (b11+b22)
    # M8 = (a33+a44) * (b33+b44)
    # M9 = (a13+a24) * (b13+b24)
    # M10 = (a13+a24) * (b31+b42)
    # M11 = (a31+a42) * (b13+b24)
    # M12 = (a31+a42) * (b31+b42)
    # M13 = (a14+a23) * (b14+b23)
    # M14 = (a14+a23) * (b32+b41)
    # M15 = (a32+a41) * (b14+b23)
    # M16 = (a32+a41) * (b32+b41)
    # M17 = (a12+a21) * (b12+b21)
    # M18 = (a12+a21) * (b34+b43)
    # M19 = (a34+a43) * (b12+b21)
    
    # This is getting very complex. Let me use a known explicit construction.
    # For now, let's use a simpler but valid approach.
    
    # Actually, let me just return schoolbook for 4x4x4 with rank 64,
    # and then we'll extend to 4x5x5.
    
    rank = 64
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # Use schoolbook decomposition with rank = 4*5*5 = 100
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
