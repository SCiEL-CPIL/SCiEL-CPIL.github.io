"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def _strassen_222():
    """Return (U, V, W) for M_{2,2,2} with rank 7 (Strassen)."""
    # U: shape (4, 7), rows indexed by i*2+j
    U = [
        [1,  0, 1,  0, 1, -1,  0],  # A00
        [0,  0, 0,  0, 1,  0,  1],  # A01
        [0,  1, 0,  0, 0,  1,  0],  # A10
        [1,  1, 0,  1, 0,  0, -1],  # A11
    ]
    # V: shape (4, 7), rows indexed by j*2+k
    V = [
        [ 1, 1, 0, -1, 0, 1, 0],  # B00
        [ 0, 0, 1,  0, 0, 1, 0],  # B01
        [ 0, 0, 0,  1, 0, 0, 0],  # B10  -- wait, let me recheck
        [ 1, 0,-1,  0, 1, 0, 1],  # B11
    ]
    # Let me recompute V carefully.
    # M1 = (A00+A11)*(B00+B11): B00 coeff=1, B11 coeff=1
    # M2 = (A10+A11)*B00: B00 coeff=1
    # M3 = A00*(B01-B11): B01 coeff=1, B11 coeff=-1
    # M4 = A11*(B10-B00): B10 coeff=1, B00 coeff=-1
    # M5 = (A00+A01)*B11: B11 coeff=1
    # M6 = (A10-A00)*(B00+B01): B00 coeff=1, B01 coeff=1
    # M7 = (A01-A11)*(B10+B11): B10 coeff=1, B11 coeff=1
    
    # V row 0 = B00: [1, 1, 0, -1, 0, 1, 0]
    # V row 1 = B01: [0, 0, 1, 0, 0, 1, 0]
    # V row 2 = B10: [0, 0, 0, 1, 0, 0, 1]
    # V row 3 = B11: [1, 0, -1, 0, 1, 0, 1]
    
    V = [
        [ 1, 1,  0, -1, 0, 1,  0],  # B00
        [ 0, 0,  1,  0, 0, 1,  0],  # B01
        [ 0, 0,  0,  1, 0, 0,  1],  # B10
        [ 1, 0, -1,  0, 1, 0,  1],  # B11
    ]
    
    # W: shape (4, 7), rows indexed by k*2+i
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W row 0 = C00 (k=0,i=0): [1, 0, 0, 1, -1, 0, 1]
    # W row 1 = C10 (k=0,i=1): [0, 1, 0, 1, 0, 0, 0]
    # W row 2 = C01 (k=1,i=0): [0, 0, 1, 0, 1, 0, 0]
    # W row 3 = C11 (k=1,i=1): [1, -1, 1, 0, 0, 1, 0]
    
    W = [
        [ 1,  0, 0,  1, -1, 0, 1],  # C00
        [ 0,  1, 0,  1,  0, 0, 0],  # C10
        [ 0,  0, 1,  0,  1, 0, 0],  # C01
        [ 1, -1, 1,  0,  0, 1, 0],  # C11
    ]
    
    return U, V, W


def _trivial_112():
    """Return (U, V, W) for M_{1,1,2} with rank 2."""
    U = [[1, 1]]
    V = [[1, 0],
         [0, 1]]
    W = [[1, 0],
         [0, 1]]
    return U, V, W


def _schoolbook(n, m, p):
    """Return (U, V, W) for M_{n,m,p} with schoolbook rank n*m*p."""
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
    return U, V, W


def _tensor_product(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Compute tensor product of two decompositions.
    T1 is for M_{n1,m1,p1} with rank r1.
    T2 is for M_{n2,m2,p2} with rank r2.
    Result is for M_{n1*n2, m1*m2, p1*p2} with rank r1*r2.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for r1i in range(r1):
        for r2i in range(r2):
            ridx = r1i * r2 + r2i
            for i1 in range(n1):
                for i2 in range(n2):
                    for j1 in range(m1):
                        for j2 in range(m2):
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            val = U1[i1 * m1 + j1][r1i] * U2[i2 * m2 + j2][r2i]
                            U[i * m + j][ridx] = val
                            
                            # V
                            for k1 in range(p1):
                                for k2 in range(p2):
                                    k = k1 * p2 + k2
                                    val_v = V1[j1 * p1 + k1][r1i] * V2[j2 * p2 + k2][r2i]
                                    V[j * p + k][ridx] = val_v
                                    
                                    # W
                                    val_w = W1[k1 * n1 + i1][r1i] * W2[k2 * n2 + i2][r2i]
                                    W[k * n + i][ridx] = val_w
    
    return U, V, W


def _concatenate(U1, V1, W1, U2, V2, W2):
    """
    Concatenate two decompositions of the same tensor (same dimensions).
    U1,V1,W1 have rank r1; U2,V2,W2 have rank r2.
    Result has rank r1+r2.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 + r2
    rows_u = len(U1)
    rows_v = len(V1)
    rows_w = len(W1)
    
    U = [row1 + row2 for row1, row2 in zip(U1, U2)]
    V = [row1 + row2 for row1, row2 in zip(V1, V2)]
    W = [row1 + row2 for row1, row2 in zip(W1, W2)]
    
    return U, V, W


def construct():
    """
    Decompose 4x4x7 matrix multiplication tensor.
    
    Strategy:
    1. M_{2,2,2} via Strassen (rank 7)
    2. M_{1,1,2} trivial (rank 2)
    3. Tensor product: M_{2,2,4} = M_{2,2,2} ⊗ M_{1,1,2} (rank 14)
    4. M_{2,2,3} schoolbook (rank 12)
    5. Concatenate: M_{2,2,7} (rank 26)
    6. M_{2,2,1} schoolbook (rank 4)
    7. Tensor product: M_{4,4,7} = M_{2,2,7} ⊗ M_{2,2,1} (rank 104)
    """
    # Step 1: Strassen for 2x2x2
    U1, V1, W1 = _strassen_222()
    
    # Step 2: Trivial 1x1x2
    U2, V2, W2 = _trivial_112()
    
    # Step 3: Tensor product for 2x2x4
    U3, V3, W3 = _tensor_product(U1, V1, W1, U2, V2, W2, 2, 2, 2, 1, 1, 2)
    
    # Step 4: Schoolbook for 2x2x3
    U4, V4, W4 = _schoolbook(2, 2, 3)
    
    # Step 5: Concatenate for 2x2x7
    U5, V5, W5 = _concatenate(U3, V3, W3, U4, V4, W4)
    
    # Step 6: Schoolbook for 2x2x1
    U6, V6, W6 = _schoolbook(2, 2, 1)
    
    # Step 7: Tensor product for 4x4x7
    U7, V7, W7 = _tensor_product(U5, V5, W5, U6, V6, W6, 2, 2, 7, 2, 2, 1)
    
    return {"U": U7, "V": V7, "W": W7}
# EVOLVE-BLOCK-END
