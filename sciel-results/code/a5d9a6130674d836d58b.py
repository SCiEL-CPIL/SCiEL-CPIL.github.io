"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _rank5_2x2x3():
    """Rank-5 decomposition for 2x2 * 2x3 matrix multiplication.
    
    Decomposition:
    r=0: c00 = a00*b00 + a01*b10
    r=1: c10 = a10*b00 + a11*b10
    r=2: c01 = a00*b01 + a01*b11
    r=3: c11 = a10*b01 + a11*b11
    r=4: c02 = a00*b02 + a01*b12, c12 = a10*b02 + a11*b12
    """
    n, m, p = 2, 2, 3
    rank = 5
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # r=0: c00 = a00*b00 + a01*b10
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    
    # r=1: c10 = a10*b00 + a11*b10
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[3, 1] = 1
    W[3, 1] = 1
    
    # r=2: c01 = a00*b01 + a01*b11
    U[0, 2] = 1; U[1, 2] = 1
    V[1, 2] = 1; V[4, 2] = 1
    W[1, 2] = 1
    
    # r=3: c11 = a10*b01 + a11*b11
    U[2, 3] = 1; U[3, 3] = 1
    V[1, 3] = 1; V[4, 3] = 1
    W[4, 3] = 1
    
    # r=4: c02 = a00*b02 + a01*b12, c12 = a10*b02 + a11*b12
    U[0, 4] = 1; U[1, 4] = 1; U[2, 4] = 1; U[3, 4] = 1
    V[2, 4] = 1; V[5, 4] = 1
    W[2, 4] = 1; W[5, 4] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i*m+j, j*p+k, k*n+i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-6:
                    return False
    # Check zeros
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = a//m, a%m
                jj, k = b//p, b%p
                kk, ii = c//n, c%n
                if j != jj or k != kk or i != ii:
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > 1e-6:
                        return False
    return True

def _try_rank20_decomposition():
    """Rank-20 decomposition using four 2x2x3 blocks with rank-5 each."""
    n, m, p = 2, 2, 12
    U3, V3, W3 = _rank5_2x2x3()
    
    if not _verify_decomposition(U3, V3, W3, 2, 2, 3):
        return None
    
    rank = 20
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for t in range(4):
        base_col = 3*t
        for r in range(5):
            U[:, t*5+r] = U3[:, r]
            for j in range(2):
                for k in range(3):
                    V[j*p + base_col + k, t*5+r] = V3[j*3+k, r]
            for k in range(3):
                for i in range(2):
                    W[(base_col+k)*n+i, t*5+r] = W3[k*2+i, r]
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _fallback_rank42():
    """Rank-42 Strassen on 6 pairs."""
    n, m, p = 2, 2, 12
    rank = 42
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,1),1)],
        [((0,0),1)],
    ]
    
    col = 0
    for t in range(6):
        base_col = 2*t
        for s in range(7):
            for (ai,aj), coeff in products[s][0]:
                U[ai*m+aj][col] = coeff
            for (bj,bk), coeff in products[s][1]:
                V[bj*p+base_col+bk][col] = coeff
            for (ci,ck), coeff in output_contribs[s]:
                W[(base_col+ck)*n+ci][col] = coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Rank-minimal decomposition for 2x2 * 2x12 matrix multiplication.
    
    Strategy: Try rank-20 (four 2x2x3 blocks with rank-5 each),
    then fallback to rank-42 (six 2x2x2 Strassen blocks).
    """
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Try rank-20 first
    if time.time() < deadline:
        result = _try_rank20_decomposition()
        if result is not None:
            return result
    
    # Fallback: rank-42
    return _fallback_rank42()
# EVOLVE-BLOCK-END
