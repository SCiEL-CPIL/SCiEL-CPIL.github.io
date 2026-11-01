"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x4():
    """Rank-10 decomposition for 2x2 * 2x4 using Strassen for first 2x2 block + 3 products for second."""
    n, m, p = 2, 2, 4
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indexing: A[i*m+j], B[j*p+k], W[k*n+i]
    # A = [[a00, a01], [a10, a11]] -> indices 0,1,2,3
    # B = [[b00, b01, b02, b03], [b10, b11, b12, b13]] -> indices 0-7
    # C = [[c00, c01, c02, c03], [c10, c11, c12, c13]] -> W indices: c00->0, c01->1, c02->2, c03->3, c10->4, c11->5, c12->6, c13->7
    
    # Strassen 7 products for B columns 0-1:
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[5, 0] = 1  # c00, c11
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[4, 1] = 1; W[5, 1] = -1  # c10, c11
    
    # M3 = a00(b10-b00)
    U[0, 2] = 1
    V[4, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[5, 2] = 1  # c01, c11
    
    # M4 = a11(b11-b01)
    U[3, 3] = 1
    V[5, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[4, 3] = 1  # c00, c10
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # c00, c01
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[5, 5] = 1  # c11
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1  # c00
    
    # 3 products for B columns 2-3:
    # M8 = (a00+a11)(b02+b13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[2, 7] = 1; W[7, 7] = 1  # c02, c13
    
    # M9 = (a10+a11)b02
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[6, 8] = 1; W[7, 8] = -1  # c12, c13
    
    # M10 = a00(b12-b02)
    U[0, 9] = 1
    V[6, 9] = 1; V[2, 9] = -1
    W[2, 9] = 1; W[6, 9] = 1  # c02, c12
    
    return U, V, W

def _strassen_2x2x6():
    """Rank-11 decomposition for 2x2 * 2x6.
    
    Uses 7 Strassen products for cols 0-1, then 4 products for cols 2-5.
    The 4 products for 4 columns use a shared structure.
    """
    n, m, p = 2, 2, 6
    rank = 11
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First 7: standard Strassen for cols 0-1 (same as in 2x2x4)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[5, 0] = 1  # c00, c11
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[4, 1] = 1; W[5, 1] = -1  # c10, c11
    
    U[0, 2] = 1
    V[4, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[5, 2] = 1  # c01, c11
    
    U[3, 3] = 1
    V[5, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[4, 3] = 1  # c00, c10
    
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # c00, c01
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[5, 5] = 1  # c11
    
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1  # c00
    
    # Now 4 products for cols 2-5 (8 output entries: c02,c03,c04,c05,c12,c13,c14,c15)
    # Strategy: pair columns (2,4) and (3,5) with shared A-factors
    # P7: (a00+a11)(b02+b04+b13+b15) -> c02,c04,c13,c15
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[4, 7] = 1; V[7, 7] = 1; V[9, 7] = 1
    W[2, 7] = 1; W[4, 7] = 1; W[7, 7] = 1; W[9, 7] = 1
    
    # P8: (a10+a11)(b02+b04) -> c12,c14
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1; V[4, 8] = 1
    W[6, 8] = 1; W[8, 8] = 1
    
    # P9: a00(b12+b14-b02-b04) -> c02,c04,c12,c14
    U[0, 9] = 1
    V[6, 9] = 1; V[8, 9] = 1; V[2, 9] = -1; V[4, 9] = -1
    W[2, 9] = 1; W[4, 9] = 1; W[6, 9] = 1; W[8, 9] = 1
    
    # P10: (a01-a11)(b13+b15) -> c03,c05
    U[1, 10] = 1; U[3, 10] = -1
    V[7, 10] = 1; V[9, 10] = 1
    W[3, 10] = 1; W[5, 10] = 1
    
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

def _try_rank22_decomposition():
    """Try to construct a rank-22 decomposition using two 2x2x6 blocks."""
    n, m, p = 2, 2, 12
    U6, V6, W6 = _strassen_2x2x6()
    
    if not _verify_decomposition(U6, V6, W6, 2, 2, 6):
        return None
    
    rank = 22
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for t in range(2):
        base_col = 6*t
        for r in range(11):
            U[:, t*11+r] = U6[:, r]
            for j in range(2):
                for k in range(6):
                    V[j*p + base_col + k, t*11+r] = V6[j*6+k, r]
            for k in range(6):
                for i in range(2):
                    W[(base_col+k)*n+i, t*11+r] = W6[k*2+i, r]
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_rank30_decomposition():
    """Rank-30 decomposition using three 2x2x4 blocks."""
    n, m, p = 2, 2, 12
    U4, V4, W4 = _strassen_2x2x4()
    
    if not _verify_decomposition(U4, V4, W4, 2, 2, 4):
        return None
    
    rank = 30
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for t in range(3):
        base_col = 4*t
        for r in range(10):
            U[:, t*10+r] = U4[:, r]
            for j in range(2):
                for k in range(4):
                    V[j*p + base_col + k, t*10+r] = V4[j*4+k, r]
            for k in range(4):
                for i in range(2):
                    W[(base_col+k)*n+i, t*10+r] = W4[k*2+i, r]
    
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
    
    Strategy: Try rank-22 (two 2x2x6 blocks), then rank-30 (three 2x2x4 blocks),
    then fallback to rank-42 (six 2x2x2 Strassen blocks).
    """
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Try rank-22 first
    if time.time() < deadline:
        result = _try_rank22_decomposition()
        if result is not None:
            return result
    
    # Try rank-30
    if time.time() < deadline:
        result = _try_rank30_decomposition()
        if result is not None:
            return result
    
    # Fallback: rank-42
    return _fallback_rank42()
# EVOLVE-BLOCK-END
