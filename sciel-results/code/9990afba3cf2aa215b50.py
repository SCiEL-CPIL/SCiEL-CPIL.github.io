"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

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

def _strassen_2x2x4_rank10():
    """Rank-10 decomposition for 2x2 * 2x4."""
    n, m, p = 2, 2, 4
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # P1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # P3 = a00(b10-b00)
    U[0, 2] = 1
    V[4, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = a11(b11-b01)
    U[3, 3] = 1
    V[5, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1
    
    # P8 = (a00+a11)(b02+b13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1
    
    # P9 = (a10+a11)b02
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[6, 8] = 1; W[7, 8] = 1
    
    # P10 = a00(b12-b02)
    U[0, 9] = 1
    V[6, 9] = 1; V[2, 9] = -1
    W[5, 9] = 1; W[7, 9] = 1
    
    return U, V, W

def _strassen_2x2x4_rank14():
    """Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2x2."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
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
    for t in range(2):
        base_col = 2*t
        for s in range(7):
            for (ai,aj), coeff in products[s][0]:
                U[ai*m+aj, col] = coeff
            for (bj,bk), coeff in products[s][1]:
                V[bj*p+base_col+bk, col] = coeff
            for (ci,ck), coeff in output_contribs[s]:
                W[(base_col+ck)*n+ci, col] = coeff
            col += 1
    return U, V, W

def construct():
    """Rank-minimal decomposition for 2x2 * 2x12 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Try rank-10 decomposition for 2x2x4
    U4, V4, W4 = _strassen_2x2x4_rank10()
    if _verify_decomposition(U4, V4, W4, 2, 2, 4):
        # Tile 3 times for 2x2x12
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
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: rank-42 Strassen on 6 pairs (2x2x4 rank 14 * 3)
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
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
                U[ai*m+aj, col] = coeff
            for (bj,bk), coeff in products[s][1]:
                V[bj*p+base_col+bk, col] = coeff
            for (ci,ck), coeff in output_contribs[s]:
                W[(base_col+ck)*n+ci, col] = coeff
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
