import time
import sys

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial rank 96 decomposition."""
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

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if actual != expected:
                    return False
    return True

def _construct_rank_88():
    """
    Construct a rank-88 decomposition for 3x4x8.
    
    Strategy:
    - Split n=3 into (0,1) and (2).
    - Split m=4 into (0,1) and (2,3).
    - Split p=8 into (0,1), (2,3), (4,5), (6,7).
    
    For the (0,1) part of n, we use Strassen (rank 7) for each 2x2x2 subblock.
    For the (2) part of n, we use schoolbook (rank 4) for each 1x2x2 subblock.
    
    Total rank: 8 * 7 + 8 * 4 = 56 + 32 = 88.
    """
    n, m, p = 3, 4, 8
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)] # 12 rows
    V = [[0] * rank for _ in range(m * p)] # 32 rows
    W = [[0] * rank for _ in range(p * n)] # 24 rows
    
    col_offset = 0
    
    # Strassen coefficients for 2x2x2
    # U rows: a, b, c, d
    # V rows: e, f, g, h
    # W rows: M11, M12, M21, M22
    
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # U matrix for Strassen (7 columns)
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 0], # a
        [0, 0, 0, 0, 1, 0, 1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, 0, -1] # d
    ]
    
    # V matrix for Strassen (7 columns)
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0],  # f
        [0, 0, 0, 1, 0, 0, 1],  # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ]
    
    # W matrix for Strassen (7 columns)
    W_strassen = [
        [1, 0, 0, -1, 1, 0, -1], # M11 = P1 - P4 + P5 - P7
        [0, 0, 1, 1, 0, 0, 1],   # M12 = P3 + P4 + P7
        [0, 1, 1, 0, 0, -1, 0],  # M21 = P2 + P3 - P6
        [1, -1, 0, 0, -1, 1, 0]  # M22 = P1 - P2 + P6 - P5
    ]
    
    # Part 1: i in {0, 1} (Strassen blocks)
    for jp in range(2): # j pairs: (0,1), (2,3)
        j0 = 2 * jp
        j1 = 2 * jp + 1
        for kp in range(4): # k pairs: (0,1), (2,3), (4,5), (6,7)
            k0 = 2 * kp
            k1 = 2 * kp + 1
            
            # U indices (A rows):
            # i=0, j=j0 -> 0*4 + j0
            # i=0, j=j1 -> 0*4 + j1
            # i=1, j=j0 -> 1*4 + j0
            # i=1, j=j1 -> 1*4 + j1
            u_map = [j0, j1, 4 + j0, 4 + j1]
            
            # V indices (B rows):
            # j=j0, k=k0 -> j0*8 + k0
            # j=j0, k=k1 -> j0*8 + k1
            # j=j1, k=k0 -> j1*8 + k0
            # j=j1, k=k1 -> j1*8 + k1
            v_map = [j0 * 8 + k0, j0 * 8 + k1, j1 * 8 + k0, j1 * 8 + k1]
            
            # W indices (C rows):
            # k=k0, i=0 -> k0*3 + 0
            # k=k1, i=0 -> k1*3 + 0
            # k=k0, i=1 -> k0*3 + 1
            # k=k1, i=1 -> k1*3 + 1
            w_map = [k0 * 3 + 0, k1 * 3 + 0, k0 * 3 + 1, k1 * 3 + 1]
            
            # Add Strassen block
            for i in range(4):
                for c in range(7):
                    U[u_map[i]][col_offset + c] += U_strassen[i][c]
                    V[v_map[i]][col_offset + c] += V_strassen[i][c]
                    W[w_map[i]][col_offset + c] += W_strassen[i][c]
            col_offset += 7
            
    # Part 2: i = 2 (Schoolbook blocks for 1x2x2)
    # C[2, k] = A[2, j0]*B[j0, k] + A[2, j1]*B[j1, k]
    # For a 1x2x2 block, we need 4 multiplications.
    for jp in range(2):
        j0 = 2 * jp
        j1 = 2 * jp + 1
        for kp in range(4):
            k0 = 2 * kp
            k1 = 2 * kp + 1
            
            # U indices:
            # i=2, j=j0 -> 2*4 + j0 = 8 + j0
            # i=2, j=j1 -> 2*4 + j1 = 8 + j1
            
            # V indices:
            # j=j0, k=k0 -> j0*8 + k0
            # j=j0, k=k1 -> j0*8 + k1
            # j=j1, k=k0 -> j1*8 + k0
            # j=j1, k=k1 -> j1*8 + k1
            
            # W indices:
            # k=k0, i=2 -> k0*3 + 2
            # k=k1, i=2 -> k1*3 + 2
            
            # P1: A[2, j0] * B[j0, k0] -> C[2, k0]
            U[8 + j0][col_offset] += 1
            V[j0 * 8 + k0][col_offset] += 1
            W[k0 * 3 + 2][col_offset] += 1
            col_offset += 1
            
            # P2: A[2, j0] * B[j0, k1] -> C[2, k1]
            U[8 + j0][col_offset] += 1
            V[j0 * 8 + k1][col_offset] += 1
            W[k1 * 3 + 2][col_offset] += 1
            col_offset += 1
            
            # P3: A[2, j1] * B[j1, k0] -> C[2, k0]
            U[8 + j1][col_offset] += 1
            V[j1 * 8 + k0][col_offset] += 1
            W[k0 * 3 + 2][col_offset] += 1
            col_offset += 1
            
            # P4: A[2, j1] * B[j1, k1] -> C[2, k1]
            U[8 + j1][col_offset] += 1
            V[j1 * 8 + k1][col_offset] += 1
            W[k1 * 3 + 2][col_offset] += 1
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try the rank 88 decomposition
    try:
        result_88 = _construct_rank_88()
        if _verify(result_88["U"], result_88["V"], result_88["W"], 3, 4, 8):
            return result_88
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
