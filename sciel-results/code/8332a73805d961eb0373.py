"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time

def _construct_3x3x2_rank17():
    """
    Return U, V, W for rank-17 decomposition of 3x3x2 matrix multiplication.
    n=3, m=3, p=2.
    U: (9, 17), V: (6, 17), W: (6, 17)
    T[i,j,k] = sum_c U[i*3+j, c] * V[j*2+k, c] * W[k*3+i, c] = 1 for all valid i,j,k
    """
    n, m, p = 3, 3, 2
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 9 x 17
    V = [[0] * rank for _ in range(m * p)]  # 6 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17
    
    # Strassen 2x2x2 for indices i in {0,1}, j in {0,1}, k in {0,1}
    # c=0: (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[4][0] = 1
    
    # c=1: (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1; W[4][1] = 1
    
    # c=2: A00(B11-B10)
    U[0][2] = 1
    V[3][2] = 1; V[2][2] = -1
    W[0][2] = 1; W[3][2] = -1
    
    # c=3: A11(B00-B01)
    U[4][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[1][3] = -1; W[4][3] = 1
    
    # c=4: (A00+A01)B10
    U[0][4] = 1; U[1][4] = 1
    V[2][4] = 1
    W[0][4] = 1; W[3][4] = 1
    
    # c=5: (A01+A10)(B01-B10)
    U[1][5] = 1; U[3][5] = 1
    V[1][5] = 1; V[2][5] = -1
    W[1][5] = 1; W[3][5] = -1
    
    # c=6: (A10+A11)B01
    U[3][6] = 1; U[4][6] = 1
    V[1][6] = 1
    W[1][6] = 1; W[4][6] = 1
    
    # Cross terms involving index 2
    # c=7: A02*B20 -> C00
    U[2][7] = 1; V[4][7] = 1; W[0][7] = 1
    # c=8: A02*B21 -> C01
    U[2][8] = 1; V[5][8] = 1; W[3][8] = 1
    # c=9: A12*B20 -> C10
    U[5][9] = 1; V[4][9] = 1; W[3][9] = 1
    # c=10: A12*B21 -> C11
    U[5][10] = 1; V[5][10] = 1; W[4][10] = 1
    # c=11: A20*B00 -> C20
    U[6][11] = 1; V[0][11] = 1; W[2][11] = 1
    # c=12: A20*B01 -> C21
    U[6][12] = 1; V[1][12] = 1; W[5][12] = 1
    # c=13: A21*B10 -> C20
    U[7][13] = 1; V[2][13] = 1; W[2][13] = 1
    # c=14: A21*B11 -> C21
    U[7][14] = 1; V[3][14] = 1; W[5][14] = 1
    # c=15: A22*B20 -> C20
    U[8][15] = 1; V[4][15] = 1; W[2][15] = 1
    # c=16: A22*B21 -> C21
    U[8][16] = 1; V[5][16] = 1; W[5][16] = 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _construct_3x3x6_rank51():
    """
    Construct rank-51 decomposition for 3x3x6 by using three rank-17 
    decompositions for 3x3x2.
    """
    n, m, p = 3, 3, 6
    rank_block = 17
    rank = rank_block * 3 # 51
    
    # Get one 3x3x2 decomposition
    dec = _construct_3x3x2_rank17()
    U2 = dec["U"]  # 9 x 17
    V2 = dec["V"]  # 6 x 17
    W2 = dec["W"]  # 6 x 17
    
    # Verify the 3x3x2 decomposition
    if not _verify_decomposition(U2, V2, W2, 3, 3, 2, rank_block):
        # Fall back to schoolbook
        return _schoolbook(n, m, p)
    
    # Build 3x3x6 decomposition
    U = [[0] * rank for _ in range(n * m)]  # 9 x 51
    V = [[0] * rank for _ in range(m * p)]  # 18 x 51
    W = [[0] * rank for _ in range(p * n)]  # 18 x 51
    
    for block in range(3):
        base = block * rank_block
        for i in range(n * m):
            for c in range(rank_block):
                U[i][base + c] = U2[i][c]
        
        for j in range(m):
            for k in range(2):
                v_idx = j * 2 + k
                # For block 'block', the k index in 3x3x2 maps to k + 2*block in 3x3x6
                # V index: j*p + k_global = j*6 + (2*block + k)
                for c in range(rank_block):
                    V[j * p + 2 * block + k][base + c] = V2[v_idx][c]
        
        for k in range(2):
            for i in range(n):
                w_idx = k * n + i
                for c in range(rank_block):
                    W[(2 * block + k) * n + i][base + c] = W2[w_idx][c]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    try:
        result = _construct_3x3x6_rank51()
        if time.time() > deadline:
            return _schoolbook(n, m, p)
        return result
    except Exception:
        return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
