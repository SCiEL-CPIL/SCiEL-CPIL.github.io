import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _strassen_222_factors():
    """
    Returns U, V, W for 2x2x2 multiplication with rank 7.
    Indices:
    U: (4, 7) for A (2x2)
    V: (4, 7) for B (2x2)
    W: (4, 7) for C (2x2)
    A = [[a00, a01], [a10, a11]] -> U rows: a00, a01, a10, a11
    B = [[b00, b01], [b10, b11]] -> V rows: b00, b01, b10, b11
    C = [[c00, c01], [c10, c11]] -> W rows: c00, c01, c10, c11
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1 # c00, c11
    
    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1 # c10, c11 (c11 = M1 - M2 ...)
    
    # M3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1 # c01, c11
    
    # M4 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1 # c00, c10
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1 # c00, c01
    
    # M6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # c11
    
    # M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1 # c00

    return U, V, W

def _add_factors(U, V, W, u_sub, v_sub, w_sub, offset, row_map_u, row_map_v, row_map_w):
    """
    Adds sub-factors to main factors.
    u_sub, v_sub, w_sub: lists of lists (factors for sub-problem)
    offset: column offset in main factors
    row_map_u: list of row indices in main U corresponding to sub-U rows
    """
    for i, row in enumerate(u_sub):
        for j, val in enumerate(row):
            if val != 0:
                U[row_map_u[i]][offset + j] += val
    for i, row in enumerate(v_sub):
        for j, val in enumerate(row):
            if val != 0:
                V[row_map_v[i]][offset + j] += val
    for i, row in enumerate(w_sub):
        for j, val in enumerate(row):
            if val != 0:
                W[row_map_w[i]][offset + j] += val

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    n, m, p = 3, 4, 6
    # We will build rank 54 decomposition
    rank = 54
    U = [[0]*rank for _ in range(n*m)] # 12x54
    V = [[0]*rank for _ in range(m*p)] # 24x54
    W = [[0]*rank for _ in range(p*n)] # 18x54
    
    u_222, v_222, w_222 = _strassen_222_factors()
    
    # Sub-problems:
    # 1. C[0:2, 0:4] = A[0:2, :] B[:, 0:4] (2x4x4)
    #    Split into C[0:2, 0:2] and C[0:2, 2:4]
    # 2. C[0:2, 4:6] = A[0:2, :] B[:, 4:6] (2x4x2)
    # 3. C[2:3, 0:4] = A[2:3, :] B[:, 0:4] (1x4x4)
    # 4. C[2:3, 4:6] = A[2:3, :] B[:, 4:6] (1x4x2)
    
    col_offset = 0
    
    # Helper to get 2x2x2 factors for specific indices
    # A indices: (i, j) -> row i*m + j
    # B indices: (j, k) -> row j*p + k
    # C indices: (k, i) -> row k*n + i (Note: W[k*n+i])
    
    # 1a. C[0:2, 0:2] = A[0:2, 0:2] B[0:2, 0:2] + A[0:2, 2:4] B[2:4, 0:2]
    #     Term 1: A_L (2x2) * B_T (2x2). A rows 0,1. B rows 0,1.
    #     A_L indices: (0,0), (0,1), (1,0), (1,1) -> U rows 0,1,4,5
    #     B_T indices: (0,0), (0,1), (1,0), (1,1) -> V rows 0,1,4,5
    #     C_TL indices: (0,0), (0,1), (1,0), (1,1) -> W rows 0, 1*n+0=3, 2*n+0=6, 2*n+1=7
    #     Wait, W index is k*n + i.
    #     C[0,0] -> k=0, i=0 -> 0
    #     C[0,1] -> k=0, i=1 -> 1
    #     C[1,0] -> k=1, i=0 -> 3
    #     C[1,1] -> k=1, i=1 -> 4
    #     So W rows: 0, 1, 3, 4.
    
    # Let's map 2x2 block to global indices
    def get_222_maps(i0, j0, k0):
        # A block at row i0, col j0
        u_map = [i0*m + j0, i0*m + j0+1, (i0+1)*m + j0, (i0+1)*m + j0+1]
        # B block at row j0, col k0
        v_map = [j0*p + k0, j0*p + k0+1, (j0+1)*p + k0, (j0+1)*p + k0+1]
        # C block at row k0, col i0
        w_map = [k0*n + i0, k0*n + i0+1, (k0+1)*n + i0, (k0+1)*n + i0+1]
        return u_map, v_map, w_map

    # 1a. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    u_map, v_map, w_map = get_222_maps(0, 0, 0)
    _add_factors(U, V, W, u_222, v_222, w_222, col_offset, u_map, v_map, w_map)
    col_offset += 7
    
    # 1b. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    u_map, v_map, w_map = get_222_maps(0, 2, 0)
    _add_factors(U, V, W, u_222, v_222, w_222, col_offset, u_map, v_map, w_map)
    col_offset += 7
    
    # 1c. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    u_map, v_map, w_map = get_222_maps(0, 0, 2)
    _add_factors(U, V, W, u_222, v_222, w_222, col_offset, u_map, v_map, w_map)
    col_offset += 7
    
    # 1d. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    u_map, v_map, w_map = get_222_maps(0, 2, 2)
    _add_factors(U, V, W, u_222, v_222, w_222, col_offset, u_map, v_map, w_map)
    col_offset += 7
    
    # 2. C[0:2, 4:6] = A[0:2, 0:2] B[0:2, 4:6] + A[0:2, 2:4] B[2:4, 4:6]
    # 2a. A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    u_map, v_map, w_map = get_222_maps(0, 0, 4)
    _add_factors(U, V, W, u_222, v_222, w_222, col_offset, u_map, v_map, w_map)
    col_offset += 7
    
    # 2b. A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    u_map, v_map, w_map = get_222_maps(0, 2, 4)
    _add_factors(U, V, W, u_222, v_222, w_222, col_offset, u_map, v_map, w_map)
    col_offset += 7
    
    # 3. C[2:3, 0:4] = A[2:3, 0:2] B[0:2, 0:4] + A[2:3, 2:4] B[2:4, 0:4]
    # A is 1x2. B is 2x2. C is 1x2.
    # Rank 2 for 1x2x2?
    # C[2, k] = sum_j A[2, j] B[j, k]
    # This is just dot products.
    # We can use 2 rank-1 terms.
    # Term 1: A[2,0] B[0,0] -> C[2,0]; A[2,0] B[0,1] -> C[2,1]
    # Term 2: A[2,1] B[1,0] -> C[2,0]; A[2,1] B[1,1] -> C[2,1]
    
    # 3a. A[2,0] * B[0,0:2]
    # U row 2*m+0 = 8. V rows 0,1. W rows 2*n+0=6, 2*n+1=7.
    U[8][col_offset] = 1
    V[0][col_offset] = 1
    V[1][col_offset] = 1
    W[6][col_offset] = 1
    W[7][col_offset] = 1
    col_offset += 1
    
    # 3b. A[2,1] * B[1,0:2]
    U[8+1][col_offset] = 1 # Row 9
    V[2][col_offset] = 1 # V row 1*p+0 = 4? No, B[1,0] is row 1*6+0=6?
    # B index j*p+k. j=1, k=0 -> 6. j=1, k=1 -> 7.
    V[6][col_offset] = 1
    V[7][col_offset] = 1
    W[6][col_offset] = 1
    W[7][col_offset] = 1
    col_offset += 1
    
    # 3c. A[2,0] * B[0,2:4]
    U[8][col_offset] = 1
    V[2][col_offset] = 1 # B[0,2] -> 0*6+2=2
    V[3][col_offset] = 1 # B[0,3] -> 0*6+3=3
    W[6][col_offset] = 1 # C[2,0] -> 2*3+0=6? No, C[2,0] is k=0, i=2 -> 0*3+2=2.
    # Wait, W index is k*n+i.
    # C[2,0] -> k=0, i=2 -> 2.
    # C[2,1] -> k=1, i=2 -> 5.
    # My previous W map for 2x2 was for C[0:2, 0:2].
    # For C[2:3, ...], i=2.
    # W rows for C[2,0] and C[2,1] are 2 and 5.
    
    # Let's re-verify W index.
    # W[k*n+i].
    # C[2,0]: k=0, i=2 -> 2.
    # C[2,1]: k=1, i=2 -> 5.
    
    # Correcting 3a:
    W[2][col_offset] = 1
    W[5][col_offset] = 1
    col_offset += 1
    
    # Correcting 3b:
    W[2][col_offset] = 1
    W[5][col_offset] = 1
    col_offset += 1
    
    # Correcting 3c:
    W[2][col_offset] = 1
    W[5][col_offset] = 1
    col_offset += 1
    
    # 3d. A[2,1] * B[1,2:4]
    U[9][col_offset] = 1
    V[8][col_offset] = 1 # B[1,2] -> 1*6+2=8
    V[9][col_offset] = 1 # B[1,3] -> 1*6+3=9
    W[2][col_offset] = 1
    W[5][col_offset] = 1
    col_offset += 1
    
    # 4. C[2:3, 4:6]
    # 4a. A[2,0] * B[0,4:6]
    U[8][col_offset] = 1
    V[4][col_offset] = 1 # B[0,4]
    V[5][col_offset] = 1 # B[0,5]
    W[2][col_offset] = 1
    W[5][col_offset] = 1
    col_offset += 1
    
    # 4b. A[2,1] * B[1,4:6]
    U[9][col_offset] = 1
    V[10][col_offset] = 1 # B[1,4]
    V[11][col_offset] = 1 # B[1,5]
    W[2][col_offset] = 1
    W[5][col_offset] = 1
    col_offset += 1
    
    # 4c. A[2,0] * B[2,4:6]
    U[8][col_offset] = 1
    V[16][col_offset] = 1 # B[2,4]
    V[17][col_offset] = 1
    W[2][col_offset] = 1
    W[5][col_offset] = 1
    col_offset += 1
    
    # 4d. A[2,1] * B[3,4:6]
    U[9][col_offset] = 1
    V[22][col_offset] = 1 # B[3,4]
    V[23][col_offset] = 1
    W[2][col_offset] = 1
    W[5][col_offset] = 1
    col_offset += 1
    
    # Total rank used: 7*6 + 1*8 = 42 + 8 = 50.
    # Wait, 1a-2b is 6 blocks of 7 = 42.
    # 3a-3d is 4 blocks of 1 = 4.
    # 4a-4d is 4 blocks of 1 = 4.
    # Total 50.
    
    if col_offset < rank:
        # Pad with zeros
        pass
        
    return {"U": U, "V": V, "W": W}
