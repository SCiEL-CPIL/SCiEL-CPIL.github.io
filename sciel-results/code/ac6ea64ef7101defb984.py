"""Program for matrix multiplication 3x3x8 using Strassen-based decomposition."""

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7) indexed by [i*2+j]
    V: (4, 7) indexed by [j*2+k]
    W: (4, 7) indexed by [k*2+i]
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0*2+0][0] = 1  # a00
    U[1*2+1][0] = 1  # a11
    V[0*2+0][0] = 1  # b00
    V[1*2+1][0] = 1  # b11
    W[0*2+0][0] = 1  # c00
    W[1*2+1][0] = 1  # c11
    
    # M2 = (a10+a11)b00
    U[1*2+0][1] = 1  # a10
    U[1*2+1][1] = 1  # a11
    V[0*2+0][1] = 1  # b00
    W[1*2+0][1] = 1  # c10
    
    # M3 = a00(b10-b11)
    U[0*2+0][2] = 1  # a00
    V[1*2+0][2] = 1  # b10
    V[1*2+1][2] = -1 # b11
    W[0*2+1][2] = 1  # c01
    
    # M4 = a11(b01-b00)
    U[1*2+1][3] = 1  # a11
    V[0*2+1][3] = 1  # b01
    V[0*2+0][3] = -1 # b00
    W[0*2+0][3] = 1  # c00
    W[1*2+0][3] = 1  # c10
    
    # M5 = (a00+a01)b11
    U[0*2+0][4] = 1  # a00
    U[0*2+1][4] = 1  # a01
    V[1*2+1][4] = 1  # b11
    W[0*2+0][4] = -1 # c00
    W[0*2+1][4] = 1  # c01
    
    # M6 = (a10-a00)(b00+b10)
    U[1*2+0][5] = 1  # a10
    U[0*2+0][5] = -1 # a00
    V[0*2+0][5] = 1  # b00
    V[1*2+0][5] = 1  # b10
    W[1*2+1][5] = 1  # c11
    
    # M7 = (a01-a11)(b11+b01)
    U[0*2+1][6] = 1  # a01
    U[1*2+1][6] = -1 # a11
    V[1*2+1][6] = 1  # b11
    V[0*2+1][6] = 1  # b01
    W[0*2+0][6] = 1  # c00
    W[1*2+1][6] = 1  # c11
    
    return U, V, W

def _embed_2x2_in_4x4x4(base_i, base_j, base_k, U_2, V_2, W_2, U_4, V_4, W_4, offset):
    """
    Embed a 2x2x2 Strassen block into a 4x4x4 decomposition.
    base_i, base_j, base_k: starting indices (0 or 2)
    U_2, V_2, W_2: the 2x2 Strassen matrices
    U_4, V_4, W_4: the 4x4x4 matrices being built
    offset: column offset in the 4x4x4 decomposition
    """
    for i in range(2):
        for j in range(2):
            for c in range(7):
                if U_2[i*2+j][c] != 0:
                    U_4[(base_i + i) * 4 + (base_j + j)][offset + c] += U_2[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                if V_2[j*2+k][c] != 0:
                    V_4[(base_j + j) * 4 + (base_k + k)][offset + c] += V_2[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                if W_2[k*2+i][c] != 0:
                    W_4[(base_k + k) * 4 + (base_i + i)][offset + c] += W_2[k*2+i][c]

def _strassen_4x4x4():
    """
    Build a 4x4x4 decomposition using 4 Strassen 2x2x2 blocks.
    """
    U_2, V_2, W_2 = _strassen_2x2()
    
    U_4 = [[0] * 28 for _ in range(16)]
    V_4 = [[0] * 28 for _ in range(16)]
    W_4 = [[0] * 28 for _ in range(16)]
    
    # 4 blocks: (0,0), (0,2), (2,0), (2,2)
    _embed_2x2_in_4x4x4(0, 0, 0, U_2, V_2, W_2, U_4, V_4, W_4, 0)
    _embed_2x2_in_4x4x4(0, 2, 0, U_2, V_2, W_2, U_4, V_4, W_4, 7)
    _embed_2x2_in_4x4x4(2, 0, 0, U_2, V_2, W_2, U_4, V_4, W_4, 14)
    _embed_2x2_in_4x4x4(2, 2, 0, U_2, V_2, W_2, U_4, V_4, W_4, 21)
    
    return U_4, V_4, W_4

def _embed_4x4x4_in_4x4x8(base_k, U_4, V_4, W_4, U_8, V_8, W_8, offset):
    """
    Embed a 4x4x4 Strassen block into a 4x4x8 decomposition.
    base_k: starting k index (0 or 4)
    """
    for i in range(4):
        for j in range(4):
            for c in range(28):
                if U_4[i*4+j][c] != 0:
                    U_8[i*4+j][offset + c] += U_4[i*4+j][c]
    for j in range(4):
        for k in range(4):
            for c in range(28):
                if V_4[j*4+k][c] != 0:
                    V_8[j*4 + (base_k + k)][offset + c] += V_4[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(28):
                if W_4[k*4+i][c] != 0:
                    W_8[(base_k + k)*4 + i][offset + c] += W_4[k*4+i][c]

def _strassen_4x4x8():
    """
    Build a 4x4x8 decomposition using 2 Strassen 4x4x4 blocks.
    """
    U_4, V_4, W_4 = _strassen_4x4x4()
    
    U_8 = [[0] * 56 for _ in range(16)]
    V_8 = [[0] * 56 for _ in range(32)]
    W_8 = [[0] * 56 for _ in range(32)]
    
    # 2 blocks: k=0..3 and k=4..7
    _embed_4x4x4_in_4x4x8(0, U_4, V_4, W_4, U_8, V_8, W_8, 0)
    _embed_4x4x4_in_4x4x8(4, U_4, V_4, W_4, U_8, V_8, W_8, 28)
    
    return U_8, V_8, W_8

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x8 matrix.
    
    Strategy: Build a 4x4x8 Strassen-based decomposition, then restrict
    to the 3x3x8 sub-tensor by keeping only the relevant rows.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Build the full 4x4x8 decomposition
    U_full, V_full, W_full = _strassen_4x4x8()
    
    # For U: rows indexed by [i*4+j] where i in 0..2, j in 0..2
    # Keep rows where i < 3 and j < 3
    u_rows = []
    for i in range(3):
        for j in range(3):
            u_rows.append(i * 4 + j)
    
    # For V: rows indexed by [j*4+k] where j in 0..2, k in 0..7
    # Keep rows where j < 3
    v_rows = []
    for j in range(3):
        for k in range(8):
            v_rows.append(j * 4 + k)
    
    # For W: rows indexed by [k*4+i] where k in 0..7, i in 0..2
    # Keep rows where i < 3
    w_rows = []
    for k in range(8):
        for i in range(3):
            w_rows.append(k * 4 + i)
    
    # Extract the sub-matrices
    rank = 56
    U = [[U_full[r][c] for c in range(rank)] for r in u_rows]
    V = [[V_full[r][c] for c in range(rank)] for r in v_rows]
    W = [[W_full[r][c] for c in range(rank)] for r in w_rows]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
