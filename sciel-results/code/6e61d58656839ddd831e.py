"""Combined approach: block Strassen (2x2x2) with optimized small blocks for 3x5x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7) indexed by [i*2+a]
    V: (4, 7) indexed by [b*2+k]
    W: (4, 7) indexed by [k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0 (M0): (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1 (M1): (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # c=2 (M2): A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # c=3 (M3): A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[0, 3] = -1
    
    # c=4 (M4): (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1
    
    # c=5 (M5): (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    
    # c=6 (M6): (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W

def _block_2x2x1():
    """Return (U, V, W) for 2x2x1 block. Rank 2."""
    U = np.zeros((4, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    
    # c=0: a=0, b=0
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1
    
    # c=1: a=1, b=1
    U[1, 1] = 1; U[3, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1; W[1, 1] = 1
    
    return U, V, W

def _block_2x1x2():
    """Return (U, V, W) for 2x1x2 block. Rank 1."""
    U = np.ones((2, 1))
    V = np.ones((2, 1))
    W = np.ones((2, 1))
    return U, V, W

def _block_2x1x1():
    """Return (U, V, W) for 2x1x1 block. Rank 1."""
    U = np.ones((2, 1))
    V = np.ones((1, 1))
    W = np.ones((2, 1))
    return U, V, W

def _block_1x2x2():
    """Return (U, V, W) for 1x2x2 block. Rank 2."""
    U = np.zeros((2, 2))
    V = np.zeros((4, 2))
    W = np.zeros((2, 2))
    
    # c=0: a=0, b=0
    U[0, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1
    
    # c=1: a=1, b=1
    U[1, 1] = 1
    V[2, 1] = 1; V[3, 1] = 1
    W[0, 1] = 1; W[1, 1] = 1
    
    return U, V, W

def _block_1x2x1():
    """Return (U, V, W) for 1x2x1 block. Rank 2."""
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    
    # c=0: a=0, b=0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # c=1: a=1, b=1
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    return U, V, W

def _block_1x1x2():
    """Return (U, V, W) for 1x1x2 block. Rank 1."""
    U = np.ones((1, 1))
    V = np.ones((2, 1))
    W = np.ones((2, 1))
    return U, V, W

def _block_1x1x1():
    """Return (U, V, W) for 1x1x1 block. Rank 1."""
    U = np.ones((1, 1))
    V = np.ones((1, 1))
    W = np.ones((1, 1))
    return U, V, W

def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    """
    Check that the factor matrices define the matrix multiplication tensor.
    Constraint: for all i in [0,n), a in [0,m), b in [0,m), k in [0,p):
      sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c] = (1 if a==b else 0)
    """
    rank = len(U[0])
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + a][c] * V[b * p + k][c]
                    expected = 1.0 if a == b else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def _construct_optimized_block_decomposition(n, m, p):
    """
    Construct a decomposition using block Strassen for 2x2x2 sub-blocks
    and minimal rank decompositions for remaining blocks.
    """
    # Partitions
    I_parts = [2, 1]
    J_parts = [2, 2, 1]
    K_parts = [2, 2, 2, 1]
    
    I_starts = []
    pos = 0
    for s in I_parts:
        I_starts.append(pos)
        pos += s
    J_starts = []
    pos = 0
    for s in J_parts:
        J_starts.append(pos)
        pos += s
    K_starts = []
    pos = 0
    for s in K_parts:
        K_starts.append(pos)
        pos += s
    
    # Precompute block factors
    S_U, S_V, S_W = _strassen_2x2x2()
    B221_U, B221_V, B221_W = _block_2x2x1()
    B212_U, B212_V, B212_W = _block_2x1x2()
    B211_U, B211_V, B211_W = _block_2x1x1()
    B122_U, B122_V, B122_W = _block_1x2x2()
    B121_U, B121_V, B121_W = _block_1x2x1()
    B112_U, B112_V, B112_W = _block_1x1x2()
    B111_U, B111_V, B111_W = _block_1x1x1()
    
    # Calculate total rank
    total_rank = 0
    for ii in range(len(I_parts)):
        for jj in range(len(J_parts)):
            for kk in range(len(K_parts)):
                i_size = I_parts[ii]
                j_size = J_parts[jj]
                k_size = K_parts[kk]
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    total_rank += 7
                elif i_size == 2 and j_size == 2 and k_size == 1:
                    total_rank += 2
                elif i_size == 2 and j_size == 1 and k_size == 2:
                    total_rank += 1
                elif i_size == 2 and j_size == 1 and k_size == 1:
                    total_rank += 1
                elif i_size == 1 and j_size == 2 and k_size == 2:
                    total_rank += 2
                elif i_size == 1 and j_size == 2 and k_size == 1:
                    total_rank += 2
                elif i_size == 1 and j_size == 1 and k_size == 2:
                    total_rank += 1
                elif i_size == 1 and j_size == 1 and k_size == 1:
                    total_rank += 1
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ii in range(len(I_parts)):
        for jj in range(len(J_parts)):
            for kk in range(len(K_parts)):
                i_size = I_parts[ii]
                j_size = J_parts[jj]
                k_size = K_parts[kk]
                i_start = I_starts[ii]
                j_start = J_starts[jj]
                k_start = K_starts[kk]
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Strassen
                    for local_i in range(2):
                        for local_a in range(2):
                            gi = i_start + local_i
                            ga = j_start + local_a
                            u_idx = gi * m + ga
                            for c in range(7):
                                U[u_idx][col + c] = S_U[local_i * 2 + local_a][c]
                    for local_b in range(2):
                        for local_k in range(2):
                            gb = j_start + local_b
                            gk = k_start + local_k
                            v_idx = gb * p + gk
                            for c in range(7):
                                V[v_idx][col + c] = S_V[local_b * 2 + local_k][c]
                    for local_k in range(2):
                        for local_i in range(2):
                            gk = k_start + local_k
                            gi = i_start + local_i
                            w_idx = gk * n + gi
                            for c in range(7):
                                W[w_idx][col + c] = S_W[local_k * 2 + local_i][c]
                    col += 7
                elif i_size == 2 and j_size == 2 and k_size == 1:
                    # 2x2x1
                    for local_i in range(2):
                        for local_a in range(2):
                            gi = i_start + local_i
                            ga = j_start + local_a
                            u_idx = gi * m + ga
                            for c in range(2):
                                U[u_idx][col + c] = B221_U[local_i * 2 + local_a][c]
                    for local_b in range(2):
                        local_k = 0
                        gb = j_start + local_b
                        gk = k_start + local_k
                        v_idx = gb * p + gk
                        for c in range(2):
                            V[v_idx][col + c] = B221_V[local_b][c]
                    for local_k in range(1):
                        for local_i in range(2):
                            gk = k_start + local_k
                            gi = i_start + local_i
                            w_idx = gk * n + gi
                            for c in range(2):
                                W[w_idx][col + c] = B221_W[local_k * 2 + local_i][c]
                    col += 2
                elif i_size == 2 and j_size == 1 and k_size == 2:
                    # 2x1x2
                    for local_i in range(2):
                        local_a = 0
                        gi = i_start + local_i
                        ga = j_start + local_a
                        u_idx = gi * m + ga
                        for c in range(1):
                            U[u_idx][col + c] = B212_U[local_i][c]
                    for local_b in range(1):
                        for local_k in range(2):
                            gb = j_start + local_b
                            gk = k_start + local_k
                            v_idx = gb * p + gk
                            for c in range(1):
                                V[v_idx][col + c] = B212_V[local_b * 2 + local_k][c]
                    for local_k in range(2):
                        for local_i in range(2):
                            gk = k_start + local_k
                            gi = i_start + local_i
                            w_idx = gk * n + gi
                            for c in range(1):
                                W[w_idx][col + c] = B212_W[local_k * 2 + local_i][c]
                    col += 1
                elif i_size == 2 and j_size == 1 and k_size == 1:
                    # 2x1x1
                    for local_i in range(2):
                        local_a = 0
                        gi = i_start + local_i
                        ga = j_start + local_a
                        u_idx = gi * m + ga
                        for c in range(1):
                            U[u_idx][col + c] = B211_U[local_i][c]
                    local_b = 0
                    local_k = 0
                    gb = j_start + local_b
                    gk = k_start + local_k
                    v_idx = gb * p + gk
                    for c in range(1):
                        V[v_idx][col + c] = B211_V[local_b][c]
                    for local_i in range(2):
                        gk = k_start + local_k
                        gi = i_start + local_i
                        w_idx = gk * n + gi
                        for c in range(1):
                            W[w_idx][col + c] = B211_W[local_i][c]
                    col += 1
                elif i_size == 1 and j_size == 2 and k_size == 2:
                    # 1x2x2
                    local_i = 0
                    for local_a in range(2):
                        gi = i_start + local_i
                        ga = j_start + local_a
                        u_idx = gi * m + ga
                        for c in range(2):
                            U[u_idx][col + c] = B122_U[local_a][c]
                    for local_b in range(2):
                        for local_k in range(2):
                            gb = j_start + local_b
                            gk = k_start + local_k
                            v_idx = gb * p + gk
                            for c in range(2):
                                V[v_idx][col + c] = B122_V[local_b * 2 + local_k][c]
                    for local_k in range(2):
                        gi = i_start + local_i
                        gk = k_start + local_k
                        w_idx = gk * n + gi
                        for c in range(2):
                            W[w_idx][col + c] = B122_W[local_k][c]
                    col += 2
                elif i_size == 1 and j_size == 2 and k_size == 1:
                    # 1x2x1
                    local_i = 0
                    for local_a in range(2):
                        gi = i_start + local_i
                        ga = j_start + local_a
                        u_idx = gi * m + ga
                        for c in range(2):
                            U[u_idx][col + c] = B121_U[local_a][c]
                    for local_b in range(2):
                        local_k = 0
                        gb = j_start + local_b
                        gk = k_start + local_k
                        v_idx = gb * p + gk
                        for c in range(2):
                            V[v_idx][col + c] = B121_V[local_b][c]
                    gi = i_start + local_i
                    gk = k_start + local_k
                    w_idx = gk * n + gi
                    for c in range(2):
                        W[w_idx][col + c] = B121_W[0][c]
                    col += 2
                elif i_size == 1 and j_size == 1 and k_size == 2:
                    # 1x1x2
                    local_i = 0
                    local_a = 0
                    gi = i_start + local_i
                    ga = j_start + local_a
                    u_idx = gi * m + ga
                    for c in range(1):
                        U[u_idx][col + c] = B112_U[0][c]
                    local_b = 0
                    for local_k in range(2):
                        gb = j_start + local_b
                        gk = k_start + local_k
                        v_idx = gb * p + gk
                        for c in range(1):
                            V[v_idx][col + c] = B112_V[local_k][c]
                    for local_k in range(2):
                        gi = i_start + local_i
                        gk = k_start + local_k
                        w_idx = gk * n + gi
                        for c in range(1):
                            W[w_idx][col + c] = B112_W[local_k][c]
                    col += 1
                elif i_size == 1 and j_size == 1 and k_size == 1:
                    # 1x1x1
                    local_i = 0
                    local_a = 0
                    gi = i_start + local_i
                    ga = j_start + local_a
                    u_idx = gi * m + ga
                    U[u_idx][col] = B111_U[0][0]
                    local_b = 0
                    local_k = 0
                    gb = j_start + local_b
                    gk = k_start + local_k
                    v_idx = gb * p + gk
                    V[v_idx][col] = B111_V[0][0]
                    gi = i_start + local_i
                    gk = k_start + local_k
                    w_idx = gk * n + gi
                    W[w_idx][col] = B111_W[0][0]
                    col += 1
    
    # Validate
    if not _validate_decomposition(U, V, W, n, m, p):
        return None
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5 by 7 matrix.
    
    Returns: {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    result = _construct_optimized_block_decomposition(n, m, p)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
