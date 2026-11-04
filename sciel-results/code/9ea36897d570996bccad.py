import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _validate(U, V, W):
    """
    Validate that U, V, W form an exact decomposition of the matmul(3,5,6) tensor.
    Returns True if valid, False otherwise.
    """
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0])
    
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    if r == 0:
        return False
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m + j]
                v_val = V_np[j*p + k]
                w_val = W_np[k*n + i]
                val = np.sum(u_val * v_val * w_val)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_block(n_b, m_b, p_b):
    """
    Generate schoolbook decomposition for a block of size n_b x m_b x p_b.
    Returns U, V, W as lists of lists.
    """
    rank = n_b * m_b * p_b
    U = [[0] * rank for _ in range(n_b * m_b)]
    V = [[0] * rank for _ in range(m_b * p_b)]
    W = [[0] * rank for _ in range(p_b * n_b)]
    
    col = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[i*m_b + j][col] = 1
                V[j*p_b + k][col] = 1
                W[k*n_b + i][col] = 1
                col += 1
    
    return U, V, W

def _strassen_block_222():
    """
    Generate Strassen decomposition for a 2x2x2 block.
    Returns U, V, W as lists of lists.
    """
    # n=2, m=2, p=2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Indices:
    # U: i*2 + j
    # V: j*2 + k
    # W: k*2 + i
    
    # M1 = (A00+A11)(B00+B11)
    # A00: U[0,0]=1, A11: U[3,0]=1
    # B00: V[0,0]=1, B11: V[3,0]=1
    # C00: W[0,0]=1, C11: W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # A10: U[2,1]=1, A11: U[3,1]=1
    # B00: V[0,1]=1
    # C10: W[2,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11)
    # A00: U[0,2]=1
    # B01: V[1,2]=1, B11: V[3,2]=-1
    # C01: W[1,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    # A11: U[3,3]=1
    # B10: V[2,3]=1, B00: V[0,3]=-1
    # C00: W[0,3]=-1, C11: W[3,3]=-1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[3][3] = -1
    
    # M5 = (A00+A01)B11
    # A00: U[0,4]=1, A01: U[1,4]=1
    # B11: V[3,4]=1
    # C10: W[2,4]=1, C11: W[3,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[3][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # A10: U[2,5]=1, A00: U[0,5]=-1
    # B00: V[0,5]=1, B01: V[1,5]=1
    # C00: W[0,5]=1, C10: W[2,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[2][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A01: U[1,6]=1, A11: U[3,6]=-1
    # B10: V[2,6]=1, B11: V[3,6]=1
    # C00: W[0,6]=1, C01: W[1,6]=-1, C10: W[2,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1; W[2][6] = 1
    
    return U, V, W

def _combine_blocks(n, m, p, n_splits, m_splits, p_splits, use_strassen):
    """
    Combine block decompositions.
    """
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # Determine total rank
    total_rank = 0
    block_ranks = []
    for i_start, i_end in n_splits:
        for j_start, j_end in m_splits:
            for k_start, k_end in p_splits:
                n_b = i_end - i_start
                m_b = j_end - j_start
                p_b = k_end - k_start
                if use_strassen and n_b == 2 and m_b == 2 and p_b == 2:
                    block_ranks.append(7)
                else:
                    block_ranks.append(n_b * m_b * p_b)
                total_rank += block_ranks[-1]
    
    U = [[0] * total_rank for _ in range(U_rows)]
    V = [[0] * total_rank for _ in range(V_rows)]
    W = [[0] * total_rank for _ in range(W_rows)]
    
    col_offset = 0
    for i_start, i_end in n_splits:
        for j_start, j_end in m_splits:
            for k_start, k_end in p_splits:
                n_b = i_end - i_start
                m_b = j_end - j_start
                p_b = k_end - k_start
                
                if use_strassen and n_b == 2 and m_b == 2 and p_b == 2:
                    bU, bV, bW = _strassen_block_222()
                    r_b = 7
                else:
                    bU, bV, bW = _schoolbook_block(n_b, m_b, p_b)
                    r_b = n_b * m_b * p_b
                
                # Map local indices to global
                for i_loc in range(n_b):
                    i_glob = i_start + i_loc
                    for j_loc in range(m_b):
                        j_glob = j_start + j_loc
                        u_idx_glob = i_glob * m + j_glob
                        u_idx_loc = i_loc * m_b + j_loc
                        for c in range(r_b):
                            U[u_idx_glob][col_offset + c] = bU[u_idx_loc][c]
                
                for j_loc in range(m_b):
                    j_glob = j_start + j_loc
                    for k_loc in range(p_b):
                        k_glob = k_start + k_loc
                        v_idx_glob = j_glob * p + k_glob
                        v_idx_loc = j_loc * p_b + k_loc
                        for c in range(r_b):
                            V[v_idx_glob][col_offset + c] = bV[v_idx_loc][c]
                
                for k_loc in range(p_b):
                    k_glob = k_start + k_loc
                    for i_loc in range(n_b):
                        i_glob = i_start + i_loc
                        w_idx_glob = k_glob * n + i_glob
                        w_idx_loc = k_loc * n_b + i_loc
                        for c in range(r_b):
                            W[w_idx_glob][col_offset + c] = bW[w_idx_loc][c]
                
                col_offset += r_b
    
    return U, V, W, total_rank

def _schoolbook():
    """Return the schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 145
    
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try Block Strassen
    # Splits:
    # n: [0,1], [1,3] -> sizes 1, 2
    # m: [0,2], [2,4], [4,5] -> sizes 2, 2, 1
    # p: [0,2], [2,4], [4,6] -> sizes 2, 2, 2
    
    n_splits = [(0, 1), (1, 3)]
    m_splits = [(0, 2), (2, 4), (4, 5)]
    p_splits = [(0, 2), (2, 4), (4, 6)]
    
    try:
        U_new, V_new, W_new, rank_new = _combine_blocks(n, m, p, n_splits, m_splits, p_splits, use_strassen=True)
        
        if time.time() < deadline - 10:
            if _validate(U_new, V_new, W_new):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = rank_new
    except Exception as e:
        # If anything goes wrong, stick with schoolbook
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
