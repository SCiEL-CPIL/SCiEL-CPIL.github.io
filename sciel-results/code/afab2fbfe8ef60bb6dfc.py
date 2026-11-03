"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j, :]
                v = V[j * p + k, :]
                w = W[k * n + i, :]
                val = np.dot(u, v * w)
                if abs(val - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where indices are:
    U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a10-a00)*(b00+b10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01-a11)*(b01+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_block_decomposition(n, m, p):
    """
    Build a block-diagonal decomposition for n×m×p.
    
    Strategy: partition the middle dimension m into blocks of size 2 and 1,
    the row dimension n into blocks of size 2 and 1, and the column dimension
    p into blocks of size 2 and 1. For each combination of (i-block, j-block,
    k-block), use Strassen if both the i-block and k-block have size 2,
    otherwise use schoolbook for that sub-block.
    
    For 3×4×5:
    - n=3: blocks [2, 1] (rows 0-1, row 2)
    - m=4: blocks [2, 2] (cols 0-1, cols 2-3)
    - p=5: blocks [2, 2, 1] (cols 0-1, cols 2-3, col 4)
    
    For each (i_blk, j_blk, k_blk):
    - If i_blk size = 2 and k_blk size = 2: use Strassen (rank 7)
    - Otherwise: use schoolbook for that sub-block
    
    Total rank calculation:
    - i_blk=2 (size 2), j_blk=0 (size 2), k_blk=0 (size 2): Strassen, rank 7
    - i_blk=2 (size 2), j_blk=0 (size 2), k_blk=1 (size 2): Strassen, rank 7
    - i_blk=2 (size 2), j_blk=0 (size 2), k_blk=2 (size 1): schoolbook, rank 2*2*1=4
    - i_blk=2 (size 2), j_blk=1 (size 2), k_blk=0 (size 2): Strassen, rank 7
    - i_blk=2 (size 2), j_blk=1 (size 2), k_blk=1 (size 2): Strassen, rank 7
    - i_blk=2 (size 2), j_blk=1 (size 2), k_blk=2 (size 1): schoolbook, rank 2*2*1=4
    - i_blk=1 (size 1), j_blk=0 (size 2), k_blk=0 (size 2): schoolbook, rank 1*2*2=4
    - i_blk=1 (size 1), j_blk=0 (size 2), k_blk=1 (size 2): schoolbook, rank 1*2*2=4
    - i_blk=1 (size 1), j_blk=0 (size 2), k_blk=2 (size 1): schoolbook, rank 1*2*1=2
    - i_blk=1 (size 1), j_blk=1 (size 2), k_blk=0 (size 2): schoolbook, rank 1*2*2=4
    - i_blk=1 (size 1), j_blk=1 (size 2), k_blk=1 (size 2): schoolbook, rank 1*2*2=4
    - i_blk=1 (size 1), j_blk=1 (size 2), k_blk=2 (size 1): schoolbook, rank 1*2*1=2
    
    Total = 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    su, sv, sw = _strassen_2x2x2()
    
    # Define blocks: (start, size)
    n_blocks = [(0, 2), (2, 1)]  # rows
    m_blocks = [(0, 2), (2, 2)]  # middle
    p_blocks = [(0, 2), (2, 2), (4, 1)]  # cols
    
    for i_start, i_size in n_blocks:
        for j_start, j_size in m_blocks:
            for k_start, k_size in p_blocks:
                if i_size == 2 and k_size == 2:
                    # Use Strassen for 2x2x2 sub-block
                    for r in range(7):
                        u_col = np.zeros(n * m)
                        v_col = np.zeros(m * p)
                        w_col = np.zeros(p * n)
                        
                        for li in range(2):
                            for lj in range(2):
                                gi = i_start + li
                                gj = j_start + lj
                                u_col[gi * m + gj] = su[li * 2 + lj, r]
                        
                        for lj in range(2):
                            for lk in range(2):
                                gj = j_start + lj
                                gk = k_start + lk
                                v_col[gj * p + gk] = sv[lj * 2 + lk, r]
                        
                        for lk in range(2):
                            for li in range(2):
                                gk = k_start + lk
                                gi = i_start + li
                                w_col[gk * n + gi] = sw[lk * 2 + li, r]
                        
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                else:
                    # Use schoolbook for this sub-block
                    for li in range(i_size):
                        for lj in range(j_size):
                            for lk in range(k_size):
                                gi = i_start + li
                                gj = j_start + lj
                                gk = k_start + lk
                                u_col = np.zeros(n * m)
                                v_col = np.zeros(m * p)
                                w_col = np.zeros(p * n)
                                u_col[gi * m + gj] = 1
                                v_col[gj * p + gk] = 1
                                w_col[gk * n + gi] = 1
                                U_cols.append(u_col)
                                V_cols.append(v_col)
                                W_cols.append(w_col)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = U_sb.shape[1]
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try block decomposition
    if time.time() < deadline - 20:
        U_bd, V_bd, W_bd = _build_block_decomposition(n, m, p)
        if _verify(U_bd, V_bd, W_bd, n, m, p, tol=1e-9):
            bd_rank = U_bd.shape[1]
            if bd_rank < best_rank:
                best_rank = bd_rank
                best_result = {"U": U_bd.tolist(), "V": V_bd.tolist(), "W": W_bd.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
