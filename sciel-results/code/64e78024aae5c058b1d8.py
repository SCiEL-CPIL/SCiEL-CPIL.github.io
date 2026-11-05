import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0
_START_TIME = time.time()

def _remaining():
    return _DEADLINE - time.time()

def _time_ok(margin=5.0):
    return time.time() - _START_TIME < _DEADLINE - margin

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _strassen_block_2x2():
    """
    Strassen's algorithm for 2x2 * 2x2 multiplication.
    Returns U, V, W factors for the block.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Row indexing: row = i*2 + j for U (A block), row = j*2 + k for V (B block), row = k*2 + i for W (C block)
    """
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (A21+A22)*B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # M3 = A11*(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1 # B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4 = A22*(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1 # B11
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5 = (A11+A12)*B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1 # C11
    W[1, 4] = 1  # C12
    
    # M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1 # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22
    
    # M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1 # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11
    
    return U, V, W

def _build_strassen_based(n, m, p):
    """
    Build a Strassen-based decomposition for 2x5 * 5x5.
    Partition B (5x5) into blocks: rows [0:2, 2:4, 4:5], cols [0:2, 2:4, 4:5]
    Partition A (2x5) into blocks: cols [0:2, 2:4, 4:5]
    Partition C (2x5) into blocks: cols [0:2, 2:4, 4:5]
    
    For each pair of blocks (A_block, B_block), use appropriate multiplication:
    - 2x2 * 2x2: use Strassen (rank 7)
    - 2x1 * 1x2: rank 4 (schoolbook)
    - 1x2 * 2x1: rank 4 (schoolbook)
    - 2x1 * 1x1: rank 2
    - 1x2 * 1x2: rank 4
    - 1x1 * 1x1: rank 1
    - 1x1 * 2x1: rank 2
    - 2x2 * 2x1: rank 6
    - 2x2 * 1x2: rank 6
    - 2x1 * 2x2: rank 6
    - 1x2 * 2x2: rank 6
    - 2x2 * 1x1: rank 2
    - 2x1 * 1x2: rank 4
    - 1x2 * 1x1: rank 2
    - 1x1 * 2x2: rank 4
    - 1x1 * 1x2: rank 2
    """
    # Block boundaries for dimension 5: [0, 2, 4, 5]
    # Block sizes: [2, 2, 1]
    block_sizes = [2, 2, 1]
    block_starts = [0, 2, 4]
    
    # A is 2x5, so A blocks are 2 x block_size[j]
    # B is 5x5, so B blocks are block_size[i] x block_size[k]  
    # C is 2x5, so C blocks are 2 x block_size[k]
    
    U_list = []
    V_list = []
    W_list = []
    
    for j in range(3):  # A column block
        a_cols = block_sizes[j]
        a_start = block_starts[j]
        
        for i in range(3):  # B row block (inner dimension)
            b_rows = block_sizes[i]
            b_start = block_starts[i]
            
            for k in range(3):  # B column block
                b_cols = block_sizes[k]
                b_start_col = block_starts[k]
                c_start = block_starts[k]
                
                # A block: 2 x a_cols (rows 0-1, cols a_start to a_start+a_cols-1)
                # B block: b_rows x b_cols (rows b_start to b_start+b_rows-1, cols b_start_col to b_start_col+b_cols-1)
                # C block: 2 x b_cols (rows 0-1, cols c_start to c_start+b_cols-1)
                
                # Use schoolbook for simplicity, but Strassen for 2x2*2x2
                if a_cols == 2 and b_rows == 2 and b_cols == 2:
                    # Strassen 2x2 * 2x2
                    Us, Vs, Ws = _strassen_block_2x2()
                    # Remap indices
                    U_block = np.zeros((2 * a_cols, 7))
                    V_block = np.zeros((b_rows * b_cols, 7))
                    W_block = np.zeros((2 * b_cols, 7))
                    
                    # U: A is 2 x 2, rows are (i, j) where i in [0,1], j in [0,1]
                    # U row = i * 2 + j
                    for r in range(4):
                        U_block[a_start + r, :] = Us[r, :]
                    
                    # V: B is 2 x 2, rows are (j, k) where j in [0,1], k in [0,1]
                    # V row = j * 2 + k
                    for r in range(4):
                        V_block[b_start + r, :] = Vs[r, :]
                    
                    # W: C is 2 x 2, rows are (k, i) where k in [0,1], i in [0,1]
                    # W row = k * 2 + i
                    for r in range(4):
                        W_block[c_start + r, :] = Ws[r, :]
                    
                    U_list.append(U_block)
                    V_list.append(V_block)
                    W_list.append(W_block)
                else:
                    # Schoolbook for other block sizes
                    rank = 2 * b_rows * b_cols
                    U_block = np.zeros((2 * a_cols, rank))
                    V_block = np.zeros((b_rows * b_cols, rank))
                    W_block = np.zeros((2 * b_cols, rank))
                    
                    c = 0
                    for ii in range(2):
                        for jj in range(b_rows):
                            for kk in range(b_cols):
                                # A element at (ii, a_start + jj)
                                u_idx = ii * a_cols + jj
                                # B element at (b_start + jj, b_start_col + kk)
                                v_idx = jj * b_cols + kk
                                # C element at (kk, ii) in block, global: (c_start + kk, ii)
                                w_idx = kk * 2 + ii
                                
                                U_block[u_idx, c] = 1
                                V_block[v_idx, c] = 1
                                W_block[w_idx, c] = 1
                                c += 1
                    
                    U_list.append(U_block)
                    V_list.append(V_block)
                    W_list.append(W_block)
    
    # Concatenate all blocks
    U_full = np.hstack(U_list)
    V_full = np.hstack(V_list)
    W_full = np.hstack(W_list)
    
    rank = U_full.shape[1]
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}, rank

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-based construction
    if _time_ok(margin=10.0):
        try:
            strassen_result, strassen_rank = _build_strassen_based(n, m, p)
            if strassen_rank < best_rank:
                if _verify(n, m, p, strassen_result['U'], strassen_result['V'], strassen_result['W']):
                    best = strassen_result
                    best_rank = strassen_rank
        except Exception:
            pass
    
    # Try to find rank 49
    if _time_ok(margin=5.0):
        result = _try_rank_49(n, m, p)
        if result is not None:
            if _verify(n, m, p, result['U'], result['V'], result['W']):
                rank_49 = len(result['U'][0])
                if rank_49 < best_rank:
                    return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
