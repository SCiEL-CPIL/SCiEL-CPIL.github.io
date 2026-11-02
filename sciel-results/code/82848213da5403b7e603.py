"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check some invalid entries (spot check)
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-6:
                        return False
    return True

def _strassen_2x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 2x2x2 block at given offsets.
    """
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[offset_a + 0, 0] = 1
    U[offset_a + 2, 0] = 1
    V[offset_b + 0, 0] = 1
    V[offset_b + 2, 0] = 1
    W[offset_c + 0, 0] = 1
    W[offset_c + 2, 0] = 1
    
    # M2 = (A10+A11)B00
    U[offset_a + 1, 1] = 1
    U[offset_a + 2, 1] = 1
    V[offset_b + 0, 1] = 1
    W[offset_c + 1, 1] = 1
    W[offset_c + 2, 1] = 1
    
    # M3 = A00(B01-B11)
    U[offset_a + 0, 2] = 1
    V[offset_b + 1, 2] = 1
    V[offset_b + 2, 2] = -1
    W[offset_c + 1, 2] = 1
    W[offset_c + 2, 2] = 1
    
    # M4 = A11(B10-B00)
    U[offset_a + 2, 3] = 1
    V[offset_b + 2, 3] = 1
    V[offset_b + 0, 3] = -1
    W[offset_c + 1, 3] = 1
    W[offset_c + 2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[offset_a + 0, 4] = 1
    U[offset_a + 1, 4] = 1
    V[offset_b + 2, 4] = 1
    W[offset_c + 1, 4] = 1
    W[offset_c + 2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[offset_a + 1, 5] = 1
    U[offset_a + 0, 5] = -1
    V[offset_b + 0, 5] = 1
    V[offset_b + 1, 5] = 1
    W[offset_c + 1, 5] = 1
    W[offset_c + 2, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[offset_a + 1, 6] = 1
    U[offset_a + 2, 6] = -1
    V[offset_b + 2, 6] = 1
    V[offset_b + 3, 6] = 1
    W[offset_c + 0, 6] = 1
    W[offset_c + 2, 6] = 1
    
    return U, V, W

def _dot_1x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 1x2x2 block at given offsets.
    """
    U = np.zeros((n*m, 4))
    V = np.zeros((m*p, 4))
    W = np.zeros((p*n, 4))
    
    # j=0, k=0
    U[offset_a + 0, 0] = 1
    V[offset_b + 0, 0] = 1
    W[offset_c + 0, 0] = 1
    
    # j=0, k=1
    U[offset_a + 0, 1] = 1
    V[offset_b + 1, 1] = 1
    W[offset_c + 1, 1] = 1
    
    # j=1, k=0
    U[offset_a + 1, 2] = 1
    V[offset_b + 2, 2] = 1
    W[offset_c + 0, 2] = 1
    
    # j=1, k=1
    U[offset_a + 1, 3] = 1
    V[offset_b + 3, 3] = 1
    W[offset_c + 1, 3] = 1
    
    return U, V, W

def _build_decomposition():
    n, m, p = DIMENSIONS
    
    U_total = np.zeros((n*m, 0))
    V_total = np.zeros((m*p, 0))
    W_total = np.zeros((p*n, 0))
    
    def _add_block(U_block, V_block, W_block):
        nonlocal U_total, V_total, W_total
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # Block 1: 1x4x8 (Row 0 of A)
    # Split m=4 into 2+2
    # 1x2x8 (j=0,1)
    #   Split p=8 into 4+4
    #   1x2x4 (k=0..3)
    #     Split p=4 into 2+2
    #     1x2x2 (k=0,1)
    #     1x2x2 (k=2,3)
    #   1x2x4 (k=4..7)
    #     Split p=4 into 2+2
    #     1x2x2 (k=4,5)
    #     1x2x2 (k=6,7)
    # 1x2x8 (j=2,3)
    #   ... similar
    
    # Helper to add 1x2x2 block
    def _add_1x2x2(i, j_start, k_start):
        # i is fixed row of A (0 in this block)
        # j_start is start of j indices (0 or 2)
        # k_start is start of k indices (0, 2, 4, 6)
        off_a = i * m + j_start
        off_b = j_start * p + k_start
        off_c = k_start * n + i
        U_b, V_b, W_b = _dot_1x2x2(off_a, off_b, off_c, n, m, p)
        _add_block(U_b, V_b, W_b)
        
    # Add 1x4x8 blocks for i=0
    for j_start in [0, 2]:
        for k_start in [0, 2, 4, 6]:
            _add_1x2x2(0, j_start, k_start)
            
    # Block 2: 2x4x8 (Rows 1,2 of A)
    # Split n=2 (rows 1,2)
    # Split m=4 into 2+2
    # 2x2x8 (j=0,1)
    #   Split p=8 into 4+4
    #   2x2x4 (k=0..3)
    #     Split p=4 into 2+2
    #     2x2x2 (k=0,1) -> Strassen
    #     2x2x2 (k=2,3) -> Strassen
    #   2x2x4 (k=4..7)
    #     ...
    # 2x2x8 (j=2,3)
    #   ...
    
    # Helper to add 2x2x2 block (Strassen)
    def _add_2x2x2(i_start, j_start, k_start):
        # i_start: 1 or 2 (relative to global, but here i_start is 1 for the 2x block)
        # Actually, the 2x2x2 block covers i in [i_start, i_start+1], j in [j_start, j_start+1], k in [k_start, k_start+1]
        off_a = i_start * m + j_start
        off_b = j_start * p + k_start
        off_c = k_start * n + i_start
        U_b, V_b, W_b = _strassen_2x2x2(off_a, off_b, off_c, n, m, p)
        _add_block(U_b, V_b, W_b)
        
    # Add 2x4x8 blocks for i=1,2
    for j_start in [0, 2]:
        for k_start in [0, 2, 4, 6]:
            _add_2x2x2(1, j_start, k_start)
            
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    return result, U_total.shape[1]

def _schoolbook():
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

def construct():
    n, m, p = DIMENSIONS
    
    try:
        result, rank = _build_decomposition()
        if _verify(result['U'], result['V'], result['W']):
            return result
    except Exception:
        pass
    
    return _schoolbook()
# EVOLVE-BLOCK-END
