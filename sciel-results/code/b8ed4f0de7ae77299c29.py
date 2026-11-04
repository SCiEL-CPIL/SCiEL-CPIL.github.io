import time
import numpy as np

DIMENSIONS = [2, 4, 6]
_DEADLINE_START = time.time()
_DEADLINE = _DEADLINE_START + 145

def _elapsed():
    return time.time() - _DEADLINE_START

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Attempt to construct a rank 40 decomposition
    # M_{2,4,6} splits into two M_{2,2,6}
    # M_{2,2,6} can be rank 20 (7 for first 2 cols, 13 for next 4)
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Strassen 2x2x2 block helper
    def add_strassen_block(i_start, j_start, k_start, offset):
        # M1
        col = offset
        add_term((i_start+0)*m + (j_start+0), 1, (j_start+0)*p + (k_start+0), 1, (k_start+0)*n + (i_start+0), 1, col)
        add_term((i_start+1)*m + (j_start+1), 1, (j_start+1)*p + (k_start+1), 1, (k_start+0)*n + (i_start+0), 1, col)
        add_term((i_start+1)*m + (j_start+1), 1, (j_start+1)*p + (k_start+1), 1, (k_start+1)*n + (i_start+1), 1, col)
        # M2
        col = offset + 1
        add_term((i_start+1)*m + (j_start+0), 1, (j_start+0)*p + (k_start+0), 1, (k_start+0)*n + (i_start+1), 1, col)
        add_term((i_start+1)*m + (j_start+1), 1, (j_start+0)*p + (k_start+0), 1, (k_start+0)*n + (i_start+1), 1, col)
        add_term((i_start+1)*m + (j_start+1), 1, (j_start+0)*p + (k_start+0), 1, (k_start+1)*n + (i_start+1), -1, col)
        # M3
        col = offset + 2
        add_term((i_start+0)*m + (j_start+0), 1, (j_start+0)*p + (k_start+1), 1, (k_start+1)*n + (i_start+0), 1, col)
        add_term((i_start+0)*m + (j_start+0), 1, (j_start+1)*p + (k_start+1), -1, (k_start+1)*n + (i_start+0), 1, col)
        add_term((i_start+0)*m + (j_start+0), 1, (j_start+1)*p + (k_start+1), -1, (k_start+1)*n + (i_start+1), 1, col)
        # M4
        col = offset + 3
        add_term((i_start+1)*m + (j_start+1), 1, (j_start+1)*p + (k_start+0), 1, (k_start+0)*n + (i_start+0), 1, col)
        add_term((i_start+1)*m + (j_start+1), 1, (j_start+0)*p + (k_start+0), -1, (k_start+0)*n + (i_start+0), 1, col)
        add_term((i_start+1)*m + (j_start+1), 1, (j_start+0)*p + (k_start+0), -1, (k_start+0)*n + (i_start+1), 1, col)
        # M5
        col = offset + 4
        add_term((i_start+0)*m + (j_start+0), 1, (j_start+1)*p + (k_start+1), 1, (k_start+0)*n + (i_start+0), -1, col)
        add_term((i_start+0)*m + (j_start+1), 1, (j_start+1)*p + (k_start+1), 1, (k_start+0)*n + (i_start+0), -1, col)
        add_term((i_start+0)*m + (j_start+1), 1, (j_start+1)*p + (k_start+1), 1, (k_start+1)*n + (i_start+0), 1, col)
        # M6
        col = offset + 5
        add_term((i_start+1)*m + (j_start+0), 1, (j_start+0)*p + (k_start+0), 1, (k_start+1)*n + (i_start+1), 1, col)
        add_term((i_start+0)*m + (j_start+0), -1, (j_start+0)*p + (k_start+0), 1, (k_start+1)*n + (i_start+1), 1, col)
        add_term((i_start+0)*m + (j_start+0), -1, (j_start+0)*p + (k_start+1), 1, (k_start+1)*n + (i_start+1), 1, col)
        add_term((i_start+1)*m + (j_start+0), 1, (j_start+0)*p + (k_start+1), 1, (k_start+1)*n + (i_start+1), 1, col)
        # M7
        col = offset + 6
        add_term((i_start+0)*m + (j_start+1), 1, (j_start+1)*p + (k_start+0), 1, (k_start+0)*n + (i_start+0), 1, col)
        add_term((i_start+1)*m + (j_start+1), -1, (j_start+1)*p + (k_start+0), 1, (k_start+0)*n + (i_start+0), 1, col)
        add_term((i_start+0)*m + (j_start+1), 1, (j_start+1)*p + (k_start+1), 1, (k_start+0)*n + (i_start+0), 1, col)
        add_term((i_start+1)*m + (j_start+1), -1, (j_start+1)*p + (k_start+1), 1, (k_start+0)*n + (i_start+0), 1, col)

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 0, 2, 7)
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6]
    # This is the 3rd pair. We need to save ranks here.
    # For now, use Strassen (7 terms) -> Total 21 for first half.
    add_strassen_block(0, 0, 4, 14)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(0, 2, 0, 21)
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(0, 2, 2, 28)
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6]
    add_strassen_block(0, 2, 4, 35)
    
    # This gives rank 42.
    # To get rank 40, we need to optimize the last two blocks (or any two) to use 6 terms each instead of 7?
    # No, M_{2,2,2} is 7.
    # The savings come from merging across the 3 blocks of the half.
    # Since I cannot easily implement the merge, I return rank 42.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
