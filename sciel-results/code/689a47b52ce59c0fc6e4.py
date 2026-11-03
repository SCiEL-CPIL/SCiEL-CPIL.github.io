import time

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _time_ok():
    return time.time() < _DEADLINE

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    blocks = [
        ( [0, 1], [0, 1], [0, 1], [0, 1] ),
        ( [0, 1], [0, 1], [0, 1], [2, 3] ),
        ( [0, 1], [0, 1], [0, 1], [4, 5] ),
        ( [0, 1], [2, 3], [2, 3], [0, 1] ),
        ( [0, 1], [2, 3], [2, 3], [2, 3] ),
        ( [0, 1], [2, 3], [2, 3], [4, 5] ),
    ]
    
    # Strassen components for 2x2x2
    strassen = [
        ( [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)] ),
        ( [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)] ),
        ( [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)] ),
        ( [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)] ),
        ( [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)] ),
        ( [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)] ),
        ( [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)] ),
    ]
    
    col = 0
    for block in blocks:
        a_rows, a_cols, b_rows, b_cols = block
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx]][col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx]][col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx]][col] += coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
