import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    for u_idx in range(n * m):
        i = u_idx // m
        j = u_idx % m
        for v_idx in range(m * p):
            j2 = v_idx // p
            k = v_idx % p
            if j != j2:
                continue
            for w_idx in range(p * n):
                k2 = w_idx // n
                i2 = w_idx % n
                val = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(rank))
                if i == i2 and j == j2 and k == k2:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    # A is 2x4, B is 4x6.
    # Split A into A_left (cols 0-1) and A_right (cols 2-3).
    # Split B into B_top (rows 0-1) and B_bot (rows 2-3).
    # The multiplication is A_left * B_top + A_right * B_bot.
    # These two terms have disjoint support in the middle dimension j (0,1 vs 2,3).
    # Thus the rank is the sum of the ranks of the two sub-problems.
    # Each sub-problem is M_{2,2,6}.
    # We decompose M_{2,2,6} into three M_{2,2,2} Strassen blocks.
    # Each M_{2,2,2} has rank 7.
    # Total rank per M_{2,2,6} is 21.
    # Total rank for M_{2,4,6} is 42.
    
    blocks = [
        # (a_rows, a_cols, b_rows, b_cols)
        # Block 0: A_left * B_top_left (cols 0-1 of B_top) -> M_{2,2,2}
        ( [0, 1], [0, 1], [0, 1], [0, 1] ),
        # Block 1: A_left * B_top_mid (cols 2-3 of B_top) -> M_{2,2,2}
        ( [0, 1], [0, 1], [0, 1], [2, 3] ),
        # Block 2: A_left * B_top_right (cols 4-5 of B_top) -> M_{2,2,2}
        ( [0, 1], [0, 1], [0, 1], [4, 5] ),
        # Block 3: A_right * B_bot_left (cols 0-1 of B_bot) -> M_{2,2,2}
        ( [0, 1], [2, 3], [2, 3], [0, 1] ),
        # Block 4: A_right * B_bot_mid (cols 2-3 of B_bot) -> M_{2,2,2}
        ( [0, 1], [2, 3], [2, 3], [2, 3] ),
        # Block 5: A_right * B_bot_right (cols 4-5 of B_bot) -> M_{2,2,2}
        ( [0, 1], [2, 3], [2, 3], [4, 5] ),
    ]
    
    # Strassen components for M_{2,2,2}
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b11)
    # M4 = a22(b12-b22)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # Indices in local 2x2 matrix:
    # 0: 0,0   1: 0,1
    # 2: 1,0   3: 1,1
    
    strassen = [
        # M1
        ( [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)] ),
        # M2
        ( [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)] ),
        # M3
        ( [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)] ),
        # M4
        ( [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)] ),
        # M5
        ( [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)] ),
        # M6
        ( [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)] ),
        # M7
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
        
        # W index: k * n + i
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                global_idx = a_map[local_idx]
                U[global_idx][col] += coeff
            
            for local_idx, coeff in v_coeffs:
                global_idx = b_map[local_idx]
                V[global_idx][col] += coeff
            
            for local_idx, coeff in w_coeffs:
                global_idx = w_map[local_idx]
                W[global_idx][col] += coeff
            
            col += 1
            
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Invalid decomposition")
    return {"U": U, "V": V, "W": W}
