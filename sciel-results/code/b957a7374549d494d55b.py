"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 42)
    
    # Strassen coefficients for 2x2x2
    U_strassen = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    V_strassen = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    W_strassen = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]
    
    col = 0
    # m_idx corresponds to splitting A into A1 (cols 0-1) and A2 (cols 2-3)
    # and B into B1 (rows 0-1) and B2 (rows 2-3)
    for m_idx in range(2):
        # k_idx corresponds to splitting the 6 columns of B_block into three 2-col blocks
        for k_idx in range(3):
            # Perform Strassen for 2x2 A_block * 2x2 B_block
            for s in range(7):
                # U: A_block indices (i, j_local)
                # Global A: row i, col (2*m_idx + j_local)
                for (i_loc, j_loc), coeff in U_strassen[s].items():
                    i = i_loc
                    j = m_idx * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # V: B_block indices (j_local, k_local)
                # Global B: row (2*m_idx + j_local), col (2*k_idx + k_local)
                for (j_loc, k_loc), coeff in V_strassen[s].items():
                    j = m_idx * 2 + j_loc
                    k = k_idx * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # W: C_block indices (i, k_local)
                # Global C: row (2*k_idx + k_local), col i
                # Note: W index is k*n + i. Here C is 2x6.
                # C row index is global k. C col index is i.
                for (i_loc, k_loc), coeff in W_strassen[s].items():
                    i = i_loc
                    k = k_idx * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                col += 1
    
    if time.time() - start_time > 130:
        raise TimeoutError()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
