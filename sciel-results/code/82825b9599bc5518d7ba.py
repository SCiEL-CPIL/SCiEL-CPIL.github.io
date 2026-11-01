"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # U_local: 4x7, V_local: 4x7, W_local: 4x7
    # Indices: U: i*m'+j', V: j'*p'+k', W: k'*n'+i'
    # m'=2, p'=2, n'=2
    
    U_local = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    V_local = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, -1, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    W_local = [
        [1, 0, 0, 0],  # M1 -> Z00
        [0, 1, 0, 0],  # M2 -> Z10
        [0, 0, 1, 0],  # M3 -> Z01
        [0, 1, 0, 0],  # M4 -> Z10
        [-1, 0, 1, 0], # M5 -> -Z00 + Z01
        [0, 0, 0, 1],  # M6 -> Z11
        [1, 0, 0, 0]   # M7 -> Z00
    ]
    
    # Blocks: (j_offset, k_offset)
    # A is 2x4, B is 4x4
    # A1: cols 0,1. A2: cols 2,3
    # B11: rows 0,1 cols 0,1. B12: rows 0,1 cols 2,3
    # B21: rows 2,3 cols 0,1. B22: rows 2,3 cols 2,3
    blocks = [
        (0, 0), # A1, B11
        (0, 2), # A1, B12
        (2, 0), # A2, B21
        (2, 2)  # A2, B22
    ]
    
    for b_idx, (j_off, k_off) in enumerate(blocks):
        for t_idx in range(7):
            c = b_idx * 7 + t_idx
            
            # Fill U
            for i_prime in range(2):
                for j_prime in range(2):
                    u_val = U_local[t_idx][i_prime * 2 + j_prime]
                    if u_val != 0:
                        i_global = i_prime
                        j_global = j_off + j_prime
                        U[i_global * m + j_global][c] += u_val
                        
            # Fill V
            for j_prime in range(2):
                for k_prime in range(2):
                    v_val = V_local[t_idx][j_prime * 2 + k_prime]
                    if v_val != 0:
                        j_global = j_off + j_prime
                        k_global = k_off + k_prime
                        V[j_global * p + k_global][c] += v_val
                        
            # Fill W
            for k_prime in range(2):
                for i_prime in range(2):
                    w_val = W_local[t_idx][k_prime * 2 + i_prime]
                    if w_val != 0:
                        k_global = k_off + k_prime
                        i_global = i_prime
                        W[k_global * n + i_global][c] += w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
