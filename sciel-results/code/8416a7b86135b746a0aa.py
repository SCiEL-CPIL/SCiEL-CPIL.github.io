"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 14 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2
    # u, v, w are length 4, indexed by (i, j) or (j, k) or (k, i) flattened
    # u: i*2 + j
    # v: j*2 + k
    # w: k*2 + i
    
    strassen_terms = [
        # P1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # P3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # P4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # P5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # P6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # P7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    col = 0
    for gamma in range(4):
        for alpha in range(2):
            for u_local, v_local, w_local in strassen_terms:
                # u_local indices: i in 0..1, j in 0..1
                # Global i = i_local
                # Global j = 2*alpha + j_local
                # Global k = 2*gamma + k_local
                
                # Fill U
                for i_l in range(2):
                    for j_l in range(2):
                        val = u_local[i_l * 2 + j_l]
                        if val != 0:
                            i_g = i_l
                            j_g = 2 * alpha + j_l
                            U[i_g * m + j_g][col] = val
                
                # Fill V
                for j_l in range(2):
                    for k_l in range(2):
                        val = v_local[j_l * 2 + k_l]
                        if val != 0:
                            j_g = 2 * alpha + j_l
                            k_g = 2 * gamma + k_l
                            V[j_g * p + k_g][col] = val
                
                # Fill W
                for k_l in range(2):
                    for i_l in range(2):
                        val = w_local[k_l * 2 + i_l]
                        if val != 0:
                            k_g = 2 * gamma + k_l
                            i_g = i_l
                            W[k_g * n + i_g][col] = val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
