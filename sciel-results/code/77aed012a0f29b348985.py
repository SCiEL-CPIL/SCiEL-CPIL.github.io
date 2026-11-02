import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_mults = [
        {
            'x': [(0, 1), (3, 1)],
            'y': [(0, 1), (3, 1)],
            'z': [(0, 1), (3, 1)]
        },
        {
            'x': [(2, 1), (3, 1)],
            'y': [(0, 1)],
            'z': [(2, 1), (3, -1)]
        },
        {
            'x': [(0, 1)],
            'y': [(2, 1), (3, -1)],
            'z': [(1, 1), (3, 1)]
        },
        {
            'x': [(3, 1)],
            'y': [(1, 1), (0, -1)],
            'z': [(0, 1), (2, 1)]
        },
        {
            'x': [(0, 1), (1, 1)],
            'y': [(3, 1)],
            'z': [(0, -1), (1, 1)]
        },
        {
            'x': [(2, 1), (0, -1)],
            'y': [(0, 1), (2, 1)],
            'z': [(3, 1)]
        },
        {
            'x': [(1, 1), (3, -1)],
            'y': [(3, 1), (1, 1)],
            'z': [(0, 1)]
        }
    ]
    
    col_offset = 0
    for r in range(2):
        for c in range(2):
            for s in range(3):
                for mult_idx in range(7):
                    if time.time() > deadline:
                        return {"U": U, "V": V, "W": W}
                        
                    col = col_offset + mult_idx
                    mult = strassen_mults[mult_idx]
                    
                    for local_x, coeff_x in mult['x']:
                        i_loc, j_loc = divmod(local_x, 2)
                        i_global = 2*r + i_loc
                        j_global = 2*c + j_loc
                        u_idx = i_global * m + j_global
                        U[u_idx][col] += coeff_x
                        
                    for local_y, coeff_y in mult['y']:
                        j_loc, k_loc = divmod(local_y, 2)
                        j_global = 2*c + j_loc
                        k_global = 2*s + k_loc
                        v_idx = j_global * p + k_global
                        V[v_idx][col] += coeff_y
                        
                    for local_z, coeff_z in mult['z']:
                        i_loc, k_loc = divmod(local_z, 2)
                        i_global = 2*r + i_loc
                        k_global = 2*s + k_loc
                        w_idx = k_global * n + i_global
                        W[w_idx][col] += coeff_z
                        
                col_offset += 7
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
