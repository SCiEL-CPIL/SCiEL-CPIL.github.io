"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Block-Strassen: 3x3 grid of 2x2 blocks, Strassen for each 2x2
    # Rank = 27 * 7 = 189
    
    # Strassen coefficients
    U_forms = [
        [1, 0, 0, 1],   # s1: (a+d)
        [0, 0, 1, 1],   # s2: (c+d)
        [1, 0, 0, 0],   # s3: a
        [0, 0, 0, 1],   # s4: d
        [1, 1, 0, 0],   # s5: (a+b)
        [-1, 0, 1, 0],  # s6: (c-a)
        [0, 1, 0, -1],  # s7: (b-d)
    ]
    V_forms = [
        [1, 0, 0, 1],   # s1: (e+h)
        [1, 0, 0, 0],   # s2: e
        [0, 1, 0, -1],  # s3: (f-h)
        [-1, 0, 1, 0],  # s4: (g-e)
        [0, 0, 0, 1],   # s5: h
        [1, 1, 0, 0],   # s6: (e+f)
        [0, 0, 1, 1],   # s7: (g+h)
    ]
    W_forms = [
        [1, 0, 0, 1],   # s1
        [0, 0, 1, -1],  # s2
        [0, 1, 0, 1],   # s3
        [1, 0, 1, 0],   # s4
        [-1, 1, 0, 0],  # s5
        [0, 0, 0, 1],   # s6
        [1, 0, 0, 0],   # s7
    ]
    
    rank = 27 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ib in range(3):
        for jb in range(3):
            for kb in range(3):
                for s in range(7):
                    # A block (ib, jb) entries: [a,b,c,d]
                    A_pos = [(ib*2+di, jb*2+dj) for di in range(2) for dj in range(2)]
                    # But I need the order: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
                    A_pos = [
                        (ib*2+0, jb*2+0),
                        (ib*2+0, jb*2+1),
                        (ib*2+1, jb*2+0),
                        (ib*2+1, jb*2+1),
                    ]
                    for idx in range(4):
                        r, c_a = A_pos[idx]
                        U[r * m + c_a][col] = U_forms[s][idx]
                    
                    # B block (jb, kb) entries: [e,f,g,h]
                    B_pos = [
                        (jb*2+0, kb*2+0),
                        (jb*2+0, kb*2+1),
                        (jb*2+1, kb*2+0),
                        (jb*2+1, kb*2+1),
                    ]
                    for idx in range(4):
                        r_b, c_b = B_pos[idx]
                        V[r_b * p + c_b][col] = V_forms[s][idx]
                    
                    # C block (ib, kb) entries: [C00,C01,C10,C11]
                    # W index: [k*n + i, col]
                    C_pos = [
                        (ib*2+0, kb*2+0),  # C00: i=ib*2+0, k=kb*2+0
                        (ib*2+0, kb*2+1),  # C01: i=ib*2+0, k=kb*2+1
                        (ib*2+1, kb*2+0),  # C10: i=ib*2+1, k=kb*2+0
                        (ib*2+1, kb*2+1),  # C11: i=ib*2+1, k=kb*2+1
                    ]
                    for idx in range(4):
                        i_c, k_c = C_pos[idx]
                        W[k_c * n + i_c][col] = W_forms[s][idx]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
