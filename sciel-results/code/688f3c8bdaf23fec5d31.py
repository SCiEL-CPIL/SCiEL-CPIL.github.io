import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block-Strassen decomposition
    # 6x6 matrices partitioned into 3x3 blocks of 2x2 matrices
    # 27 block products, each decomposed into 7 scalar multiplications
    rank = 27 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Block A_ik, B_kj, C_ij
                # Strassen terms for 2x2 matrices
                # u: [A00, A01, A10, A11]
                # v: [B00, B01, B10, B11]
                # w: [[C00, C01], [C10, C11]]
                
                terms = [
                    # (u, v, w)
                    ([1, 0, 0, 1], [1, 0, 0, 1], [[1, 0], [0, 1]]),    # M1
                    ([0, 0, 1, 1], [1, 0, 0, 0], [[0, 0], [1, -1]]),   # M2
                    ([1, 0, 0, 0], [0, 1, 0, -1], [[0, 1], [1, 0]]),    # M3
                    ([0, 0, 0, 1], [0, 1, -1, 0], [[1, 0], [1, 0]]),    # M4
                    ([1, 1, 0, 0], [0, 0, 0, 1], [[-1, 1], [0, 0]]),    # M5
                    ([0, 0, 1, -1], [1, 1, 0, 0], [[0, 0], [0, 1]]),    # M6
                    ([0, 1, -1, 0], [0, 0, 1, 1], [[1, 0], [0, 0]])     # M7
                ]
                
                for u, v, w in terms:
                    # A_ik elements
                    # A_ik[a,b] is at row 2i+a, col 2k+b
                    for a in range(2):
                        for b in range(2):
                            if u[a*2+b] != 0:
                                x = (2*i + a) * 6 + (2*k + b)
                                U[x][col] += u[a*2+b]
                    
                    # B_kj elements
                    # B_kj[c,d] is at row 2k+c, col 2j+d
                    for c in range(2):
                        for d in range(2):
                            if v[c*2+d] != 0:
                                y = (2*k + c) * 6 + (2*j + d)
                                V[y][col] += v[c*2+d]
                    
                    # C_ij elements
                    # C_ij[e,f] is at row 2i+e, col 2j+f
                    for e in range(2):
                        for f in range(2):
                            if w[e][f] != 0:
                                z = (2*i + e) * 6 + (2*j + f)
                                W[z][col] += w[e][f]
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
