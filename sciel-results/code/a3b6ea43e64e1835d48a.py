import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44
    R = 44
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Helper to add a Strassen block for 2x2x2
    # A indices: i0, i1 in global i; j0, j1 in global j
    # B indices: j0, j1 in global j; k0, k1 in global k
    # C indices: i0, i1 in global i; k0, k1 in global k
    def add_strassen_block(i0, j0, k0, col_offset):
        # Local indices 0,1 map to global i0, i0+1, etc.
        # Strassen multiplications
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # U vectors (coeffs for A)
        # A = [[a, b], [c, d]]
        # u1 = [[1, 0], [0, 1]] (a+d) -> a:1, d:1
        # u2 = [[0, 0], [1, 1]] (c+d) -> c:1, d:1
        # u3 = [[1, 0], [0, 0]] (a)
        # u4 = [[0, 0], [0, 1]] (d)
        # u5 = [[1, 1], [0, 0]] (a+b)
        # u6 = [[1, 0], [-1, 0]] (c-a) -> a:-1, c:1
        # u7 = [[0, 1], [0, -1]] (b-d) -> b:1, d:-1
        
        # V vectors (coeffs for B)
        # B = [[e, f], [g, h]]
        # v1 = [[1, 1], [1, 1]] (e+h) -> e:1, h:1
        # v2 = [[1, 0], [0, 0]] (e)
        # v3 = [[0, 1], [0, -1]] (f-h) -> f:1, h:-1
        # v4 = [[0, -1], [1, 0]] (g-e) -> e:-1, g:1
        # v5 = [[0, 0], [0, 1]] (h)
        # v6 = [[1, 1], [0, 0]] (e+f) -> e:1, f:1
        # v7 = [[0, 1], [1, 1]] (g+h) -> g:1, h:1
        
        # W vectors (coeffs for C)
        # C = [[c00, c01], [c10, c11]]
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # w1 (for m1): [[1, 0], [0, 1]] -> c00:1, c11:1
        # w2 (for m2): [[0, 0], [1, 0]] -> c10:1
        # w3 (for m3): [[0, 1], [0, 1]] -> c01:1, c11:1
        # w4 (for m4): [[1, 0], [1, 0]] -> c00:1, c10:1
        # w5 (for m5): [[-1, 1], [0, 0]] -> c00:-1, c01:1
        # w6 (for m6): [[0, 0], [0, 1]] -> c11:1
        # w7 (for m7): [[1, 0], [0, 0]] -> c00:1
        
        u_coeffs = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[1, 0], [-1, 0]],
            [[0, 1], [0, -1]]
        ]
        v_coeffs = [
            [[1, 1], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[0, -1], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 1], [1, 1]]
        ]
        w_coeffs = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
            [[-1, 1], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        for t in range(7):
            col = col_offset + t
            # Fill U
            for li in range(2):
                for lj in range(2):
                    val = u_coeffs[t][li][lj]
                    if val != 0:
                        gi = i0 + li
                        gj = j0 + lj
                        U[gi * m + gj][col] += val
                        
            # Fill V
            for lj in range(2):
                for lk in range(2):
                    val = v_coeffs[t][lj][lk]
                    if val != 0:
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] += val
                        
            # Fill W
            for lk in range(2):
                for li in range(2):
                    val = w_coeffs[t][li][lk]
                    if val != 0:
                        gk = k0 + lk
                        gi = i0 + li
                        W[gk * n + gi][col] += val

    # Add 4 Strassen blocks for the 2x4 part (i=0,1)
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0)
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 0, 7)
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 2, 14)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2, 21)
    
    # Add 16 terms for the 1x4 part (i=2)
    # C[2, k] = sum_j A[2, j] B[j, k]
    # Term (j, k): U[2*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+2, col] = 1
    col = 28
    for j in range(4):
        for k in range(4):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
