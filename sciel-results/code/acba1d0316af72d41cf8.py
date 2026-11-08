"""Block-Strassen rank-28 decomposition for 2x4 @ 4x4."""

def construct():
    n, m, p = 2, 4, 4
    R = 28

    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    def add_strassen(r0, i_off, j_off, k_off):
        U_coeff = {
            0: [(0,0,1),(1,1,1)],
            1: [(1,0,1),(1,1,1)],
            2: [(0,0,1)],
            3: [(1,1,1)],
            4: [(0,0,1),(0,1,1)],
            5: [(1,0,1),(0,0,-1)],
            6: [(0,1,1),(1,1,-1)],
        }
        V_coeff = {
            0: [(0,0,1),(1,1,1)],
            1: [(0,0,1)],
            2: [(1,0,1),(1,1,-1)],
            3: [(0,1,1),(0,0,-1)],
            4: [(1,1,1)],
            5: [(0,0,1),(0,1,1)],
            6: [(1,0,1),(1,1,1)],
        }
        W_coeff = {
            0: [(0,0,1),(1,1,1)],
            1: [(1,0,1),(1,1,-1)],
            2: [(0,1,1),(1,1,1)],
            3: [(0,0,1),(1,0,1)],
            4: [(0,0,-1),(0,1,1)],
            5: [(1,1,1)],
            6: [(0,0,1),(1,1,1)],
        }
        for prod in range(7):
            r = r0 + prod
            for (ii, jj, c) in U_coeff[prod]:
                U[(i_off+ii)*m + (j_off+jj)][r] += c
            for (jj, kk, c) in V_coeff[prod]:
                V[(j_off+jj)*p + (k_off+kk)][r] += c
            for (kk, ii, c) in W_coeff[prod]:
                W[(k_off+kk)*n + (i_off+ii)][r] += c

    add_strassen(0, 0, 0, 0)
    add_strassen(7, 0, 0, 2)
    add_strassen(14, 0, 2, 0)
    add_strassen(21, 0, 2, 2)

    return {"U": U, "V": V, "W": W}
