"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_strassen():
        total_rank = 32
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        col = 0

        def add_strassen_block(b_col_offset):
            nonlocal col
            terms = [
                ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
                ({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1}),
                ({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
                ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):-1,(1,0):1}),
                ({(0,0):1,(0,1):1}, {(1,1):1}, {(1,0):1,(0,0):1}),
                ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {}),
                ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):-1,(1,1):1}),
            ]
            for u_dict, v_dict, w_dict in terms:
                for (i,j), c in u_dict.items():
                    U[i*m+j][col] += c
                for (j,k_loc), c in v_dict.items():
                    V[j*p + b_col_offset + k_loc][col] += c
                for (k_loc,i), c in w_dict.items():
                    W[(b_col_offset + k_loc)*n + i][col] += c
                col += 1

        def add_trivial_2x2x1(b_col_offset):
            nonlocal col
            for i in range(n):
                for j in range(m):
                    U[i*m+j][col] = 1
                    V[j*p + b_col_offset][col] = 1
                    W[b_col_offset*n + i][col] = 1
                    col += 1

        for blk in range(4):
            add_strassen_block(blk * 2)
        add_trivial_2x2x1(8)
        assert col == total_rank
        return {"U": U, "V": V, "W": W}

    def verify(result):
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    result = try_strassen()
    if verify(result):
        return result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
