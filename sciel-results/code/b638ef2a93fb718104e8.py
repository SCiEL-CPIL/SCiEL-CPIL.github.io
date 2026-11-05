"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    b = 3
    rank = 7 * 27  # 189

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def build_M(u_coeffs, v_coeffs, w_target_blocks):
        nonlocal col
        for li in range(b):
            for lj in range(b):
                for lk in range(b):
                    for (br, bc), cu in u_coeffs.items():
                        i_u = br * b + li
                        j_u = bc * b + lj
                        U[i_u * m + j_u][col] += cu
                    for (bc, br2), cv in v_coeffs.items():
                        j_v = bc * b + lj
                        k_v = br2 * b + lk
                        V[j_v * p + k_v][col] += cv
                    for (br2, br), cw in w_target_blocks.items():
                        k_w = br2 * b + lk
                        i_w = br * b + li
                        W[k_w * n + i_w][col] += cw
                    col += 1

    # M1 = (A22 - A11) * (B11 - B22)
    build_M({(1,1):1, (0,0):-1}, {(0,0):1, (1,1):-1}, {(0,0):-1, (0,1):1})
    # M2 = (A11 + A12) * B22
    build_M({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1})
    # M3 = A11 * (B22 - B21)
    build_M({(0,0):1}, {(1,1):1, (1,0):-1}, {(0,0):1, (1,1):1})
    # M4 = A22 * (B11 - B12)
    build_M({(1,1):1}, {(0,0):1, (0,1):-1}, {(0,0):1, (1,1):1})
    # M5 = (A11 + A22) * (B11 + B22)
    build_M({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,0):1})
    # M6 = (A12 - A22) * (B21 + B22)
    build_M({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,0):1})
    # M7 = (A11 - A21) * (B11 + B12)
    build_M({(0,0):1, (1,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})

    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
