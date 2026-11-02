"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Rank 21 decomposition using Strassen's algorithm for 3 blocks
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for A (U matrix)
    # Indices: 0: A00, 1: A01, 2: A10, 3: A11
    # M1..M7 correspond to k=0..6
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]

    # Strassen coefficients for B (V matrix)
    # Local indices: j in {0,1}, k_local in {0,1}
    # Global k = 2*r + k_local
    # V index = j * p + global_k
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]

    # W_map: for each k (0..6), list of (i, j_local, coeff)
    # C00: M1(1), M4(1), M5(-1), M7(1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M6(1)
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # k=0 (M1) -> C00, C11
        [(1, 0, 1), (1, 1, -1)],  # k=1 (M2) -> C10, C11
        [(0, 1, 1), (1, 1, 1)],   # k=2 (M3) -> C01, C11
        [(0, 0, 1), (1, 0, 1)],   # k=3 (M4) -> C00, C10
        [(0, 0, -1), (0, 1, 1)],  # k=4 (M5) -> C00, C01
        [(1, 1, 1)],              # k=5 (M6) -> C11
        [(0, 0, 1)]               # k=6 (M7) -> C00
    ]

    for r in range(3):
        for k in range(7):
            col = r * 7 + k

            # U
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val

            # V
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                v_idx = j_local * p + global_k
                V[v_idx][col] = val

            # W
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                w_idx = global_k * n + i
                W[w_idx][col] = val

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Verify the decomposition to ensure correctness
    def verify_decomp(U, V, W, n, m, p):
        R = len(U[0]) if U else 0
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for kp in range(p):
                                lhs = 0
                                for r in range(R):
                                    lhs += W[k*n+i][r] * U[ip*m+j][r] * V[jp*p+kp][r]
                                rhs = (1.0 if i==ip and j==jp and k==kp else 0.0)
                                if abs(lhs - rhs) > 1e-9:
                                    return False
        return True

    if not verify_decomp(U, V, W, n, m, p):
        # This should not happen with correct Strassen coefficients, but as a safeguard
        # Return a trivial rank-24 schoolbook solution as fallback
        rank_fallback = n * m * p  # 24
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U_f[i * m + j][col] = 1
                    V_f[j * p + k][col] = 1
                    W_f[k * n + i][col] = 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
