"""Combined program: Strassen-based rank-36 decomposition for 2x4x5 matmul."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140.0  # 140s deadline, 10s safety margin

    n, m, p = DIMENSIONS

    # Block structure:
    #   m=4 -> two 2-blocks (q in 0,1)
    #   p=5 -> two 2-blocks (s in 0,1) + one 1-block (s=2)
    # For each (q, s):
    #   s<2: 2x2x2 block -> Strassen rank 7
    #   s==2: 2x2x1 block -> schoolbook rank 4
    # Total rank = 2 * (2*7 + 4) = 36

    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] += val
        col += 1

    # Strassen 2x2x2 coefficients (local indices 0..3 for 2x2)
    # A: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
    # B: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
    # C: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),          # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),          # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),          # M7
    ]

    # Schoolbook 2x2x1 coefficients
    # A: 4 entries (2x2), B: 2 entries (2x1), C: 2 entries (2x1)
    standard_terms_2x2x1 = [
        ({0: 1}, {0: 1}, {0: 1}),  # A00 * B0 -> C0
        ({1: 1}, {1: 1}, {0: 1}),  # A01 * B1 -> C0
        ({2: 1}, {0: 1}, {1: 1}),  # A10 * B0 -> C1
        ({3: 1}, {1: 1}, {1: 1}),  # A11 * B1 -> C1
    ]

    for q in range(2):  # block in m
        for s in range(3):  # block in p
            if s < 2:
                # 2x2x2 Strassen block
                # k_global = 2*s + lk
                for u_t, v_t, w_t in strassen_terms:
                    u_global = {}
                    v_global = {}
                    w_global = {}

                    for idx, val in u_t.items():
                        li = idx // 2
                        lj = idx % 2
                        gi = li
                        gj = 2 * q + lj
                        u_global[gi * m + gj] = val

                    for idx, val in v_t.items():
                        lj = idx // 2
                        lk = idx % 2
                        gj = 2 * q + lj
                        gk = 2 * s + lk
                        v_global[gj * p + gk] = val

                    for idx, val in w_t.items():
                        li = idx // 2
                        lk = idx % 2
                        gi = li
                        gk = 2 * s + lk
                        w_global[gk * n + gi] = val

                    add_term(u_global, v_global, w_global)
            else:
                # 2x2x1 schoolbook block, k_global = 4
                for u_t, v_t, w_t in standard_terms_2x2x1:
                    u_global = {}
                    v_global = {}
                    w_global = {}

                    for idx, val in u_t.items():
                        li = idx // 2
                        lj = idx % 2
                        gi = li
                        gj = 2 * q + lj
                        u_global[gi * m + gj] = val

                    for idx, val in v_t.items():
                        lj = idx
                        gk = 4
                        gj = 2 * q + lj
                        v_global[gj * p + gk] = val

                    for idx, val in w_t.items():
                        li = idx
                        gk = 4
                        gi = li
                        w_global[gk * n + gi] = val

                    add_term(u_global, v_global, w_global)

    # Safety: check deadline
    if time.time() > _deadline:
        pass  # construction is O(1), we are done

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
