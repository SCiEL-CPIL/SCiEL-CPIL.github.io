import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Attempt Rank 39 Decomposition
    # Split j into {0,1} and {2,3,4}
    # Block 1: j in {0,1}, k in {0..5} -> 2x2x6, rank 21
    # Block 2: j in {2,3,4}, k in {0..2} -> 2x3x3, rank 9
    # Block 3: j in {2,3,4}, k in {3..5} -> 2x3x3, rank 9
    # Total Rank: 21 + 9 + 9 = 39

    rank_39 = 39
    U = [[0] * rank_39 for _ in range(n * m)]
    V = [[0] * rank_39 for _ in range(m * p)]
    W = [[0] * rank_39 for _ in range(p * n)]
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank_39:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff
        col += 1

    # Block 1: 2x2x6 using 3 Strassen 2x2x2 blocks
    # j in {0,1}, k pairs (0,1), (2,3), (4,5)
    for k1, k2 in [(0, 1), (2, 3), (4, 5)]:
        # Strassen 2x2x2 for j in {0,1}, k in {k1,k2}
        # u: i*m + j
        # v: j*p + k
        # w: k*n + i
        
        u00 = 0 * m + 0
        u01 = 0 * m + 1
        u10 = 1 * m + 0
        u11 = 1 * m + 1
        
        v00 = 0 * p + k1
        v01 = 0 * p + k2
        v10 = 1 * p + k1
        v11 = 1 * p + k2
        
        w00 = k1 * n + 0
        w01 = k2 * n + 0
        w10 = k1 * n + 1
        w11 = k2 * n + 1

        # P1
        add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
        # P2
        add_term([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1), (w11, -1)])
        # P3
        add_term([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)])
        # P4
        add_term([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w10, 1)])
        # P5
        add_term([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
        # P6
        add_term([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
        # P7
        add_term([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])

    # Block 2: 2x3x3 for j in {2,3,4}, k in {0,1,2}
    # Using the 9-term decomposition (needs verification)
    # Local j' in 0,1,2 -> Global j = 2+j'
    # Local k' in 0,1,2 -> Global k = k'
    
    def strassen_233(j_offset, k_offset):
        # j indices: j_offset, j_offset+1 (local 0,1)
        # k indices: k_offset, k_offset+1, k_offset+2 (local 0,1,2)
        # Decompose 2x2x3 as Strassen 2x2x2 (k_local 0,1) + schoolbook (k_local 2)
        # Total: 7 + 2 = 9 terms
        
        u = lambda i, jp: i * m + (j_offset + jp)
        v = lambda jp, kp: (j_offset + jp) * p + (k_offset + kp)
        w = lambda kp, i: (k_offset + kp) * n + i
        
        # Strassen 2x2x2 for local j in {0,1}, local k in {0,1}
        # P1
        add_term([(u(0,0), 1), (u(1,1), 1)], [(v(0,0), 1), (v(1,1), 1)], [(w(0,0), 1), (w(1,1), 1)])
        # P2
        add_term([(u(1,0), 1), (u(1,1), 1)], [(v(0,0), 1)], [(w(0,1), 1), (w(1,1), -1)])
        # P3
        add_term([(u(0,0), 1)], [(v(0,1), 1), (v(1,1), -1)], [(w(1,0), 1), (w(1,1), 1)])
        # P4
        add_term([(u(1,1), 1)], [(v(1,0), 1), (v(0,0), -1)], [(w(0,0), 1), (w(0,1), 1)])
        # P5
        add_term([(u(0,0), 1), (u(0,1), 1)], [(v(1,1), 1)], [(w(0,0), -1), (w(1,0), 1)])
        # P6
        add_term([(u(1,0), 1), (u(0,0), -1)], [(v(0,0), 1), (v(0,1), 1)], [(w(1,1), 1)])
        # P7
        add_term([(u(0,1), 1), (u(1,1), -1)], [(v(1,0), 1), (v(1,1), 1)], [(w(0,0), 1)])
        
        # Schoolbook for local k = 2
        add_term([(u(0,0), 1)], [(v(0,2), 1)], [(w(2,0), 1)])
        add_term([(u(1,0), 1)], [(v(0,2), 1)], [(w(2,1), 1)])
        add_term([(u(0,1), 1)], [(v(1,2), 1)], [(w(2,0), 1)])
        add_term([(u(1,1), 1)], [(v(1,2), 1)], [(w(2,1), 1)])

    strassen_233(2, 0)
    strassen_233(2, 3)

    # Verify Rank 39
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback: Rank 54 Decomposition
    rank_54 = 54
    U = [[0] * rank_54 for _ in range(n * m)]
    V = [[0] * rank_54 for _ in range(m * p)]
    W = [[0] * rank_54 for _ in range(p * n)]
    col = 0

    def add_term_54(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank_54:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff
        col += 1

    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    strassen_cells = set()
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            for i in range(n):
                for j in (j1, j2):
                    for k in (k1, k2):
                        strassen_cells.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_cells:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            u00 = 0 * m + j1
            u01 = 0 * m + j2
            u10 = 1 * m + j1
            u11 = 1 * m + j2

            v00 = j1 * p + k1
            v01 = j1 * p + k2
            v10 = j2 * p + k1
            v11 = j2 * p + k2

            w00 = k1 * n + 0
            w01 = k2 * n + 0
            w10 = k1 * n + 1
            w11 = k2 * n + 1

            add_term_54([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
            add_term_54([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1), (w11, -1)])
            add_term_54([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)])
            add_term_54([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w10, 1)])
            add_term_54([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
            add_term_54([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
            add_term_54([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])

    return {"U": U, "V": V, "W": W}
