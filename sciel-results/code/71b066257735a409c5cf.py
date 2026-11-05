import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    n_m = n * m
    m_p = m * p
    p_n = p * n

    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    U_cols = []
    V_cols = []
    W_cols = []

    def check_deadline():
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded during construction")

    def add_term(u_vec, v_vec, w_vec):
        check_deadline()
        u_col = np.zeros(n_m, dtype=float)
        v_col = np.zeros(m_p, dtype=float)
        w_col = np.zeros(p_n, dtype=float)
        for idx, val in u_vec:
            u_col[idx] = float(val)
        for idx, val in v_vec:
            v_col[idx] = float(val)
        for idx, val in w_vec:
            w_col[idx] = float(val)
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def schoolbook_block(i_start, j_start, k_start, i_size, j_size, k_size):
        for i_loc in range(i_size):
            for j_loc in range(j_size):
                for k_loc in range(k_size):
                    check_deadline()
                    i_glob = i_start + i_loc
                    j_glob = j_start + j_loc
                    k_glob = k_start + k_loc
                    I = i_glob * m + j_glob
                    J = j_glob * p + k_glob
                    K = k_glob * n + i_glob
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])

    def strassen_block(i_start, j_start, k_start):
        def gI(i_l, j_l): return (i_start + i_l) * m + (j_start + j_l)
        def gJ(j_l, k_l): return (j_start + j_l) * p + (k_start + k_l)
        def gK(k_l, i_l): return (k_start + k_l) * n + (i_start + i_l)

        # M1 = (A00+A11)(B00+B11)
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        # M2 = (A10+A11)B00
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        # M3 = A00(B01-B11)
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        # M4 = A11(B10-B00)
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        # M5 = (A00+A01)B11
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        # M6 = (A10-A00)(B00+B01)
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        # M7 = (A01-A11)(B10+B11)
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )

    def handle_2x2x3(i_start, j_start, k_start):
        # Use Strassen for the 2x2x2 part
        strassen_block(i_start, j_start, k_start)
        # Schoolbook for the remaining 2x2x1 part (i in 0..1, j in 0..1, k=2)
        for i_loc in range(2):
            for j_loc in range(2):
                i_glob = i_start + i_loc
                j_glob = j_start + j_loc
                k_glob = k_start + 2
                I = i_glob * m + j_glob
                J = j_glob * p + k_glob
                K = k_glob * n + i_glob
                add_term([(I, 1)], [(J, 1)], [(K, 1)])

    def handle_2x3x2(i_start, j_start, k_start):
        # Use Strassen for the 2x2x2 part
        strassen_block(i_start, j_start, k_start)
        # Schoolbook for the remaining 2x1x2 part (i in 0..1, j=2, k in 0..1)
        for i_loc in range(2):
            for k_loc in range(2):
                i_glob = i_start + i_loc
                j_glob = j_start + 2
                k_glob = k_start + k_loc
                I = i_glob * m + j_glob
                J = j_glob * p + k_glob
                K = k_glob * n + i_glob
                add_term([(I, 1)], [(J, 1)], [(K, 1)])

    def handle_2x3x3(i_start, j_start, k_start):
        # Use Strassen for the 2x2x2 part
        strassen_block(i_start, j_start, k_start)
        # Schoolbook for the remaining parts:
        # 1. i in 0..1, j=2, k in 0..2 (2x1x3)
        for i_loc in range(2):
            for k_loc in range(3):
                i_glob = i_start + i_loc
                j_glob = j_start + 2
                k_glob = k_start + k_loc
                I = i_glob * m + j_glob
                J = j_glob * p + k_glob
                K = k_glob * n + i_glob
                add_term([(I, 1)], [(J, 1)], [(K, 1)])
        # 2. i in 0..1, j in 0..1, k=2 (2x2x1)
        for i_loc in range(2):
            for j_loc in range(2):
                i_glob = i_start + i_loc
                j_glob = j_start + j_loc
                k_glob = k_start + 2
                I = i_glob * m + j_glob
                J = j_glob * p + k_glob
                K = k_glob * n + i_glob
                add_term([(I, 1)], [(J, 1)], [(K, 1)])

    # Partition strategy:
    # n=4: split into 2+2
    # m=5: split into 3+2
    # p=5: split into 3+2
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 3), (3, 2)]
    k_blocks = [(0, 3), (3, 2)]

    for i_start, i_size in i_blocks:
        for j_start, j_size in j_blocks:
            for k_start, k_size in k_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    strassen_block(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 2 and k_size == 3:
                    handle_2x2x3(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 3 and k_size == 2:
                    handle_2x3x2(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 3 and k_size == 3:
                    handle_2x3x3(i_start, j_start, k_start)
                else:
                    schoolbook_block(i_start, j_start, k_start, i_size, j_size, k_size)

    if not U_cols:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    rank = U.shape[1]

    # Attempt to reduce rank by identifying and removing linearly dependent terms
    # This is a heuristic pass. We check for exact duplicates first.
    if rank > 0:
        # Check for exact duplicates
        unique_indices = [0]
        for i in range(1, rank):
            is_dup = False
            for j in unique_indices:
                if np.array_equal(U[:, i], U[:, j]) and np.array_equal(V[:, i], V[:, j]) and np.array_equal(W[:, i], W[:, j]):
                    is_dup = True
                    break
            if not is_dup:
                unique_indices.append(i)
        
        if len(unique_indices) < rank:
            U = U[:, unique_indices]
            V = V[:, unique_indices]
            W = W[:, unique_indices]
            rank = U.shape[1]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
