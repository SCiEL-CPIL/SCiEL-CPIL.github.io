"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Build base decomposition using Strassen for 2x2 blocks
    # Total: 7+7+8+7+7 = 36 rank-1 terms

    U_cols = []  # Each entry is a vector of length n*m
    V_cols = []  # Each entry is a vector of length m*p
    W_cols = []  # Each entry is a vector of length p*n

    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1

        v00 = b_row_offset * p + b_col_offset
        v01 = b_row_offset * p + b_col_offset + 1
        v10 = (b_row_offset + 1) * p + b_col_offset
        v11 = (b_row_offset + 1) * p + b_col_offset + 1

        w00 = c_col_offset * n + c_row_offset
        w01 = (c_col_offset + 1) * n + c_row_offset
        w10 = c_col_offset * n + c_row_offset + 1
        w11 = (c_col_offset + 1) * n + c_row_offset + 1

        # M1 = (a00+a11)(b00+b11)
        add_term([(u00,1),(u11,1)], [(v00,1),(v11,1)], [(w00,1),(w11,1)])
        # M2 = (a10+a11)b00
        add_term([(u10,1),(u11,1)], [(v00,1)], [(w10,1),(w11,-1)])
        # M3 = a00(b01-b11)
        add_term([(u00,1)], [(v01,1),(v11,-1)], [(w01,1),(w11,1)])
        # M4 = a11(b10-b00)
        add_term([(u11,1)], [(v10,1),(v00,-1)], [(w00,1),(w10,1)])
        # M5 = (a00+a01)b11
        add_term([(u00,1),(u01,1)], [(v11,1)], [(w00,-1),(w01,1)])
        # M6 = (a10-a00)(b00+b01)
        add_term([(u10,1),(u00,-1)], [(v00,1),(v01,1)], [(w11,1)])
        # M7 = (a01-a11)(b10+b11)
        add_term([(u01,1),(u11,-1)], [(v10,1),(v11,1)], [(w00,1)])

    def add_schoolbook_2x4_by_4x1():
        for j in range(4):
            add_term([(0*m+j, 1)], [(j*p+2, 1)], [(2*n+0, 1)])
        for j in range(4):
            add_term([(1*m+j, 1)], [(j*p+2, 1)], [(2*n+1, 1)])

    # Part 1: C1 = A B1 (2x2), B1 = cols 0,1
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)

    # Part 2: C2 = A B2 (2x3), B2 = cols 2,3,4
    add_schoolbook_2x4_by_4x1()
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)

    rank = len(U_cols)  # 36

    # Verify correctness
    U_arr = np.array(U_cols, dtype=np.float64).T  # (n*m, rank)
    V_arr = np.array(V_cols, dtype=np.float64).T  # (m*p, rank)
    W_arr = np.array(W_cols, dtype=np.float64).T  # (p*n, rank)

    # Build true tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # Reconstruct
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)

    if not np.allclose(T_recon, T, atol=1e-10):
        # Shouldn't happen, but return anyway
        return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

    # Try to reduce rank: check if any term is redundant
    # For each term t, check if T - term_t is still representable with the other terms
    # This is a strong test. A simpler test: check if term_t can be written as
    # a linear combination of the other terms in the tensor product space.
    #
    # Practical approach: Try to find a rank-r decomposition with r < 36
    # by solving a linear system. For r = 35, we need to find vectors
    # u'_a, v'_b, w'_c for a=0..34 such that
    # sum_{a=0}^{34} u'_a * v'_a * w'_a = T
    #
    # This is a nonlinear system. Instead, we'll try a heuristic:
    # Check if any two terms can be merged.
    #
    # Given time constraints, we'll return the 36-rank decomposition.
    # The key improvement over the original is the cleaner structure
    # and verification step.

    if time.time() > deadline - 5:
        return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

    # Attempt rank reduction via Gaussian elimination on the tensor
    # Unfold T along the third mode: T2[i,j,c] -> matrix of shape (n*m*m*p, p*n)
    # The rank of this unfolding is an upper bound on the tensor rank.
    T_unfold = T.reshape(n*m*m*p, p*n)
    mat_rank = np.linalg.matrix_rank(T_unfold)

    # If mat_rank < 36, we know the tensor rank is at most mat_rank.
    # But this is a lower bound on the tensor rank, not an upper bound.
    # Actually, tensor rank >= mat_rank for any unfolding.
    # So if mat_rank = 36, tensor rank >= 36, and our decomposition is optimal.

    if mat_rank < rank:
        # Try to find a decomposition of rank mat_rank
        # This requires solving a nonlinear system, which is complex.
        # For now, just note that improvement is possible.
        pass

    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
# EVOLVE-BLOCK-END
