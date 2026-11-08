import time
import numpy as np

DIMENSIONS = [4, 4, 9]


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition:
    - Split B into B_left (4x4) and B_right (4x5)
    - A * B_left: Strassen 4x4*4x4 (rank 49)
    - A * B_right: schoolbook 4x4*4x5 (rank 80)
    - Total rank: 129
    
    Alternative: Split B into three 4x3 blocks won't work cleanly.
    Better: use 4x4 + 4x4 + 4x1:
    - A * B[0:4]: Strassen 4x4*4x4 (rank 49)
    - A * B[4:8]: Strassen 4x4*4x4 (rank 49)
    - A * B[8:9]: schoolbook 4x4*4x1 (rank 16)
    - Total rank: 114
    """
    start = time.time()
    deadline = start + 100

    n, m, p = 4, 4, 9
    num_A = n * m  # 16
    num_B = m * p  # 36
    num_C = p * n  # 36

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Strassen 2x2*2x2*2x2: 7 terms
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    S2 = [
        # (A_dict, B_dict, C_dict) with coefficients
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),     # s1=(a+d)(e+h) -> c00,c11
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),          # s2=(c+d)e -> c10,c11
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),          # s3=a(f-h) -> c01,c11
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),          # s4=d(g-e) -> c00,c10
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),          # s5=(a+b)h -> c00,c01
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),          # s6=(c-a)(e+f) -> c11
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),          # s7=(b-d)(g+h) -> c00
    ]

    # Block Strassen for 4x4*4x4 (blocks are 2x2): 7 terms
    # Block A: top-left=0, top-right=1, bottom-left=2, bottom-right=3
    # Block B: top-left=0, top-right=1, bottom-left=2, bottom-right=3
    # Block C: top-left=0, top-right=1, bottom-left=2, bottom-right=3
    BS = [
        # (A_blocks, B_blocks, C_blocks) where each is dict {block_idx: coeff}
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),     # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),          # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),          # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),          # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),          # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),          # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),          # M7
    ]

    def strassen_4x4(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """Add 49 terms for 4x4 * 4x4 matrix multiplication."""
        for Ab, Bb, Cb in BS:
            for a_elem, b_elem, c_elem in S2:
                u = [0] * num_A
                v = [0] * num_B
                w = [0] * num_C

                # A: element (li, lj) within block (br, bc)
                # Global A index: (A_i_base + br*2 + li) * m + (A_j_base + bc*2 + lj)
                for a_block_idx, a_coeff in Ab.items():
                    a_br = a_block_idx // 2  # 0 or 1
                    a_bc = a_block_idx % 2   # 0 or 1
                    for a_elem_idx, a_e_coeff in a_elem.items():
                        a_li = a_elem_idx // 2
                        a_lj = a_elem_idx % 2
                        gi = A_i_base + a_br * 2 + a_li
                        gj = A_j_base + a_bc * 2 + a_lj
                        u[gi * m + gj] += a_coeff * a_e_coeff

                # B: element (lj, lk) within block (br, bc)
                # Global B index: (B_j_base + br*2 + lj) * p + (B_k_base + bc*2 + lk)
                for b_block_idx, b_coeff in Bb.items():
                    b_br = b_block_idx // 2
                    b_bc = b_block_idx % 2
                    for b_elem_idx, b_e_coeff in b_elem.items():
                        b_lj = b_elem_idx // 2
                        b_lk = b_elem_idx % 2
                        gj = B_j_base + b_br * 2 + b_lj
                        gk = B_k_base + b_bc * 2 + b_lk
                        v[gj * p + gk] += b_coeff * b_e_coeff

                # C: element (li, lk) within block (br, bc)
                # Global C index: (C_k_base + bc*2 + lk) * n + (C_i_base + br*2 + li)
                for c_block_idx, c_coeff in Cb.items():
                    c_br = c_block_idx // 2
                    c_bc = c_block_idx % 2
                    for c_elem_idx, c_e_coeff in c_elem.items():
                        c_li = c_elem_idx // 2
                        c_lk = c_elem_idx % 2
                        gi = C_i_base + c_br * 2 + c_li
                        gk = C_k_base + c_bc * 2 + c_lk
                        w[gk * n + gi] += c_coeff * c_e_coeff

                add_term(u, v, w)

    # Part 1: A * B[:, 0:4] -> C[0:4, :]
    strassen_4x4(0, 0, 0, 0, 0, 0)

    # Part 2: A * B[:, 4:8] -> C[4:8, :]
    strassen_4x4(0, 0, 0, 4, 0, 4)

    # Part 3: A * B[:, 8:9] -> C[8:9, :], schoolbook
    for i in range(n):
        for j in range(m):
            u = [0] * num_A
            v = [0] * num_B
            w = [0] * num_C
            u[i * m + j] = 1
            v[j * p + 8] = 1
            w[8 * n + i] = 1
            add_term(u, v, w)

    rank = len(U_cols)

    # Build matrices
    U = np.zeros((num_A, rank), dtype=object)
    V = np.zeros((num_B, rank), dtype=object)
    W = np.zeros((num_C, rank), dtype=object)

    for r in range(rank):
        for i in range(num_A):
            U[i, r] = U_cols[r][i]
        for j in range(num_B):
            V[j, r] = V_cols[r][j]
        for k in range(num_C):
            W[k, r] = W_cols[r][k]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
