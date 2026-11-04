import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32

    # Strassen 2x2*2x2 decomposition (rank 7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0: M1 = (a00+a11)(b00+b11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    # c=1: M2 = (a10+a11)*b00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    # c=2: M3 = a00*(b01-b11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    # c=3: M4 = a11*(b10-b00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    # c=4: M5 = (a00+a01)*b11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    # c=5: M6 = (a10-a00)*(b00+b01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    # c=6: M7 = (a01-a11)*(b10+b11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block Strassen for 4x4*4x4 (rank 49)
    A_specs = [
        [(1, 0, 0), (1, 1, 1)],   # P0: A00 + A11
        [(1, 1, 0), (1, 1, 1)],   # P1: A10 + A11
        [(1, 0, 0)],              # P2: A00
        [(1, 1, 1)],              # P3: A11
        [(1, 0, 0), (1, 0, 1)],   # P4: A00 + A01
        [(1, 1, 0), (-1, 0, 0)],  # P5: A10 - A00
        [(1, 0, 1), (-1, 1, 1)],  # P6: A01 - A11
    ]

    B_specs = [
        [(1, 0, 0), (1, 1, 1)],   # P0: B00 + B11
        [(1, 0, 0)],              # P1: B00
        [(1, 0, 1), (-1, 1, 1)],  # P2: B01 - B11
        [(1, 1, 0), (-1, 0, 0)],  # P3: B10 - B00
        [(1, 1, 1)],              # P4: B11
        [(1, 0, 0), (1, 0, 1)],   # P5: B00 + B01
        [(1, 1, 0), (1, 1, 1)],   # P6: B10 + B11
    ]

    C_specs = [
        {(0, 0): 1, (1, 1): 1},        # P0: C00 + C11
        {(1, 0): 1, (1, 1): -1},       # P1: C10 - C11
        {(0, 1): 1, (1, 1): 1},        # P2: C01 + C11
        {(0, 0): 1, (1, 0): 1},        # P3: C00 + C10
        {(0, 0): -1, (0, 1): 1},       # P4: -C00 + C01
        {(1, 1): 1},                   # P5: C11
        {(0, 0): 1},                   # P6: C00
    ]

    def build_strassen_4x4x4(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        """Build U, V, W lists for a 4x4*4x4 block Strassen decomposition."""
        U_list = []
        V_list = []
        W_list = []

        def add_term(u_dict, v_dict, w_dict):
            U_row = np.zeros(nm)
            V_row = np.zeros(mp)
            W_row = np.zeros(pn)
            for (i, j), val in u_dict.items():
                U_row[i * m + j] += val
            for (j, k), val in v_dict.items():
                V_row[j * p + k] += val
            for (k, i), val in w_dict.items():
                W_row[k * n + i] += val
            U_list.append(U_row)
            V_list.append(V_row)
            W_list.append(W_row)

        for k_p in range(7):
            for m_s in range(7):
                u_dict = {}
                v_dict = {}
                w_dict = {}

                for (coeff, br, bc) in A_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            u_val = U_2[r * 2 + c, m_s]
                            if u_val != 0:
                                gi = A_row_off + br * 2 + r
                                gj = A_col_off + bc * 2 + c
                                u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + coeff * u_val

                for (coeff, br, bc) in B_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            v_val = V_2[r * 2 + c, m_s]
                            if v_val != 0:
                                gj = B_row_off + br * 2 + r
                                gk = B_col_off + bc * 2 + c
                                v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + coeff * v_val

                for (cbr, cbc), coeff_c in C_specs[k_p].items():
                    for r in range(2):
                        for c in range(2):
                            w_val = W_2[r * 2 + c, m_s]
                            if w_val != 0:
                                gi = C_row_off + cbr * 2 + r
                                gk = C_col_off + cbc * 2 + c
                                w_dict[(gk, gi)] = w_dict.get((gk, gi), 0) + coeff_c * w_val

                if u_dict and v_dict and w_dict:
                    add_term(u_dict, v_dict, w_dict)

        return U_list, V_list, W_list

    # Build left half (B cols 0-3, C cols 0-3)
    U_left, V_left, W_left = build_strassen_4x4x4(
        A_row_off=0, A_col_off=0,
        B_row_off=0, B_col_off=0,
        C_row_off=0, C_col_off=0
    )

    # Build right half (B cols 4-7, C cols 4-7)
    U_right, V_right, W_right = build_strassen_4x4x4(
        A_row_off=0, A_col_off=0,
        B_row_off=0, B_col_off=4,
        C_row_off=0, C_col_off=4
    )

    # KEY INSIGHT: U_left and U_right are identical (same A matrix).
    # So we can use a single shared U, and for V/W use left and right values.
    # This reduces rank from 98 to 49.
    
    # Verify U_left == U_right
    u_match = all(np.allclose(U_left[i], U_right[i]) for i in range(len(U_left)))
    
    if u_match and len(U_left) == 49:
        # Use shared U, combined V and W
        U_list = U_left  # shared
        V_list = V_left + V_right
        W_list = W_left + W_right
        
        rank = len(U_list)
        # But wait: V and W have different lengths than U now.
        # We need rank = max(len(U_list), len(V_list), len(W_list))
        # Actually no: each "term" r contributes U[:,r], V[:,r], W[:,r]
        # We need the same number of columns in U, V, W.
        # 
        # The trick: for the left half terms (r=0..48), use U_left[r], V_left[r], W_left[r]
        # For the right half terms (r=0..48), use U_right[r] = U_left[r], V_right[r], W_right[r]
        # But U_left[r] = U_right[r], so we can't just concatenate.
        #
        # Actually, the decomposition is:
        # T = sum_r U[:,r] V[:,r] W[:,r] (outer product)
        # For left: T_left = sum_{r=0}^{48} U_left[:,r] V_left[:,r] W_left[:,r]
        # For right: T_right = sum_{r=0}^{48} U_right[:,r] V_right[:,r] W_right[:,r]
        # Total: T = T_left + T_right
        # = sum_{r=0}^{48} [U_left[:,r] V_left[:,r] W_left[:,r] + U_right[:,r] V_right[:,r] W_right[:,r]]
        # Since U_left[:,r] = U_right[:,r] = U[:,r]:
        # = sum_{r=0}^{48} U[:,r] (V_left[:,r] W_left[:,r] + V_right[:,r] W_right[:,r])
        # But this isn't a valid tensor decomposition because we need separate V and W columns.
        #
        # The correct approach: we still need 98 terms, but U has only 49 unique columns.
        # However, the rank is defined by the number of terms, not unique U columns.
        #
        # Wait, let me reconsider. The tensor rank is the minimum number of rank-1 tensors
        # whose sum equals T. Each rank-1 tensor is U_r ⊗ V_r ⊗ W_r.
        # If U_left[:,r] = U_right[:,r], then the r-th term of left and r-th term of right
        # share the same U factor. But they are still two separate rank-1 tensors.
        #
        # So the rank is still 98, not 49. The sharing of U doesn't reduce the tensor rank.
        #
        # Let me go back to the rank-98 construction but verify it works.
        pass

    # Fallback: use the original rank-98 construction
    U_list = U_left + U_right
    V_list = V_left + V_right
    W_list = W_left + W_right

    rank = len(U_list)
    U = np.array(U_list).T  # shape (nm, rank)
    V = np.array(V_list).T  # shape (mp, rank)
    W = np.array(W_list).T  # shape (pn, rank)

    # Verify
    error = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j, :]
            for k in range(p):
                val = np.dot(u_row, V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    error += abs(val - 1.0)

    if error > 1e-8:
        # Fallback to schoolbook
        U_school = np.zeros((nm, 128))
        V_school = np.zeros((mp, 128))
        W_school = np.zeros((pn, 128))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
