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

    # Strassen 2x2*2x2: 7 terms
    # For blocks: A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a10)*b11
    # m6 = (a01+a11)*(b00-b10)
    # m7 = (a00+a01)*(b11-b01)
    # C00 = m1 - m4 + m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6

    # For 4x4 * 4x4 using Strassen on 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    #
    # Strassen computes:
    # M1 = (A00+A11)*(B00+B11)  -> 2x2*2x2, rank 4
    # M2 = (A10+A11)*B00        -> 2x2*2x2, rank 4
    # M3 = A00*(B01-B11)        -> 2x2*2x2, rank 4
    # M4 = A11*(B10-B00)        -> 2x2*2x2, rank 4
    # M5 = (A00+A10)*B11        -> 2x2*2x2, rank 4
    # M6 = (A01+A11)*(B00-B10)  -> 2x2*2x2, rank 4
    # M7 = (A00+A01)*(B11-B01)  -> 2x2*2x2, rank 4
    #
    # Each 2x2*2x2 product has rank 4 (schoolbook).
    # Total: 7*4 = 28 terms for 4x4*4x4.

    # For 4x4 * 4x8:
    # B = [B_left, B_right] where B_left, B_right are 4x4
    # C = A * B = [A*B_left, A*B_right]
    # Each A*B_* is a 4x4*4x4 product with rank 28
    # Total: 2*28 = 56 terms

    # Helper: add Strassen terms for a 2x2*2x2 product
    # A_block: dict mapping (local_i, local_j) -> value, where local_i,local_j in {0,1}
    # B_block: dict mapping (local_j, local_k) -> value
    # C_block: dict mapping (local_i, local_k) -> value
    # offset_i, offset_j, offset_k: global offsets
    def add_strassen_2x2(A_off, B_off, C_off, A_local, B_local):
        """Add 4 terms for a 2x2*2x2 schoolbook product.
        A_local: dict {(li, lj): val} for 2x2 A block
        B_local: dict {(lj, lk): val} for 2x2 B block
        A_off: (row_offset, col_offset) in global A indices
        B_off: (row_offset, col_offset) in global B indices
        C_off: (row_offset, col_offset) in global C indices
        """
        for (li, lj), au in A_local.items():
            for (lj2, lk), bv in B_local.items():
                if lj == lj2:
                    u_dict = {(A_off[0] + li, A_off[1] + lj): au}
                    v_dict = {(B_off[0] + lj, B_off[1] + lk): bv}
                    w_dict = {(C_off[1] + lk, C_off[0] + li): 1}
                    add_term(u_dict, v_dict, w_dict)

    # Better approach: directly build the 28-term decomposition for 4x4*4x4
    # Then apply it twice for the two halves of B.

    # For a 4x4*4x4 product with:
    # A indices: i in {0,1,2,3}, j in {0,1,2,3}
    # B indices: j in {0,1,2,3}, k in {0,1,2,3}
    # C indices: i in {0,1,2,3}, k in {0,1,2,3}
    #
    # Block structure:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    #
    # Strassen M1 = (A00+A11)*(B00+B11)
    # This is a 2x2*2x2 product where:
    #   A_part[i,j] = A[i,j] + A[i+2,j+2] for i,j in {0,1}
    #   B_part[j,k] = B[j,k] + B[j+2,k+2] for j,k in {0,1}
    #   C_part[i,k] contributes to C[i,k] and C[i,k+2] and C[i+2,k] and C[i+2,k+2]
    #
    # Actually, M1 = (A00+A11)*(B00+B11) gives a 2x2 result.
    # This 2x2 result contributes to C00, C01, C10, C11 via:
    #   C00 += M1
    #   C01 -= M1 (from C01 = M3 + M5, no wait)
    #
    # Let me re-derive. Strassen:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mx is a 2x2 matrix. So C00 is a 2x2 block, etc.
    #
    # For the tensor decomposition, each Mx = Px * Qx where Px is 2x2 and Qx is 2x2.
    # Px*Qx has rank 4 (schoolbook): sum_{a,b=0}^{1} Px[a,b] * Qx[b,c] * delta
    #
    # So M1 = (A00+A11)*(B00+B11):
    #   For each (a,b) in {0,1}^2:
    #     U contribution: A[a,b] + A[a+2,b+2] at position (a,b) and (a+2,b+2)
    #     Wait, no. U[i*m+j, r] corresponds to A[i,j].
    #     The factor (A00+A11) means U[i*m+j, r] = 1 if (i,j) in A00 or A11, with appropriate signs.
    #     Specifically: U[0*m+0, r] += 1, U[0*m+1, r] += 1, U[1*m+0, r] += 1, U[1*m+1, r] += 1
    #                 U[2*m+2, r] += 1, U[2*m+3, r] += 1, U[3*m+2, r] += 1, U[3*m+3, r] += 1
    #     Wait, A00 is A[0:2, 0:2] so indices (0,0),(0,1),(1,0),(1,1)
    #     A11 is A[2:4, 2:4] so indices (2,2),(2,3),(3,2),(3,3)
    #
    #     The factor (B00+B11) means V[j*p+k, r] = 1 if (j,k) in B00 or B11.
    #     B00 is B[0:2, 0:2] so (0,0),(0,1),(1,0),(1,1)
    #     B11 is B[2:4, 2:4] so (2,2),(2,3),(3,2),(3,3)
    #
    #     The result M1 is a 2x2 matrix. It contributes to C blocks:
    #     C00 += M1, C11 += M1 (from C11 = M1 - M2 + M3 + M6)
    #     So W[k*n+i, r] = 1 for C00 positions and C11 positions.
    #     C00 is C[0:2, 0:2], so (i,k) in {(0,0),(0,1),(1,0),(1,1)}
    #     C11 is C[2:4, 2:4], so (i,k) in {(2,2),(2,3),(3,2),(3,3)}
    #
    #     But M1 is itself a 2x2 product, so we need to decompose it further.
    #     M1[a,c] = sum_b (A00+A11)[a,b] * (B00+B11)[b,c]
    #     So for each b in {0,1}:
    #       Term: U has (A00+A11)[:,b], V has (B00+B11)[b,:], W has the C positions
    #
    # This is getting complex. Let me just enumerate all 28 terms explicitly.

    # I'll build the decomposition by explicitly listing each of the 28 terms.
    # For 4x4*4x4, Strassen gives 7 block-mults, each 2x2*2x2 = 4 scalar terms, total 28.
    # For 4x4*4x8, we do this twice (for B_left and B_right), total 56 terms.

    # Let me define the Strassen structure for 4x4*4x4:
    # A blocks: A00=(0,0),(0,1),(1,0),(1,1); A01=(0,2),(0,3),(1,2),(1,3)
    #           A10=(2,0),(2,1),(3,0),(3,1); A11=(2,2),(2,3),(3,2),(3,3)
    # B blocks: B00=(0,0),(0,1),(1,0),(1,1); B01=(0,2),(0,3),(1,2),(1,3)
    #           B10=(2,0),(2,1),(3,0),(3,1); B11=(2,2),(2,3),(3,2),(3,3)
    # C blocks: C00=(0,0),(0,1),(1,0),(1,1); C01=(0,2),(0,3),(1,2),(1,3)
    #           C10=(2,0),(2,1),(3,0),(3,1); C11=(2,2),(2,3),(3,2),(3,3)

    # Strassen M1 = (A00+A11)*(B00+B11)
    #   U: A00+A11 means U[i*m+j,r]=1 for (i,j) in A00 ∪ A11
    #   V: B00+B11 means V[j*p+k,r]=1 for (j,k) in B00 ∪ B11
    #   But this is a 2x2*2x2 product, so we need 4 sub-terms (one for each b in {0,1})
    #   For b=0: U has col 0 of (A00+A11), V has row 0 of (B00+B11)
    #   For b=1: U has col 1 of (A00+A11), V has row 1 of (B00+B11)
    #   W: C00 and C11 (both get +M1)

    # This is very tedious to do by hand. Let me write a helper.

    def strassen_4x4(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        """Add 28 terms for a 4x4*4x4 Strassen decomposition.
        A is at rows A_row_off:A_row_off+4, cols A_col_off:A_col_off+4 in global A.
        B is at rows B_row_off:B_row_off+4, cols B_col_off:B_col_off+4 in global B.
        C is at rows C_row_off:C_row_off+4, cols C_col_off:C_col_off+4 in global C.
        """
        # Local indices: A[i,j] for i,j in 0..3, B[j,k] for j,k in 0..3, C[i,k] for i,k in 0..3
        # Block offsets within local:
        # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
        # B00: (0,0), B01: (0,2), B10: (2,0), B11: (2,2)
        # C00: (0,0), C01: (0,2), C10: (2,0), C11: (2,2)

        # For each Strassen term Mx, we have a 2x2*2x2 product.
        # The 2x2 A-factor is a linear combination of A-blocks.
        # The 2x2 B-factor is a linear combination of B-blocks.
        # The result contributes to C-blocks with certain signs.

        # Define the 7 Strassen terms:
        # Each term: (A_factor, B_factor, C_contributions)
        # A_factor: dict from local (i,j) in 0..1 x 0..1 to coefficient
        #           These correspond to the 2x2 "super" elements
        # B_factor: dict from local (j,k) in 0..1 x 0..1 to coefficient
        # C_contributions: dict from C-block to coefficient, where C-block is in {00,01,10,11}

        # M1 = (A00+A11)*(B00+B11)
        # A_super[i,j] = A[i,j] + A[i+2,j+2] for i,j in 0,1
        # B_super[j,k] = B[j,k] + B[j+2,k+2] for j,k in 0,1
        # C00 += M1, C11 += M1

        # M2 = (A10+A11)*B00
        # A_super[i,j] = A[i+2,j] + A[i+2,j+2] for i,j in 0,1  (i.e., A10+A11)
        # B_super[j,k] = B[j,k] for j,k in 0,1  (i.e., B00)
        # C10 += M2, C11 -= M2

        # M3 = A00*(B01-B11)
        # A_super[i,j] = A[i,j] for i,j in 0,1  (A00)
        # B_super[j,k] = B[j,j+2] - B[j+2,j+2] ... wait
        # B01 is B[0:2, 2:4], B11 is B[2:4, 2:4]
        # B01-B11: for j in 0,1 and k in 0,1: B[j, 2+k] - B[j+2, 2+k]
        # C01 += M3, C11 += M3

        # M4 = A11*(B10-B00)
        # A_super[i,j] = A[i+2, j+2] for i,j in 0,1  (A11)
        # B_super[j,k] = B[j+2, k] - B[j, k] for j,k in 0,1  (B10-B00)
        # C00 -= M4, C10 += M4

        # M5 = (A00+A10)*B11
        # A_super[i,j] = A[i,j] + A[i+2,j] for i,j in 0,1  (A00+A10)
        # B_super[j,k] = B[j+2, k+2] for j,k in 0,1  (B11)
        # C00 += M5, C01 += M5

        # M6 = (A01+A11)*(B00-B10)
        # A_super[i,j] = A[i,j+2] + A[i+2,j+2] for i,j in 0,1  (A01+A11)
        # B_super[j,k] = B[j,k] - B[j+2,k] for j,k in 0,1  (B00-B10)
        # C11 += M6

        # M7 = (A00+A01)*(B11-B01)
        # A_super[i,j] = A[i,j] + A[i,j+2] for i,j in 0,1  (A00+A01)
        # B_super[j,k] = B[j+2,k+2] - B[j,2+k] for j,k in 0,1  (B11-B01)
        # C00 -= M7

        # Now, each Mx is a 2x2*2x2 product with rank 4.
        # For Mx = Px * Qx where Px is 2x2 and Qx is 2x2:
        #   sum_{b=0}^{1} Px[:,b] * Qx[b,:]
        #   For each b, we have a rank-1 term.

        # Let me define each term explicitly.

        def add_block_product(A_coeffs, B_coeffs, C_block_coeffs):
            """Add 4 terms for a 2x2*2x2 product.
            A_coeffs: dict from (si, sj) in {0,1}^2 to list of (global_i, global_j, coeff)
            B_coeffs: dict from (sj, sk) in {0,1}^2 to list of (global_j, global_k, coeff)
            C_block_coeffs: dict from C-block (si, sk) in {0,1}^2 to coeff
            """
            for b in range(2):  # sum over b
                u_dict = {}
                v_dict = {}
                w_dict = {}
                for (si, sj), entries in A_coeffs.items():
                    if sj == b:
                        for (gi, gj, c) in entries:
                            u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + c
                for (sj, sk), entries in B_coeffs.items():
                    if sj == b:
                        for (gj, gk, c) in entries:
                            v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + c
                for (si, sk), c in C_block_coeffs.items():
                    if c != 0:
                        # C-block (si, sk) corresponds to 2x2 block at offset (si*2, sk*2)
                        for di in range(2):
                            for dk in range(2):
                                gi = C_row_off + si * 2 + di
                                gk = C_col_off + sk * 2 + dk
                                w_dict[(gk, gi)] = w_dict.get((gk, gi), 0) + c
                if u_dict and v_dict and w_dict:
                    add_term(u_dict, v_dict, w_dict)

        # Define A-block coefficient helpers
        def A_block(block, sign=1):
            """Return dict from super-index (si,sj) to list of (global_i, global_j, coeff) for A-block."""
            # block: 0=A00, 1=A01, 2=A10, 3=A11
            bi, bj = divmod(block, 2)  # bi=block row, bj=block col
            result = {}
            for si in range(2):
                for sj in range(2):
                    gi = A_row_off + bi * 2 + si
                    gj = A_col_off + bj * 2 + sj
                    result[(si, sj)] = [(gi, gj, sign)]
            return result

        def B_block(block, sign=1):
            """Return dict from super-index (sj,sk) to list of (global_j, global_k, coeff) for B-block."""
            bi, bj = divmod(block, 2)
            result = {}
            for sj in range(2):
                for sk in range(2):
                    gj = B_row_off + bi * 2 + sj
                    gk = B_col_off + bj * 2 + sk
                    result[(sj, sk)] = [(gj, gk, sign)]
            return result

        # M1 = (A00+A11)*(B00+B11), C00+=M1, C11+=M1
        A1 = {**A_block(0, 1), **A_block(3, 1)}  # A00+A11
        B1 = {**B_block(0, 1), **B_block(3, 1)}  # B00+B11
        C1 = {(0, 0): 1, (1, 1): 1}  # C00, C11
        add_block_product(A1, B1, C1)

        # M2 = (A10+A11)*B00, C10+=M2, C11-=M2
        A2 = {**A_block(2, 1), **A_block(3, 1)}  # A10+A11
        B2 = B_block(0, 1)  # B00
        C2 = {(0, 1): 1, (1, 1): -1}  # C10, C11
        add_block_product(A2, B2, C2)

        # M3 = A00*(B01-B11), C01+=M3, C11+=M3
        A3 = A_block(0, 1)  # A00
        B3 = {**B_block(1, 1), **B_block(3, -1)}  # B01-B11
        C3 = {(0, 1): 1, (1, 1): 1}  # C01, C11
        add_block_product(A3, B3, C3)

        # M4 = A11*(B10-B00), C00-=M4, C10+=M4
        A4 = A_block(3, 1)  # A11
        B4 = {**B_block(2, 1), **B_block(0, -1)}  # B10-B00
        C4 = {(0, 0): -1, (0, 1): 1}  # C00, C10
        add_block_product(A4, B4, C4)

        # M5 = (A00+A10)*B11, C00+=M5, C01+=M5
        A5 = {**A_block(0, 1), **A_block(2, 1)}  # A00+A10
        B5 = B_block(3, 1)  # B11
        C5 = {(0, 0): 1, (0, 1): 1}  # C00, C01
        add_block_product(A5, B5, C5)

        # M6 = (A01+A11)*(B00-B10), C11+=M6
        A6 = {**A_block(1, 1), **A_block(3, 1)}  # A01+A11
        B6 = {**B_block(0, 1), **B_block(2, -1)}  # B00-B10
        C6 = {(1, 1): 1}  # C11
        add_block_product(A6, B6, C6)

        # M7 = (A00+A01)*(B11-B01), C00-=M7
        A7 = {**A_block(0, 1), **A_block(1, 1)}  # A00+A01
        B7 = {**B_block(3, 1), **B_block(1, -1)}  # B11-B01
        C7 = {(0, 0): -1}  # C00
        add_block_product(A7, B7, C7)

    # For 4x4 * 4x8:
    # A is at rows 0:4, cols 0:4 in global A (n=4, m=4)
    # B is at rows 0:4, cols 0:8 in global B (m=4, p=8)
    # C is at rows 0:4, cols 0:8 in global C (p=8, n=4) -> C[k*n+i] where k in 0..7, i in 0..3
    #
    # Split B into B_left (cols 0:4) and B_right (cols 4:8)
    # C_left = A * B_left (4x4), C_right = A * B_right (4x4)
    #
    # For C_left: C indices (i,k) where i in 0..3, k in 0..3
    #   W[k*n+i, r] = W[k*4+i, r]
    # For C_right: C indices (i,k) where i in 0..3, k in 4..7
    #   W[k*n+i, r] = W[k*4+i, r]

    # Apply Strassen 4x4*4x4 for B_left (cols 0:4 of B)
    strassen_4x4(A_row_off=0, A_col_off=0, B_row_off=0, B_col_off=0, C_row_off=0, C_col_off=0)
    # Apply Strassen 4x4*4x4 for B_right (cols 4:8 of B)
    strassen_4x4(A_row_off=0, A_col_off=0, B_row_off=0, B_col_off=4, C_row_off=0, C_col_off=4)

    rank = len(U_list)
    U = np.array(U_list).T  # shape (nm, rank)
    V = np.array(V_list).T  # shape (mp, rank)
    W = np.array(W_list).T  # shape (pn, rank)

    # Verify
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1 for all i,j,k
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-10:
                    error += abs(val - 1)
    
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
