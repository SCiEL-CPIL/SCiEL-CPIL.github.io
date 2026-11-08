import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using two-level Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+k, r], V[k*4+j, r], W[k*4+i, r]
    Tensor: T[i*4+k, k*4+j, k*4+i] = sum_r U[i*4+k,r] * V[k*4+j,r] * W[k*4+i,r] = 1
    """
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)

    def u_idx(i, k):
        return i * m + k

    def v_idx(k, j):
        return k * p + j

    def w_idx(k, i):
        return k * n + i

    # Two-level Strassen: 7 top-level terms x 7 sub-terms = 49 rank-1 tensors

    # Level 1: Split 4x4 into 2x2 blocks (A00, A01, A10, A11)
    # Apply Strassen to compute C = A*B where A,B are 2x2 block matrices

    # Strassen 2x2 block-level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each top-level Strassen term, we have a 2x2 matrix product of block matrices.
    # Each block is 2x2, so each block product itself uses Strassen (7 sub-terms).

    # Define the 7 top-level terms with their A and B block coefficients
    # Each term: (a_coeffs, b_coeffs, c_contrib)
    # a_coeffs: {(block_r, block_c): coeff} for A blocks
    # b_coeffs: {(block_r, block_c): coeff} for B blocks
    # c_contrib: {(block_r, block_c): coeff} for C blocks this term contributes to

    top_terms = [
        # Term 0: M1 = (A00+A11)(B00+B11) -> C00 +=, C11 +=
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # Term 1: M2 = (A10+A11)B00 -> C10 +=, C11 -=
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        # Term 2: M3 = A00(B01-B11) -> C01 +=, C11 +=
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        # Term 3: M4 = A11(B10-B00) -> C00 +=, C10 +=
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # Term 4: M5 = (A00+A01)B11 -> C00 -=, C01 +=
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        # Term 5: M6 = (A10-A00)(B00+B01) -> C11 +=
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # Term 6: M7 = (A01-A11)(B10+B11) -> C00 +=
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]

    # For each top-level term, we need to compute the 2x2 block product.
    # The block product L*B where L is a 2x2 matrix of linear forms on A
    # and B is a 2x2 matrix of linear forms on B.
    # We apply Strassen to this 2x2 "matrix of linear forms" product.

    # Strassen sub-term definitions for 2x2 matrix product:
    # Given L = [[L00, L01], [L10, L11]] and B = [[B00, B01], [B10, B11]]
    # sub1 = (L00+L11)(B00+B11) -> P00 +=, P11 +=
    # sub2 = (L10+L11)B00       -> P10 +=, P11 -=
    # sub3 = L00(B01-B11)       -> P01 +=, P11 +=
    # sub4 = L11(B10-B00)       -> P00 +=, P10 +=
    # sub5 = (L00+L01)B11       -> P00 -=, P01 +=
    # sub6 = (L10-L00)(B00+B01) -> P11 +=
    # sub7 = (L01-L11)(B10+B11) -> P00 +=

    # Sub-contribution patterns: which C positions each sub-term contributes to
    sub_contrib_patterns = [
        [(0, 0, 1), (1, 1, 1)],    # sub1
        [(1, 0, 1), (1, 1, -1)],   # sub2
        [(0, 1, 1), (1, 1, 1)],    # sub3
        [(0, 0, 1), (1, 0, 1)],    # sub4
        [(0, 0, -1), (0, 1, 1)],   # sub5
        [(1, 1, 1)],               # sub6
        [(0, 0, 1)],               # sub7
    ]

    # For a given top-level term, extract the 2x2 block linear forms
    # For each block position (br, bc) in {0,1}x{0,1}, the block L[br,bc] is a 2x2 matrix
    # where L[br,bc][r,c] = sum of a_coeffs[br,bc] * A_global[2*br+r, 2*bc+c]
    # Similarly for B blocks.

    # We'll directly construct U, V, W entries for each of the 49 rank-1 tensors.

    for t in range(7):
        a_coeffs, b_coeffs, c_contrib = top_terms[t]

        # Build the 2x2 "matrix of linear forms" L and R
        # L[br][bc] = linear form on A: {(i,k): coeff} where i=2*br+r, k=2*bc+c
        # R[br][bc] = linear form on B: {(k,j): coeff} where k=2*br+r, j=2*bc+c

        L = [[{} for _ in range(2)] for _ in range(2)]
        R = [[{} for _ in range(2)] for _ in range(2)]

        for (br, bc), coeff in a_coeffs.items():
            for r in range(2):
                for c in range(2):
                    i = 2 * br + r
                    k = 2 * bc + c
                    L[br][bc][i, k] = L[br][bc].get((i, k), 0) + coeff

        for (br, bc), coeff in b_coeffs.items():
            for r in range(2):
                for c in range(2):
                    k = 2 * br + r
                    j = 2 * bc + c
                    R[br][bc][k, j] = R[br][bc].get((k, j), 0) + coeff

        # Apply Strassen to compute L * R (2x2 matrix of linear forms)
        # L00, L01, L10, L11 and R00, R01, R10, R11

        L00, L01 = L[0][0], L[0][1]
        L10, L11 = L[1][0], L[1][1]
        R00, R01 = R[0][0], R[0][1]
        R10, R11 = R[1][0], R[1][1]

        # 7 Strassen sub-products
        sub_products = []

        # sub1: (L00+L11)(R00+R11)
        a1 = {k: v for k, v in {**L00, **{kk: vv + L00.get(kk, 0) for kk, vv in L11.items()}}.items() if v != 0}
        b1 = {k: v for k, v in {**R00, **{kk: vv + R00.get(kk, 0) for kk, vv in R11.items()}}.items() if v != 0}
        sub_products.append((a1, b1))

        # sub2: (L10+L11)R00
        a2 = {k: v for k, v in {**L10, **{kk: vv + L10.get(kk, 0) for kk, vv in L11.items()}}.items() if v != 0}
        b2 = dict(R00)
        sub_products.append((a2, b2))

        # sub3: L00(R01-R11)
        a3 = dict(L00)
        b3 = {k: v for k, v in {**R01, **{kk: -vv + R01.get(kk, 0) for kk, vv in R11.items()}}.items() if v != 0}
        sub_products.append((a3, b3))

        # sub4: L11(R10-R00)
        a4 = dict(L11)
        b4 = {k: v for k, v in {**R10, **{kk: -vv + R10.get(kk, 0) for kk, vv in R00.items()}}.items() if v != 0}
        sub_products.append((a4, b4))

        # sub5: (L00+L01)R11
        a5 = {k: v for k, v in {**L00, **{kk: vv + L00.get(kk, 0) for kk, vv in L01.items()}}.items() if v != 0}
        b5 = dict(R11)
        sub_products.append((a5, b5))

        # sub6: (L10-L00)(R00+R01)
        a6 = {k: v for k, v in {**L10, **{kk: -vv + L10.get(kk, 0) for kk, vv in L00.items()}}.items() if v != 0}
        b6 = {k: v for k, v in {**R00, **{kk: vv + R00.get(kk, 0) for kk, vv in R01.items()}}.items() if v != 0}
        sub_products.append((a6, b6))

        # sub7: (L01-L11)(R10+R11)
        a7 = {k: v for k, v in {**L01, **{kk: -vv + L01.get(kk, 0) for kk, vv in L11.items()}}.items() if v != 0}
        b7 = {k: v for k, v in {**R10, **{kk: vv + R10.get(kk, 0) for kk, vv in R11.items()}}.items() if v != 0}
        sub_products.append((a7, b7))

        # Now fill U, V, W for each of the 7 sub-terms
        for s in range(7):
            rank_idx = t * 7 + s
            A_form, B_form = sub_products[s]

            # Fill U
            for (i, k), coeff in A_form.items():
                if coeff != 0:
                    U[u_idx(i, k), rank_idx] = coeff

            # Fill V
            for (k, j), coeff in B_form.items():
                if coeff != 0:
                    V[v_idx(k, j), rank_idx] = coeff

            # Fill W: combine sub-contribution with top-level c_contrib
            for (lr, lc, sub_coeff) in sub_contrib_patterns[s]:
                # (lr, lc) is the local block position in the 2x2 result
                # This corresponds to C block (lr, lc)
                if (lr, lc) in c_contrib:
                    block_coeff = c_contrib[(lr, lc)]
                    total_coeff = sub_coeff * block_coeff
                    # W for block (lr, lc): W[k*4+i] where i = 2*lr+ri, k = 2*lc+ck
                    for ri in range(2):
                        for ck in range(2):
                            i = 2 * lr + ri
                            k = 2 * lc + ck
                            if total_coeff != 0:
                                W[w_idx(k, i), rank_idx] = total_coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
