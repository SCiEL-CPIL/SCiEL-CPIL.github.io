import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Block size 3
    b = 3
    # Number of blocks per side
    nb = n // b
    
    # Total rank
    rank = 7 * b * b * b # 7 * 27 = 189
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to get global index for A[i, j]
    def idx_A(i, j):
        return i * m + j
        
    # Helper to get global index for B[i, j]
    def idx_B(i, j):
        return i * p + j
        
    # Helper to get global index for C[i, j]
    # Note: W is indexed [k*n + i, r] in the prompt?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually W maps to C[i, k]? Or C[k, i]?
    # Standard: C = A @ B. C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor T[i, j, k] = A[i, j] B[j, k].
    # Decomposition: C[i, k] = sum_r U[i, j?] ...
    # The prompt says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # This suggests:
    # A is n x m. U is n*m x rank. So U row index is i*m + j.
    # B is m x p. V is m*p x rank. So V row index is j*p + k.
    # C is n x p. W is p*n x rank.
    # If W row index is k*n + i, then W[k, i] corresponds to C[i, k]?
    # Let's check the schoolbook seed.
    # Seed: W[k * n + i][column] = 1.
    # In schoolbook: U[i*m+j], V[j*p+k], W[k*n+i].
    # Product: sum_j U[i, j] V[j, k] -> contributes to W[k, i].
    # This implies C[i, k] = sum_r U[i, j_r] V[j_r, k_r] W[k_r, i_r]?
    # No, the tensor decomposition is:
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Here indices are flattened.
    # U row: i*m + j.
    # V row: j*p + k.
    # W row: k*n + i.
    # So yes, W index is k*n + i.
    
    def idx_W(k, i):
        return k * n + i

    # Strassen linear forms for X and Y
    # X and Y are 3x3 matrices.
    # We define them as lists of (coeff, block_row, block_col, local_row, local_col)
    # Or simpler: just build the 36-dim vector for each X_pr and Y_rq.
    
    def get_A_block_vec(br, bc, lr, lc):
        """Returns 36-dim vector for A[br, bc][lr, lc]"""
        v = np.zeros(n * m)
        # Global i = br * b + lr, j = bc * b + lc
        i = br * b + lr
        j = bc * b + lc
        v[idx_A(i, j)] = 1.0
        return v

    def get_B_block_vec(br, bc, lr, lc):
        """Returns 36-dim vector for B[br, bc][lr, lc]"""
        v = np.zeros(m * p)
        i = br * b + lr
        j = bc * b + lc
        v[idx_B(i, j)] = 1.0
        return v

    # Define the 7 Strassen steps
    # Each step is a list of (coeff_A, br_A, bc_A, lr_A, lc_A) for X
    # and (coeff_B, br_B, bc_B, lr_B, lc_B) for Y
    # But X and Y are full 3x3 matrices.
    # We iterate p, q, r.
    # For a given p, r, X_pr is a linear combination.
    
    # We will precompute the linear forms for each X_pr and Y_rq for each of the 7 steps.
    
    # Step 1: X = A00 + A11, Y = B00 + B11
    # Step 2: X = A10 + A11, Y = B00
    # Step 3: X = A00, Y = B01 - B11
    # Step 4: X = A11, Y = B10 - B00
    # Step 5: X = A00 + A01, Y = B11
    # Step 6: X = A10 - A00, Y = B00 + B01
    # Step 7: X = A01 - A11, Y = B10 + B11
    
    # We need to map local indices (lr, lc) to the 3x3 block structure.
    # Block (br, bc) contains elements with local indices (lr, lc).
    # Global A index: (br*b + lr, bc*b + lc).
    
    def make_linear_form_A(terms):
        """
        terms: list of (coeff, br, bc)
        Returns a function that takes (lr, lc) and returns the 36-dim vector.
        Actually, X_pr depends on lr, lc.
        The terms in X are sums of blocks.
        X = c1 * A_b1 + c2 * A_b2 ...
        X[lr, lc] = c1 * A_b1[lr, lc] + c2 * A_b2[lr, lc] ...
        """
        def get_vec(lr, lc):
            v = np.zeros(n * m)
            for coeff, br, bc in terms:
                i = br * b + lr
                j = bc * b + lc
                v[idx_A(i, j)] += coeff
            return v
        return get_vec

    def make_linear_form_B(terms):
        def get_vec(lr, lc):
            v = np.zeros(m * p)
            for coeff, br, bc in terms:
                i = br * b + lr
                j = bc * b + lc
                v[idx_B(i, j)] += coeff
            return v
        return get_vec

    # Define the 7 steps' X and Y linear forms
    # X terms: (coeff, br, bc)
    # Y terms: (coeff, br, bc)
    
    steps = [
        # 1
        ([(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (1, 1, 1)]),
        # 2
        ([(1, 1, 0), (1, 1, 1)], [(1, 0, 0)]),
        # 3
        ([(1, 0, 0)], [(1, 0, 1), (-1, 1, 1)]),
        # 4
        ([(1, 1, 1)], [(1, 1, 0), (-1, 0, 0)]),
        # 5
        ([(1, 0, 0), (1, 0, 1)], [(1, 1, 1)]),
        # 6
        ([(1, 1, 0), (-1, 0, 0)], [(1, 0, 0), (1, 0, 1)]),
        # 7
        ([(1, 0, 1), (-1, 1, 1)], [(1, 1, 0), (1, 1, 1)])
    ]

    # Coefficients for C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_coeffs = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: 1}, # M1, M4, -M5, M7 (indices 0,3,4,6)
        (0, 1): {2: 1, 4: 1},               # M3, M5
        (1, 0): {1: 1, 3: 1},               # M2, M4
        (1, 1): {0: 1, 1: -1, 2: 1, 5: 1}   # M1, -M2, M3, M6
    }

    for step_idx in range(7):
        x_terms, y_terms = steps[step_idx]
        get_X = make_linear_form_A(x_terms)
        get_Y = make_linear_form_B(y_terms)
        
        for p in range(b):
            for q in range(b):
                for r in range(b):
                    # Get vectors for X[p, r] and Y[r, q]
                    u_vec = get_X(p, r)
                    v_vec = get_Y(r, q)
                    
                    # Fill U and V
                    U[:, col] = u_vec
                    V[:, col] = v_vec
                    
                    # Fill W
                    # This term contributes to M_step_idx[p, q]
                    # Which contributes to C blocks
                    w_vec = np.zeros(p * n)
                    for (cb_r, cb_c), coeffs in c_coeffs.items():
                        if step_idx in coeffs:
                            coeff = coeffs[step_idx]
                            # M_step_idx[p, q] is at block (cb_r, cb_c) local (p, q)
                            i = cb_r * b + p
                            k = cb_c * b + q
                            w_vec[idx_W(k, i)] += coeff
                            
                    W[:, col] = w_vec
                    col += 1

    return {"U": U, "V": V, "W": W}
