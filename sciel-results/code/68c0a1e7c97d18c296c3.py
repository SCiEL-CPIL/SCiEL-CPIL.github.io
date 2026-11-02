import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = 2, 4, 4

    # Build the rank-28 Strassen decomposition
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1

        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1

        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)

        # M1 = (a00+a11)(b00+b11)
        U[a00_u, offset + 0] += 1
        U[a11_u, offset + 0] += 1
        V[b00_v, offset + 0] += 1
        V[b11_v, offset + 0] += 1
        W[c00_w, offset + 0] += 1
        W[c11_w, offset + 0] += 1

        # M2 = (a10+a11)b00
        U[a10_u, offset + 1] += 1
        U[a11_u, offset + 1] += 1
        V[b00_v, offset + 1] += 1
        W[c10_w, offset + 1] += 1
        W[c11_w, offset + 1] -= 1

        # M3 = a00(b01-b11)
        U[a00_u, offset + 2] += 1
        V[b01_v, offset + 2] += 1
        V[b11_v, offset + 2] -= 1
        W[c01_w, offset + 2] += 1
        W[c11_w, offset + 2] += 1

        # M4 = a11(b10-b00)
        U[a11_u, offset + 3] += 1
        V[b10_v, offset + 3] += 1
        V[b00_v, offset + 3] -= 1
        W[c00_w, offset + 3] += 1
        W[c10_w, offset + 3] += 1

        # M5 = (a00+a01)b11
        U[a00_u, offset + 4] += 1
        U[a01_u, offset + 4] += 1
        V[b11_v, offset + 4] += 1
        W[c00_w, offset + 4] -= 1
        W[c01_w, offset + 4] += 1

        # M6 = (a10-a00)(b00+b01)
        U[a10_u, offset + 5] += 1
        U[a00_u, offset + 5] -= 1
        V[b00_v, offset + 5] += 1
        V[b01_v, offset + 5] += 1
        W[c11_w, offset + 5] += 1

        # M7 = (a01-a11)(b10+b11)
        U[a01_u, offset + 6] += 1
        U[a11_u, offset + 6] -= 1
        V[b10_v, offset + 6] += 1
        V[b11_v, offset + 6] += 1
        W[c00_w, offset + 6] += 1

    # 4 blocks
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(7, 0, 0, 0, 2, 0, 2)
    add_strassen_block(14, 0, 2, 2, 0, 0, 0)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)

    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Fast verification using random matrix products."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()

            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(p):
                    s = 0.0
                    for r in range(U_mat.shape[1]):
                        u_val = np.dot(U_mat[:, r], A_flat)
                        v_val = np.dot(V_mat[:, r], B_flat)
                        s += u_val * v_val * W_mat[j * n + i, r]
                    C_computed[i, j] = s

            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    # Verify the base decomposition
    if not verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by checking for linear dependencies using SVD
    # Build the tensor matrix: each column r contributes U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    vecs = []
    for r in range(rank):
        vec = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r]).flatten()
        vecs.append(vec)
    mat = np.array(vecs).T  # (n*m*p, rank) = (32, 28)

    try:
        s_vals, _, Vt = np.linalg.svd(mat, full_matrices=False)
        rank_mat = np.sum(s_vals > 1e-10)

        if rank_mat < rank:
            # The columns are linearly dependent; try to eliminate one
            null_vec = Vt[rank_mat]  # First null vector
            max_idx = np.argmax(np.abs(null_vec))

            U_sub = np.delete(U, max_idx, axis=1)
            V_sub = np.delete(V, max_idx, axis=1)
            W_sub = np.delete(W, max_idx, axis=1)

            if verify_decomposition(U_sub, V_sub, W_sub, n, m, p):
                return {"U": U_sub.tolist(), "V": V_sub.tolist(), "W": W_sub.tolist()}
    except Exception:
        pass

    # Additional refinement: try to find a rank-27 decomposition via
    # solving the tensor equations with one column removed and adjusting
    # the remaining columns. This is a more sophisticated approach.
    # We'll use a least-squares approach: for each column c0 to remove,
    # try to express its tensor contribution as a combination of others.

    # For a rank-1 tensor t_c = U_c ⊗ V_c ⊗ W_c, we need:
    # t_c0 = sum_{c != c0} alpha_c * t_c
    # This is equivalent to: for all (i,j,k):
    # U_c0[i*m+j]*V_c0[j*p+k]*W_c0[k*n+i] = sum_{c != c0} alpha_c * U_c[i*m+j]*V_c[j*p+k]*W_c[k*n+i]
    #
    # This is a system of 32 equations in (rank-1) unknowns.
    # Let's set up the system:
    # Let T_c be the flattened tensor for column c.
    # We need: T_c0 = sum_{c != c0} alpha_c * T_c
    # This is: [T_0, T_1, ..., T_{c0-1}, T_{c0+1}, ..., T_{27}] * alpha = T_c0
    # where alpha has 27 components.

    # Build the matrix of all tensor contributions
    tensor_mat = np.array(vecs).T  # (32, 28)

    best_rank_found = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    for c0 in range(rank):
        if time.time() > deadline:
            break

        # Remove column c0
        mask = [i for i in range(rank) if i != c0]
        tensor_reduced = tensor_mat[:, mask]  # (32, 27)
        target = tensor_mat[:, c0]

        # Solve least squares: tensor_reduced * alpha = target
        try:
            alpha, residuals, _, _ = np.linalg.lstsq(tensor_reduced, target, rcond=None)
        except Exception:
            continue

        # Check if the solution is good
        if len(residuals) > 0 and residuals[0] < 1e-8:
            # The tensor contribution of column c0 can be exactly represented
            # by a combination of the other columns. But this doesn't directly
            # give us a valid decomposition because the U, V, W matrices need
            # to be adjusted.
            #
            # Actually, if T_c0 = sum alpha_c * T_c, then the full tensor is:
            # sum_c T_c = sum_{c != c0} (1 + alpha_c) * T_c + (1 - 1) * T_c0
            # Wait, that's not right. Let me think again.
            #
            # The full tensor is: sum_{c=0}^{27} T_c
            # If T_c0 = sum_{c != c0} alpha_c * T_c, then:
            # sum_{c=0}^{27} T_c = sum_{c != c0} T_c + T_c0 = sum_{c != c0} (1 + alpha_c) * T_c
            #
            # But (1 + alpha_c) * T_c = (1 + alpha_c) * (U_c ⊗ V_c ⊗ W_c)
            # This is NOT a valid rank-1 tensor in general, because the scaling
            # must be applied to one of the factors.
            #
            # So this approach doesn't directly work. Let's try a different strategy.
            pass

    # Try a more direct approach: for each pair of columns, try to merge them
    # into a single column by finding U', V', W' such that:
    # U_c1 ⊗ V_c1 ⊗ W_c1 + U_c2 ⊗ V_c2 ⊗ W_c2 = U' ⊗ V' ⊗ W'
    # This is generally not possible, but let's check if any pair has a special structure.

    # Alternative: try to find a rank-27 decomposition by solving the full
    # system of tensor equations with 27 unknown columns.
    # This is a nonlinear system, so we'll use a random search approach.

    # For now, return the best result found (rank 28)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
