"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    dim_u = n * m   # 6
    dim_v = m * p   # 30
    dim_w = p * n   # 20

    # Build the true tensor
    T_true = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        if U_mat.shape[1] == 0:
            return np.allclose(T_true, 0, atol=tol)
        T_rec = np.einsum('ic,jc,kc->ijk', U_mat, V_mat, W_mat)
        return np.allclose(T_rec, T_true, atol=tol)

    # Build the rank-55 Strassen-based decomposition
    rank = 55
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))

    col = 0
    # Part A: j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1

    # Part B: j=1,2 Strassen on 2x2 blocks
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1; V[v_22, col] += 1
        W[w_00, col] += 1; W[w_11, col] += 1
        col += 1
        # M2 = (a21 + a22)b11
        U[u_21, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_10, col] += 1; W[w_11, col] += -1
        col += 1
        # M3 = a11(b12 - b22)
        U[u_11, col] += 1
        V[v_12, col] += 1; V[v_22, col] += -1
        W[w_01, col] += 1; W[w_11, col] += 1
        col += 1
        # M4 = a22(b21 - b11)
        U[u_22, col] += 1
        V[v_21, col] += 1; V[v_11, col] += -1
        W[w_00, col] += 1; W[w_10, col] += 1
        col += 1
        # M5 = (a11 + a12)b22
        U[u_11, col] += 1; U[u_12, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += -1; W[w_01, col] += 1
        col += 1
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21, col] += 1; U[u_11, col] += -1
        V[v_11, col] += 1; V[v_12, col] += 1
        W[w_11, col] += 1
        col += 1
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12, col] += 1; U[u_22, col] += -1
        V[v_21, col] += 1; V[v_22, col] += 1
        W[w_00, col] += 1
        col += 1

    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank

    # Vectorized peeling: check if column c is in the span of others
    def peel_round(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        if r < 2:
            return U_mat, V_mat, W_mat, False

        # Build the "other" tensor matrix: for each c, the rank-1 tensor
        # A: (dim_u*dim_v*dim_w, r) where A[:,c] = vec(U[:,c] outer V[:,c] outer W[:,c])
        # But that's huge. Instead use the structure:
        # T = sum_c U[:,c] V[:,c]^T W[:,c]^T (3-way)
        # Flatten: for each c, the vector is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # We can build this efficiently.

        # Actually, let's use a different approach:
        # For peeling, we check if column c can be expressed as linear combo of others.
        # T = sum_c u_c ⊗ v_c ⊗ w_c
        # Column c is redundant if u_c ⊗ v_c ⊗ w_c ∈ span{u_d ⊗ v_d ⊗ w_d : d ≠ c}

        # Efficient check: build matrix M where M[:,d] = vec(u_d ⊗ v_d ⊗ w_d)
        # But dim_u*dim_v*dim_w = 6*30*20 = 3600, and r up to 55
        # So M is 3600 x 55, which is manageable.

        # Build M = [vec(u_0⊗v_0⊗w_0), ..., vec(u_{r-1}⊗v_{r-1}⊗w_{r-1})]
        # vec(u⊗v⊗w) can be built as: for each (i,j,k), entry = u[i]*v[j]*w[k]
        # M[i*dim_v*dim_w + j*dim_w + k, c] = U[i,c]*V[j,c]*W[k,c]

        M = np.einsum('ic,jc,kc->ijkc', U_mat, V_mat, W_mat)
        M = M.reshape(-1, r)

        for c in range(r):
            if time.time() > deadline - 20:
                break
            others = [i for i in range(r) if i != c]
            M_others = M[:, others]
            b = M[:, c]

            # Check if b is in column space of M_others
            try:
                # Use SVD for numerical stability
                Q, R = np.linalg.qr(M_others)
                # Project b onto col space
                coeffs = Q.T @ b
                proj = Q @ coeffs
                residual = b - proj
                if np.linalg.norm(residual) < 1e-8 * max(1.0, np.linalg.norm(b)):
                    # Column c is redundant
                    current_U = np.delete(U_mat, c, axis=1)
                    current_V = np.delete(V_mat, c, axis=1)
                    current_W = np.delete(W_mat, c, axis=1)
                    return current_U, current_V, current_W, True
            except Exception:
                continue

        return U_mat, V_mat, W_mat, False

    # Iterative peeling
    peeled = True
    while peeled and time.time() < deadline - 30:
        current_U, current_V, current_W, peeled = peel_round(current_U, current_V, current_W)

    if verify(current_U, current_V, current_W):
        if current_rank < rank:
            rank = current_rank
        # Try to go even lower with a targeted search
        # The tensor has a specific structure; try rank r = current_rank - 1, -2, ...
        # Use a random initialization + ALS with vectorized updates

        best_U = current_U.copy()
        best_V = current_V.copy()
        best_W = current_W.copy()
        best_rank = current_rank

        # Try lower ranks with vectorized ALS
        for target_rank in range(best_rank - 1, max(best_rank - 10, 40), -1):
            if time.time() > deadline - 25:
                break

            # Initialize: random or from current
            np.random.seed(42)
            U_init = np.random.randn(dim_u, target_rank) * 0.1
            V_init = np.random.randn(dim_v, target_rank) * 0.1
            W_init = np.random.randn(dim_w, target_rank) * 0.1

            # Also try initializing from the current decomposition's first target_rank columns
            U_init2 = current_U[:, :target_rank].copy()
            V_init2 = current_V[:, :target_rank].copy()
            W_init2 = current_W[:, :target_rank].copy()

            for U_i, V_i, W_i in [(U_init, V_init, W_init), (U_init2, V_init2, W_init2)]:
                if time.time() > deadline - 20:
                    break

                # Vectorized ALS:
                # Update U: for each i, solve U[i,:] @ (V⊗W) = T[i,:,:]
                # (V⊗W) has shape (target_rank, dim_v*dim_w)
                # VW[c, j*dim_w+k] = V[j,c]*W[k,c]
                # Actually: the "matrix" for U[i,:] is:
                # M_U[c, (j,k)] = V[j,c] * W[k,c]  shape (target_rank, dim_v*dim_w)
                # Then U[i,:] @ M_U = T_true[i,:,:].reshape(-1)
                # So: M_U.T @ U[i,:] = T_true[i,:,:].reshape(-1)

                for iteration in range(30):
                    if time.time() > deadline - 20:
                        break

                    # Update U
                    VW = np.einsum('jc,kc->cjk', V_i, W_i).reshape(target_rank, -1)
                    for i in range(dim_u):
                        t_i = T_true[i, :, :].reshape(-1)
                        try:
                            U_i[i, :], _, _, _ = np.linalg.lstsq(VW.T, t_i, rcond=None)
                        except Exception:
                            pass

                    # Update V
                    UW = np.einsum('ic,kc->cik', U_i, W_i).reshape(target_rank, -1)
                    for j in range(dim_v):
                        t_j = T_true[:, j, :].reshape(-1)
                        try:
                            V_i[j, :], _, _, _ = np.linalg.lstsq(UW.T, t_j, rcond=None)
                        except Exception:
                            pass

                    # Update W
                    UV = np.einsum('ic,jc->cij', U_i, V_i).reshape(target_rank, -1)
                    for k in range(dim_w):
                        t_k = T_true[:, :, k].reshape(-1)
                        try:
                            W_i[k, :], _, _, _ = np.linalg.lstsq(UV.T, t_k, rcond=None)
                        except Exception:
                            pass

                    # Check convergence
                    T_rec = np.einsum('ic,jc,kc->ijk', U_i, V_i, W_i)
                    residual = np.linalg.norm(T_rec - T_true) / max(1e-12, np.linalg.norm(T_true))
                    if residual < 1e-9:
                        break

                # Final check
                if time.time() > deadline - 15:
                    break
                T_rec = np.einsum('ic,jc,kc->ijk', U_i, V_i, W_i)
                if np.allclose(T_rec, T_true, atol=1e-8):
                    best_U = U_i
                    best_V = V_i
                    best_W = W_i
                    best_rank = target_rank
                    break

            if best_rank == target_rank:
                break

        if best_rank < current_rank:
            current_U = best_U
            current_V = best_V
            current_W = best_W
            current_rank = best_rank

        if verify(current_U, current_V, current_W):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}

    # Final verification of the Strassen decomposition
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col_s = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col_s] = 1
                V_school[j * p + k, col_s] = 1
                W_school[k * n + i, col_s] = 1
                col_s += 1
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
