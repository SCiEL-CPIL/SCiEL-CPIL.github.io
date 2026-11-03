import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Known rank-28 solution via Strassen on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # M1: (x00+x11)(y00+y11)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        # M2: (x10+x11)y00
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        # M3: x00(y01-y11)
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        # M4: x11(y10-y00)
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        # M5: (x00+x01)y11
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        # M6: (x10-x00)(y00+y01)
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        # M7: (x01-x11)(y10+y11)
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: A_L B_TL -> C_LL
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    # Block 2: A_R B_BL -> C_LR
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    # Block 3: A_L B_TR -> C_RL
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    # Block 4: A_R B_BR -> C_RR
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l       # index in U
                b = l * p + k       # index in V
                c = k * n + i       # index in W (trace convention: k*n+i)
                T[a, b, c] = 1.0

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        """Check that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c."""
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Verify rank-28 solution
    max_err = verify_decomp(U_arr, V_arr, W_arr, rank)
    if max_err > 1e-10:
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}

    # Search for rank reduction by checking linear dependencies
    # Build M matrix where M[:, r] is the r-th rank-1 tensor flattened
    M = np.zeros((nm * mp * pn, rank), dtype=float)
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten()

    # Check rank of M
    try:
        U_svd, S_svd, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        num_nonzero = np.sum(S_svd > tol)
    except Exception:
        num_nonzero = rank

    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()

    if num_nonzero < rank:
        # Try to find which columns can be removed
        current_rank = rank
        current_U = U_arr.copy()
        current_V = V_arr.copy()
        current_W = W_arr.copy()
        current_M = M.copy()

        while True:
            if time.time() > deadline - 5:
                break

            found_reduction = False
            for r_elim in range(current_rank):
                if time.time() > deadline - 5:
                    break

                other_cols = [s for s in range(current_rank) if s != r_elim]
                if len(other_cols) == 0:
                    break

                M_sub = current_M[:, other_cols]
                rhs = current_M[:, r_elim]

                try:
                    alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                    residual = np.max(np.abs(M_sub @ alpha - rhs))
                    if residual < 1e-8:
                        # Term r_elim is redundant! Remove it
                        new_rank = current_rank - 1
                        new_U = np.delete(current_U, r_elim, axis=1)
                        new_V = np.delete(current_V, r_elim, axis=1)
                        new_W = np.delete(current_W, r_elim, axis=1)

                        # Verify
                        err = verify_decomp(new_U, new_V, new_W, new_rank)
                        if err < 1e-10:
                            best_rank = new_rank
                            best_U = new_U
                            best_V = new_V
                            best_W = new_W

                            # Rebuild M with the reduced set
                            current_rank = new_rank
                            current_U = new_U
                            current_V = new_V
                            current_W = new_W
                            current_M = np.zeros((nm * mp * pn, current_rank), dtype=float)
                            for r in range(current_rank):
                                current_M[:, r] = np.einsum('a,b,c->abc', current_U[:, r], current_V[:, r], current_W[:, r]).flatten()

                            found_reduction = True
                            break
                except np.linalg.LinAlgError:
                    continue

            if not found_reduction:
                break

    # Convert back to lists with integer/rational values
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(val)
            result.append(row_list)
        return result

    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)

    return {"U": U_final, "V": V_final, "W": W_final}
