import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin vs 150s kill

    # Build the exact multiplication tensor T[a,b,c]
    # a = i*m+l (A[i,l]), b = l*p+k (B[l,k]), c = k*n+i (C[i,k])
    T = np.zeros((nm, mp, pn), dtype=float)
    T_indices = []
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                T_indices.append((a, b, c))
    T_idx_arr = np.array(T_indices, dtype=int)  # shape (32, 3)

    # All positions where T is zero
    all_positions = []
    nonzero_set = set()
    for (a, b, c) in T_indices:
        idx = a * mp * pn + b * pn + c
        nonzero_set.add(idx)
        all_positions.append((a, b, c))
    # Add a representative set of zero positions for verification
    zero_positions = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                idx = a * mp * pn + b * pn + c
                if idx not in nonzero_set:
                    zero_positions.append((a, b, c))
    # Use all zero positions for exact verification (1024 - 32 = 992)
    all_check_positions = np.array(T_indices + zero_positions, dtype=int)  # shape (1024, 3)
    all_check_vals = np.zeros(len(all_check_positions))
    for idx, (a, b, c) in enumerate(T_indices):
        all_check_vals[idx] = 1.0

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        """Check sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c."""
        if rank_val == 0:
            return 1.0
        S_vals = np.zeros(len(all_check_positions), dtype=float)
        for r in range(rank_val):
            u_r = U_mat[:, r]
            v_r = V_mat[:, r]
            w_r = W_mat[:, r]
            S_vals += u_r[all_check_positions[:, 0]] * v_r[all_check_positions[:, 1]] * w_r[all_check_positions[:, 2]]
        err = float(np.max(np.abs(S_vals - all_check_vals)))
        return err

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-9:
                    row_list.append(rounded)
                else:
                    row_list.append(val)
            result.append(row_list)
        return result

    # ---- Structural seed: Strassen rank-7 applied to four 2x2 blocks -> rank 28 ----
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: A_L B_TL -> C_LL
    add_block(0,  [0, 1, 4, 5], [0, 1, 4, 5],  [0, 2, 1, 3])
    # Block 2: A_R B_BL -> C_LR
    add_block(7,  [2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3])
    # Block 3: A_L B_TR -> C_RL
    add_block(14, [0, 1, 4, 5], [2, 3, 6, 7],  [4, 6, 5, 7])
    # Block 4: A_R B_BR -> C_RR
    add_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    if verify_decomp(U_arr, V_arr, W_arr, rank) > 1e-9:
        return {"U": to_list(U_arr), "V": to_list(V_arr), "W": to_list(W_arr)}

    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()

    # ---- Build full constraint matrix for rank reduction ----
    def build_full_constraint_matrix(U_mat, V_mat, W_mat, r):
        """Build the full 1024x r constraint matrix where row (a,b,c) = U[a,r]*V[b,r]*W[c,r]."""
        M = np.zeros((len(all_check_positions), r), dtype=float)
        for ri in range(r):
            u_r = U_mat[:, ri]
            v_r = V_mat[:, ri]
            w_r = W_mat[:, ri]
            M[:, ri] = u_r[all_check_positions[:, 0]] * v_r[all_check_positions[:, 1]] * w_r[all_check_positions[:, 2]]
        return M

    def try_remove_columns(U_cur, V_cur, W_cur, cols_to_remove):
        remaining = [c for c in range(U_cur.shape[1]) if c not in cols_to_remove]
        if len(remaining) == 0:
            return None
        U_new = U_cur[:, remaining]
        V_new = V_cur[:, remaining]
        W_new = W_cur[:, remaining]
        if verify_decomp(U_new, V_new, W_new, len(remaining)) < 1e-9:
            return (U_new, V_new, W_new)
        return None

    # Phase A: Full constraint-based linear dependency removal
    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = rank

    for iteration in range(30):
        if time.time() > deadline - 30:
            break
        if current_rank <= best_rank:
            break
        M = build_full_constraint_matrix(current_U, current_V, current_W, current_rank)
        removed_any = False
        for r_elim in range(current_rank):
            if time.time() > deadline - 25:
                break
            other_cols = [s for s in range(current_rank) if s != r_elim]
            M_sub = M[:, other_cols]
            rhs = M[:, r_elim]
            try:
                alpha, _, _, _ = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                residual = float(np.max(np.abs(M_sub @ alpha - rhs)))
                if residual < 1e-7:
                    result = try_remove_columns(current_U, current_V, current_W, [r_elim])
                    if result is not None:
                        current_U, current_V, current_W = result
                        current_rank -= 1
                        removed_any = True
                        if current_rank < best_rank:
                            best_rank = current_rank
                            best_U = current_U.copy()
                            best_V = current_V.copy()
                            best_W = current_W.copy()
                        break
            except np.linalg.LinAlgError:
                continue
        if not removed_any:
            break

    # Phase B: Term merging - try to merge two rank-1 terms into one
    # If U[:,i]*V[:,i]*W[:,i] + U[:,j]*V[:,j]*W[:,j] can be written as a single rank-1 term
    def try_merge_terms(U_cur, V_cur, W_cur, i, j):
        """Try to merge terms i and j into a single rank-1 term."""
        # The sum of two rank-1 tensors is rank-1 iff they are proportional
        # Check if the tensor T_i + T_j is rank-1
        T_i = np.einsum('a,b,c->abc', U_cur[:, i], V_cur[:, i], W_cur[:, i])
        T_j = np.einsum('a,b,c->abc', U_cur[:, j], V_cur[:, j], W_cur[:, j])
        T_sum = T_i + T_j
        
        # Check rank-1: flatten and check if it's a rank-1 matrix in any mode
        # Reshape to (nm, mp*pn) and check rank
        flat = T_sum.reshape(nm, mp * pn)
        try:
            sv = np.linalg.svd(flat, full_matrices=False)
            if len(sv) > 0 and sv[0] > 1e-10 and (len(sv) == 1 or sv[1] < 1e-10 * sv[0]):
                # It's rank-1, extract the components
                u_new = sv[0] * flat[:, 0] / np.linalg.norm(flat[:, 0]) if np.linalg.norm(flat[:, 0]) > 1e-10 else None
                if u_new is None:
                    return None
                # Actually, let's use a cleaner extraction
                Uf, Sf, Vt = np.linalg.svd(flat, full_matrices=False)
                if Sf[0] < 1e-10:
                    return None
                u = Uf[:, 0] * Sf[0]
                v_w = Vt[0].reshape(mp, pn)
                Vv, Sv, Wt = np.linalg.svd(v_w, full_matrices=False)
                if len(Sv) == 0 or Sv[0] < 1e-10:
                    return None
                v = Vv[:, 0] * Sv[0]
                w = Wt[0]
                
                # Round to half-integers
                u = np.round(u * 2) / 2.0
                v = np.round(v * 2) / 2.0
                w = np.round(w * 2) / 2.0
                
                # Verify the merged term is exact
                T_check = np.einsum('a,b,c->abc', u, v, w)
                if np.max(np.abs(T_check - T_sum)) > 1e-9:
                    return None
                
                # Build new matrices without i and j, with merged term
                keep = [k for k in range(U_cur.shape[1]) if k != i and k != j]
                U_new = np.column_stack([U_cur[:, keep], u]) if len(keep) > 0 else u[:, None]
                V_new = np.column_stack([V_cur[:, keep], v]) if len(keep) > 0 else v[:, None]
                W_new = np.column_stack([W_cur[:, keep], w]) if len(keep) > 0 else w[:, None]
                
                if verify_decomp(U_new, V_new, W_new, U_new.shape[1]) < 1e-9:
                    return (U_new, V_new, W_new)
        except np.linalg.LinAlgError:
            pass
        return None

    if time.time() < deadline - 25:
        for iter_m in range(20):
            if time.time() > deadline - 20:
                break
            merged_any = False
            for i in range(current_rank):
                if time.time() > deadline - 18:
                    break
                for j in range(i + 1, current_rank):
                    result = try_merge_terms(current_U, current_V, current_W, i, j)
                    if result is not None:
                        current_U, current_V, current_W = result
                        current_rank -= 1
                        merged_any = True
                        if current_rank < best_rank:
                            best_rank = current_rank
                            best_U = current_U.copy()
                            best_V = current_V.copy()
                            best_W = current_W.copy()
                        break
                if merged_any:
                    break
            if not merged_any:
                break

    # Phase C: Greedy peeling with full verification
    def greedy_peel(U_cur, V_cur, W_cur):
        cur_U = U_cur.copy()
        cur_V = V_cur.copy()
        cur_W = W_cur.copy()
        cur_r = cur_U.shape[1]
        residual = T.copy()
        for _ in range(cur_r):
            if time.time() > deadline - 15:
                break
            best = None
            for mode in range(3):
                if mode == 0:
                    flat = residual.transpose(0, 1, 2).reshape(nm, mp * pn)
                elif mode == 1:
                    flat = residual.transpose(1, 0, 2).reshape(mp, nm * pn)
                else:
                    flat = residual.transpose(2, 0, 1).reshape(pn, nm * mp)
                norm = float(np.linalg.norm(flat))
                if best is None or norm > best[0]:
                    best = (norm, mode, flat)
            if best is None or best[0] < 1e-9:
                break
            _, mode, flat = best
            try:
                Uf, Sf, Vt = np.linalg.svd(flat, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            s0 = float(Sf[0])
            if s0 < 1e-9:
                break
            if mode == 0:
                u = Uf[:, 0] * s0
                rest = Vt[0].reshape(mp, pn)
                Vv, Sv, Wt = np.linalg.svd(rest, full_matrices=False)
                if len(Sv) == 0:
                    break
                v = Vv[:, 0] * Sv[0]
                w = Wt[0]
            elif mode == 1:
                v = Uf[:, 0] * s0
                rest = Vt[0].reshape(nm, pn)
                Uu, Su, Wt = np.linalg.svd(rest, full_matrices=False)
                if len(Su) == 0:
                    break
                u = Uu[:, 0] * Su[0]
                w = Wt[0]
            else:
                w = Uf[:, 0] * s0
                rest = Vt[0].reshape(nm, mp)
                Uu, Su, Vt2 = np.linalg.svd(rest, full_matrices=False)
                if len(Su) == 0:
                    break
                u = Uu[:, 0] * Su[0]
                v = Vt2[0]

            u = np.asarray(u, dtype=float)
            v = np.asarray(v, dtype=float)
            w = np.asarray(w, dtype=float)
            u = np.round(u * 2) / 2.0
            v = np.round(v * 2) / 2.0
            w = np.round(w * 2) / 2.0

            new_U = np.column_stack([cur_U, u]) if cur_U.size else u[:, None]
            new_V = np.column_stack([cur_V, v]) if cur_V.size else v[:, None]
            new_W = np.column_stack([cur_W, w]) if cur_W.size else w[:, None]
            cur_r += 1
            residual = residual - np.einsum('a,b,c->abc', u, v, w)

            nz = [c for c in range(cur_r) if np.any(np.abs(new_U[:, c]) > 1e-12)
                  and np.any(np.abs(new_V[:, c]) > 1e-12)
                  and np.any(np.abs(new_W[:, c]) > 1e-12)]
            new_U = new_U[:, nz]
            new_V = new_V[:, nz]
            new_W = new_W[:, nz]
            cur_U, cur_V, cur_W = new_U, new_V, new_W
            cur_r = cur_U.shape[1]

            if verify_decomp(cur_U, cur_V, cur_W, cur_r) < 1e-9 and cur_r < best_rank:
                best_rank = cur_r
                best_U = cur_U.copy()
                best_V = cur_V.copy()
                best_W = cur_W.copy()
        return cur_U, cur_V, cur_W, cur_r

    if time.time() < deadline - 20:
        gU, gV, gW, gr = greedy_peel(U_arr, V_arr, W_arr)
        for iteration in range(20):
            if time.time() > deadline - 15:
                break
            if gr <= best_rank:
                break
            M = build_full_constraint_matrix(gU, gV, gW, gr)
            removed_any = False
            for r_elim in range(gr):
                if time.time() > deadline - 12:
                    break
                other_cols = [s for s in range(gr) if s != r_elim]
                M_sub = M[:, other_cols]
                rhs = M[:, r_elim]
                try:
                    alpha, _, _, _ = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                    residual = float(np.max(np.abs(M_sub @ alpha - rhs)))
                    if residual < 1e-7:
                        result = try_remove_columns(gU, gV, gW, [r_elim])
                        if result is not None:
                            gU, gV, gW = result
                            gr -= 1
                            removed_any = True
                            if gr < best_rank:
                                best_rank = gr
                                best_U = gU.copy()
                                best_V = gV.copy()
                                best_W = gW.copy()
                            break
                except np.linalg.LinAlgError:
                    continue
            if not removed_any:
                break

    # Final verification
    final_err = verify_decomp(best_U, best_V, best_W, best_rank)
    if final_err > 1e-9:
        best_U, best_V, best_W, best_rank = U_arr, V_arr, W_arr, 28
        final_err = verify_decomp(best_U, best_V, best_W, best_rank)
        if final_err > 1e-9:
            return {"U": to_list(U_arr), "V": to_list(V_arr), "W": to_list(W_arr)}

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
