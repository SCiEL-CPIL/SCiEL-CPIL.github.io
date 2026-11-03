"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 4, 6

    # Helper to convert float array to list with integer/half-integer entries
    def to_list(arr):
        result = []
        for row in arr:
            result.append([])
            for val in row:
                frac = round(val * 2) / 2
                if abs(frac - round(frac)) < 1e-10:
                    frac = int(round(frac))
                result[-1].append(frac)
        return result

    # Helper to verify decomposition validity
    def verify_decomp(U_list, V_list, W_list, rnk):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rnk):
                        total += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if total != 1:
                        return False
        return True

    # Build the matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Convention: W[k*n + i, r] contributes to C[i,k]
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]

    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]

    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)

    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]

    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)

    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]

    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)

    if not verify_decomp(U, V, W, rank):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}

    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Build the tensor matrix: column r = flattened (u_r ⊗ v_r ⊗ w_r)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (n*m*p, rank) = (48, 42)

    # Best result so far
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank

    # Strategy 1: SVD-based null space elimination
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-9
        effective_rank = int(np.sum(s_vals > tol))

        if effective_rank < rank:
            new_U = U_np.copy()
            new_V = V_np.copy()
            new_W = W_np.copy()
            eliminated = set()

            for iteration in range(rank - effective_rank + 5):
                if time.time() > deadline - 20:
                    break

                remaining = [c for c in range(rank) if c not in eliminated]
                if len(remaining) <= 1:
                    break

                M_sub = M[:, remaining]

                try:
                    U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
                except np.linalg.LinAlgError:
                    break

                tol_sub = 1e-9
                eff_rank_sub = int(np.sum(s_sub > tol_sub))

                if eff_rank_sub >= len(remaining):
                    break

                null_sub = Vt_sub[eff_rank_sub:, :]

                # Try all null vectors and pick the best elimination
                best_col = None
                best_score = -1
                best_null_idx = 0

                for nv_idx in range(null_sub.shape[0]):
                    nv = null_sub[nv_idx]
                    for local_c, global_c in enumerate(remaining):
                        score = abs(nv[local_c])
                        if score > best_score and score > 1e-8:
                            best_score = score
                            best_col = global_c
                            best_null_idx = nv_idx

                if best_col is None:
                    break

                nv = null_sub[best_null_idx]
                local_best = remaining.index(best_col)
                pivot_val = nv[local_best]

                if abs(pivot_val) < 1e-10:
                    continue

                for local_c, global_c in enumerate(remaining):
                    if global_c == best_col:
                        continue
                    coeff = -nv[local_c] / pivot_val
                    if abs(coeff) > 1e-10:
                        new_U[:, global_c] += coeff * new_U[:, best_col]
                        new_V[:, global_c] += coeff * new_V[:, best_col]
                        new_W[:, global_c] += coeff * new_W[:, best_col]

                eliminated.add(best_col)

            if eliminated:
                keep = [c for c in range(rank) if c not in eliminated]
                new_rank = len(keep)
                final_U = new_U[:, keep]
                final_V = new_V[:, keep]
                final_W = new_W[:, keep]

                # Verify with tighter tolerance
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for r in range(new_rank):
                                total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                            if abs(total - 1.0) > 1e-7:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break

                if valid and new_rank < best_rank:
                    best_rank = new_rank
                    best_result = {
                        "U": to_list(final_U),
                        "V": to_list(final_V),
                        "W": to_list(final_W)
                    }
    except Exception:
        pass

    # Strategy 2: Try structural simplification - look for pairs of terms that
    # can be combined. If two terms share the same U or V pattern, they might
    # be mergeable.
    try:
        if time.time() < deadline - 30:
            # Try to find if any two columns of U are proportional
            for i in range(rank):
                if time.time() > deadline - 25:
                    break
                for j in range(i + 1, rank):
                    if np.linalg.norm(U_np[:, i]) > 1e-10 and np.linalg.norm(U_np[:, j]) > 1e-10:
                        ratio = np.dot(U_np[:, i], U_np[:, j]) / (np.linalg.norm(U_np[:, i]) ** 2)
                        if abs(ratio) > 1e-10:
                            residual = U_np[:, j] - ratio * U_np[:, i]
                            if np.linalg.norm(residual) / np.linalg.norm(U_np[:, j]) < 1e-10:
                                pass  # Complex to handle, skip for now
    except Exception:
        pass

    # Strategy 3: Try random restarts with different elimination orderings
    try:
        if time.time() < deadline - 30:
            np.random.seed(42)
            best_random_rank = best_rank
            best_random_result = best_result

            for trial in range(20):
                if time.time() > deadline - 25:
                    break

                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()

                col_order = list(range(rank))
                np.random.shuffle(col_order)

                for _ in range(rank - effective_rank + 3):
                    if time.time() > deadline - 25:
                        break

                    remaining = [c for c in col_order if c not in eliminated]
                    if len(remaining) <= 1:
                        break

                    M_sub = M[:, remaining]

                    try:
                        U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
                    except np.linalg.LinAlgError:
                        break

                    tol_sub = 1e-9
                    eff_rank_sub = int(np.sum(s_sub > tol_sub))

                    if eff_rank_sub >= len(remaining):
                        break

                    null_sub = Vt_sub[eff_rank_sub:, :]

                    best_col = None
                    best_score = -1
                    best_null_idx = 0

                    for nv_idx in range(null_sub.shape[0]):
                        nv = null_sub[nv_idx]
                        for local_c, global_c in enumerate(remaining):
                            score = abs(nv[local_c]) + np.random.random() * 0.1
                            if score > best_score and abs(nv[local_c]) > 1e-8:
                                best_score = score
                                best_col = global_c
                                best_null_idx = nv_idx

                    if best_col is None:
                        break

                    nv = null_sub[best_null_idx]
                    local_best = remaining.index(best_col)
                    pivot_val = nv[local_best]

                    if abs(pivot_val) < 1e-10:
                        continue

                    for local_c, global_c in enumerate(remaining):
                        if global_c == best_col:
                            continue
                        coeff = -nv[local_c] / pivot_val
                        if abs(coeff) > 1e-10:
                            new_U[:, global_c] += coeff * new_U[:, best_col]
                            new_V[:, global_c] += coeff * new_V[:, best_col]
                            new_W[:, global_c] += coeff * new_W[:, best_col]

                    eliminated.add(best_col)

                if eliminated:
                    keep = [c for c in range(rank) if c not in eliminated]
                    new_rank = len(keep)
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]

                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0.0
                                for r in range(new_rank):
                                    total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                                if abs(total - 1.0) > 1e-7:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break

                    if valid and new_rank < best_random_rank:
                        best_random_rank = new_rank
                        best_random_result = {
                            "U": to_list(final_U),
                            "V": to_list(final_V),
                            "W": to_list(final_W)
                        }

            if best_random_rank < best_rank:
                best_rank = best_random_rank
                best_result = best_random_result
    except Exception:
        pass

    # Strategy 4: Aggressive multi-pass elimination with different orderings
    try:
        if time.time() < deadline - 15:
            # Try multiple elimination strategies with different column orderings
            np.random.seed(123)
            for trial in range(50):
                if time.time() > deadline - 10:
                    break
                
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()
                
                # Random column ordering
                col_order = list(range(rank))
                np.random.shuffle(col_order)
                
                max_elim = rank - 36  # try to get down to rank 36
                for iteration in range(max_elim + 3):
                    if time.time() > deadline - 8:
                        break
                    
                    remaining = [c for c in col_order if c not in eliminated]
                    if len(remaining) <= 1:
                        break
                    
                    M_sub = M[:, remaining]
                    
                    try:
                        U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
                    except np.linalg.LinAlgError:
                        break
                    
                    eff_rank_sub = int(np.sum(s_sub > 1e-9))
                    
                    if eff_rank_sub >= len(remaining):
                        break
                    
                    null_sub = Vt_sub[eff_rank_sub:, :]
                    
                    # Try each null vector and pick the elimination that maximizes
                    # the number of zeros created in the U, V, W matrices
                    best_col = None
                    best_sparsity_gain = -1
                    best_null_idx = 0
                    
                    for nv_idx in range(null_sub.shape[0]):
                        nv = null_sub[nv_idx]
                        for local_c, global_c in enumerate(remaining):
                            if abs(nv[local_c]) < 1e-8:
                                continue
                            
                            # Estimate sparsity gain: how many new zeros would be created
                            pivot_val = nv[local_c]
                            gain = 0
                            for other_local, other_global in enumerate(remaining):
                                if other_global == global_c:
                                    continue
                                coeff = -nv[other_local] / pivot_val
                                # Check if this would create zeros in U, V, W
                                u_row = new_U[:, global_c] + coeff * new_U[:, other_global]
                                v_row = new_V[:, global_c] + coeff * new_V[:, other_global]
                                w_row = new_W[:, global_c] + coeff * new_W[:, other_global]
                                gain += np.sum(np.abs(u_row) < 1e-8) + np.sum(np.abs(v_row) < 1e-8) + np.sum(np.abs(w_row) < 1e-8)
                            
                            if gain > best_sparsity_gain:
                                best_sparsity_gain = gain
                                best_col = global_c
                                best_null_idx = nv_idx
                    
                    if best_col is None:
                        break
                    
                    nv = null_sub[best_null_idx]
                    local_best = remaining.index(best_col)
                    pivot_val = nv[local_best]
                    
                    if abs(pivot_val) < 1e-10:
                        continue
                    
                    for local_c, global_c in enumerate(remaining):
                        if global_c == best_col:
                            continue
                        coeff = -nv[local_c] / pivot_val
                        if abs(coeff) > 1e-10:
                            new_U[:, global_c] += coeff * new_U[:, best_col]
                            new_V[:, global_c] += coeff * new_V[:, best_col]
                            new_W[:, global_c] += coeff * new_W[:, best_col]
                    
                    eliminated.add(best_col)
                
                if eliminated:
                    keep = [c for c in range(rank) if c not in eliminated]
                    new_rank = len(keep)
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0.0
                                for r in range(new_rank):
                                    total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                                if abs(total - 1.0) > 1e-7:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid and new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": to_list(final_U),
                            "V": to_list(final_V),
                            "W": to_list(final_W)
                        }
    except Exception:
        pass

    return best_result
# EVOLVE-BLOCK-END
