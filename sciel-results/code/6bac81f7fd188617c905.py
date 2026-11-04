"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
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
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
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
    
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
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
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor matrix: column r = flattened (u_r ⊗ v_r ⊗ w_r)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
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
    
    def verify_float(Uf, Vf, Wf, rnk, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rnk):
                        total += Uf[i*m+j, r] * Vf[j*p+k, r] * Wf[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    # Check SVD for linear dependencies
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol_svd = 1e-9
        effective_rank = int(np.sum(s_vals > tol_svd))
    except Exception:
        effective_rank = rank
    
    # Strategy 1: SVD-based null space elimination
    try:
        if effective_rank < rank:
            for trial in range(10):
                if time.time() > deadline - 20:
                    break
                
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()
                
                np.random.seed(42 + trial * 137)
                col_order = list(range(rank))
                np.random.shuffle(col_order)
                
                for iteration in range(rank - effective_rank + 5):
                    if time.time() > deadline - 20:
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
                            score = abs(nv[local_c]) + np.random.random() * 0.05
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
                    
                    if verify_float(final_U, final_V, final_W, new_rank, tol=1e-7) and new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": to_list(final_U),
                            "V": to_list(final_V),
                            "W": to_list(final_W)
                        }
    except Exception:
        pass
    
    # Strategy 2: Try to reduce rank by 1 via linear dependency check
    # Check if any column can be expressed as linear combination of others
    # by solving least squares for each candidate
    try:
        if time.time() < deadline - 40 and best_rank > 41:
            # For each column, check if it's in the span of the others
            for target_col in range(rank):
                if time.time() > deadline - 35:
                    break
                
                others = [c for c in range(rank) if c != target_col]
                M_others = M[:, others]
                M_target = M[:, target_col]
                
                # Solve M_others @ coeffs = M_target
                try:
                    coeffs, residuals, _, _ = np.linalg.lstsq(M_others, M_target, rcond=1e-10)
                except Exception:
                    continue
                
                # Check residual
                residual = np.linalg.norm(M_target - M_others @ coeffs)
                if residual < 1e-7:
                    # Column is linearly dependent! Remove it.
                    new_U = U_np.copy()
                    new_V = V_np.copy()
                    new_W = W_np.copy()
                    # The coefficient vector tells us the relationship, but since
                    # we're removing the column, we just drop it
                    keep = [c for c in range(rank) if c != target_col]
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]
                    
                    if verify_float(final_U, final_V, final_W, len(keep), tol=1e-7) and len(keep) < best_rank:
                        best_rank = len(keep)
                        best_result = {
                            "U": to_list(final_U),
                            "V": to_list(final_V),
                            "W": to_list(final_W)
                        }
                        break  # Only need to remove one
    except Exception:
        pass
    
    # Strategy 3: Try a different block partition that might yield lower rank
    # Use overlapping Strassen blocks or a different decomposition
    try:
        if time.time() < deadline - 30 and best_rank > 41:
            # Try: partition A into 2×2 blocks, B into 4×2 blocks, but use
            # a shared multiplication structure
            # Alternative: use rank-8 for M_{2,2,3} blocks
            # M_{2,2,3} has rank 8. We need 2 (A-col blocks) × 3 (B-col blocks) = 6 blocks
            # But A blocks are shared, so it's more like a tensor network
            # 
            # Try constructing with explicit rank-8 M_{2,2,3} decompositions
            # M_{2,2,3}: A(2x2), B(2x3), C(2x3)
            # Known rank-8 decomposition for 2x2*2x3:
            # We can use Strassen for the first 2x2 part and add one term for the 3rd col
            # Or use a known optimal decomposition
            #
            # Let's try: for each pair (A_col_block, B_col_block), use rank 8
            # But this gives 6*8 = 48 which is worse.
            #
            # Instead, try to find if rank 41 is achievable by checking
            # the border rank or using a different construction.
            
            # Try random search: perturb the Strassen decomposition and see
            # if we can find a valid rank-41 decomposition
            np.random.seed(12345)
            for attempt in range(20):
                if time.time() > deadline - 25:
                    break
                
                # Try removing one column and adjusting others
                target_col = attempt % rank
                others = [c for c in range(rank) if c != target_col]
                
                # Get the target column's contribution
                u_t = U_np[:, target_col].copy()
                v_t = V_np[:, target_col].copy()
                w_t = W_np[:, target_col].copy()
                
                # Try to express this as combination of others by solving
                # for each (i,j,k) triple: sum_r u[i,j,r]*v[j,k,r]*w[k,i,r] = 1
                # After removing target, we need to adjust remaining columns
                # This is underdetermined, so use least squares on the tensor
                # 
                # Build system: for each (i,j,k), the equation is
                # sum_{r in others} u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 - (contribution of target)
                # This is nonlinear. Instead, fix u and v, solve for w, or use
                # a linearization approach.
                #
                # Simpler: just check if removing the column and scaling others
                # works. Try uniform scaling.
                scale = 1.0
                new_U = U_np[:, others].copy()
                new_V = V_np[:, others].copy()
                new_W = W_np[:, others].copy()
                
                # Check if the sum of remaining terms equals the full tensor
                # without the target term
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total_without = 0.0
                            for idx, r in enumerate(others):
                                total_without += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                            target_contrib = U_np[i*m+j, target_col] * V_np[j*p+k, target_col] * W_np[k*n+i, target_col]
                            if abs(total_without + target_contrib - 1.0) > 1e-9:
                                scale = -1
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    # All consistent, check if just dropping works
                    if verify_float(new_U, new_V, new_W, len(others), tol=1e-7):
                        best_rank = len(others)
                        best_result = {
                            "U": to_list(new_U),
                            "V": to_list(new_V),
                            "W": to_list(new_W)
                        }
                        break
    except Exception:
        pass
    
    # Strategy 4: Final polish - try to round to half-integers and verify
    if best_rank < rank:
        try:
            U_rounded = to_list(np.array(best_result["U"], dtype=float))
            V_rounded = to_list(np.array(best_result["V"], dtype=float))
            W_rounded = to_list(np.array(best_result["W"], dtype=float))
            
            U_r = np.array(U_rounded, dtype=float)
            V_r = np.array(V_rounded, dtype=float)
            W_r = np.array(W_rounded, dtype=float)
            
            if verify_float(U_r, V_r, W_r, best_rank, tol=1e-6):
                best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
