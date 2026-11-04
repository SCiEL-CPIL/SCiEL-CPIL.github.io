"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

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
    
    # Verify the decomposition
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
    
    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor matrix: column r = flattened (u_r ⊗ v_r ⊗ w_r)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    # Best result so far
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
    
    # Strategy 1: SVD-based null space elimination with multiple restarts
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-9
        effective_rank = int(np.sum(s_vals > tol))
        
        if effective_rank < rank:
            # Multiple elimination strategies
            for trial in range(30):
                if time.time() > deadline - 20:
                    break
                
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()
                
                # Different random seeds for each trial
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
                    
                    # Pick best elimination column
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
    
    # Strategy 2: Try to find a rank-40 decomposition by checking if the 
    # tensor space has rank <= 40 using a more careful analysis
    try:
        if time.time() < deadline - 30 and best_rank > 40:
            # Check if we can eliminate 2 more columns
            # Try all pairs of columns to see if eliminating either helps
            new_U = U_np.copy()
            new_V = V_np.copy()
            new_W = W_np.copy()
            eliminated = set()
            
            # First pass: standard elimination
            for iteration in range(10):
                if time.time() > deadline - 25:
                    break
                
                remaining = [c for c in range(rank) if c not in eliminated]
                if len(remaining) <= 1:
                    break
                
                M_sub = M[:, remaining]
                
                try:
                    U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
                except np.linalg.LinAlgError:
                    break
                
                tol_sub = 1e-12
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
                
                if verify_float(final_U, final_V, final_W, new_rank, tol=1e-9) and new_rank < best_rank:
                    best_rank = new_rank
                    best_result = {
                        "U": to_list(final_U),
                        "V": to_list(final_V),
                        "W": to_list(final_W)
                    }
    except Exception:
        pass
    
    # Strategy 3: High-precision polish with exact local solve
    # After finding a good decomposition, try to reduce rank further by 
    # solving a system of equations to find dependencies
    try:
        if time.time() < deadline - 15 and best_rank > 40:
            # Use the best decomposition found so far
            if best_rank < rank:
                best_U_np = np.array(best_result["U"], dtype=float)
                best_V_np = np.array(best_result["V"], dtype=float)
                best_W_np = np.array(best_result["W"], dtype=float)
            else:
                best_U_np = U_np
                best_V_np = V_np
                best_W_np = W_np
            
            current_rank = best_U_np.shape[1]
            
            # Try to eliminate one more column at a time
            for attempt in range(current_rank - 40):
                if time.time() > deadline - 15:
                    break
                
                # Try each possible column to eliminate
                best_elimination = None
                best_new_rank = current_rank
                
                for col_to_elim in range(current_rank):
                    if time.time() > deadline - 15:
                        break
                    
                    # Try to eliminate this column
                    new_U = best_U_np.copy()
                    new_V = best_V_np.copy()
                    new_W = best_W_np.copy()
                    
                    # Check if we can express this column as a combination of others
                    # by solving a linear system
                    remaining = [c for c in range(current_rank) if c != col_to_elim]
                    
                    if len(remaining) == 0:
                        continue
                    
                    # For each tensor component, check if the column can be expressed
                    # as a linear combination of the remaining columns
                    # This is equivalent to checking if the column is in the span of others
                    
                    M_sub = M[:, remaining]
                    M_col = M[:, col_to_elim]
                    
                    try:
                        # Solve M_sub * x = M_col using least squares
                        coeffs, residuals, rank_sub, singular_vals = np.linalg.lstsq(M_sub, M_col, rcond=1e-10)
                        
                        # Check if the solution is good
                        residual = np.linalg.norm(M_sub @ coeffs - M_col)
                        if residual < 1e-8:
                            # Check if we can actually eliminate this column
                            # by modifying the other columns
                            new_rank = current_rank - 1
                            
                            # Apply the elimination
                            for local_c, global_c in enumerate(remaining):
                                if abs(coeffs[local_c]) > 1e-10:
                                    new_U[:, global_c] += coeffs[local_c] * new_U[:, col_to_elim]
                                    new_V[:, global_c] += coeffs[local_c] * new_V[:, col_to_elim]
                                    new_W[:, global_c] += coeffs[local_c] * new_W[:, col_to_elim]
                            
                            # Remove the column
                            final_U = new_U[:, remaining]
                            final_V = new_V[:, remaining]
                            final_W = new_W[:, remaining]
                            
                            if verify_float(final_U, final_V, final_W, new_rank, tol=1e-9):
                                if new_rank < best_new_rank:
                                    best_new_rank = new_rank
                                    best_elimination = (col_to_elim, final_U, final_V, final_W)
                    except Exception:
                        continue
                
                if best_elimination is not None:
                    col_to_elim, final_U, final_V, final_W = best_elimination
                    best_rank = best_new_rank
                    best_result = {
                        "U": to_list(final_U),
                        "V": to_list(final_V),
                        "W": to_list(final_W)
                    }
                    # Update for next iteration
                    best_U_np = final_U
                    best_V_np = final_V
                    best_W_np = final_W
                else:
                    break
    except Exception:
        pass
    
    return best_result
# EVOLVE-BLOCK-END
