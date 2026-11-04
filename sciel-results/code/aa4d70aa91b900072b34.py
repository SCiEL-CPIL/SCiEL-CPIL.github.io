"""Improved rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication."""

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the rank-42 Strassen decomposition
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (24, rank)  
    # W has shape (p*n, rank) = (12, rank)
    # Convention: W[k*n + i, r] contributes to C[i,k]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
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
    def verify_decomp(U_mat, V_mat, W_mat, rk):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rk):
                        total += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback to schoolbook
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
    
    # Convert to float arrays for numerical computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def to_list(arr):
        result = []
        for row in arr:
            result.append([])
            for val in row:
                # Try to find closest half-integer or integer
                frac_val = round(val * 2) / 2.0
                if abs(frac_val - round(frac_val)) < 1e-10:
                    frac_val = int(round(frac_val))
                result[-1].append(frac_val)
        return result
    
    def verify_float(final_U, final_V, final_W, new_rank, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(new_rank):
                        total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    # Build the tensor matrix: each column r is the flattened outer product
    # M has shape (n*m*p, rank) = (2304, 42)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    # Best result so far
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Strategy: Systematic null space elimination
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-9
        effective_rank = int(np.sum(s_vals > tol))
        
        if effective_rank < rank:
            # Multiple elimination strategies
            strategies = ['greedy_max', 'greedy_min', 'random']
            best_elim_rank = rank
            best_elim_result = None
            
            for strat in strategies:
                if time.time() > deadline - 30:
                    break
                
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()
                
                num_iters = rank - effective_rank + 5
                
                for iteration in range(num_iters):
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
                    
                    tol_sub = 1e-9
                    eff_rank_sub = int(np.sum(s_sub > tol_sub))
                    
                    if eff_rank_sub >= len(remaining):
                        break
                    
                    null_sub = Vt_sub[eff_rank_sub:, :]
                    
                    if null_sub.shape[0] == 0:
                        break
                    
                    # Choose pivot based on strategy
                    best_col = None
                    best_score = -1
                    best_null_idx = 0
                    
                    if strat == 'greedy_max':
                        for nv_idx in range(null_sub.shape[0]):
                            nv = null_sub[nv_idx]
                            for local_c, global_c in enumerate(remaining):
                                score = abs(nv[local_c])
                                if score > best_score and score > 1e-8:
                                    best_score = score
                                    best_col = global_c
                                    best_null_idx = nv_idx
                    elif strat == 'greedy_min':
                        for nv_idx in range(null_sub.shape[0]):
                            nv = null_sub[nv_idx]
                            for local_c, global_c in enumerate(remaining):
                                score = -abs(nv[local_c])
                                if score > best_score and abs(nv[local_c]) > 1e-8:
                                    best_score = score
                                    best_col = global_c
                                    best_null_idx = nv_idx
                    else:  # random
                        np.random.seed(iteration * 1000 + len(strat))
                        for nv_idx in range(null_sub.shape[0]):
                            nv = null_sub[nv_idx]
                            for local_c, global_c in enumerate(remaining):
                                score = abs(nv[local_c]) * (0.5 + np.random.random())
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
                    
                    if verify_float(final_U, final_V, final_W, new_rank) and new_rank < best_elim_rank:
                        best_elim_rank = new_rank
                        best_elim_result = {
                            "U": to_list(final_U),
                            "V": to_list(final_V),
                            "W": to_list(final_W)
                        }
            
            if best_elim_rank < best_rank:
                best_rank = best_elim_rank
                best_result = best_elim_result
    except Exception:
        pass
    
    # Strategy 2: Random elimination with more trials
    try:
        if time.time() < deadline - 50:
            np.random.seed(12345)
            best_random_rank = best_rank
            best_random_result = best_result
            
            num_trials = 30
            for trial in range(num_trials):
                if time.time() > deadline - 45:
                    break
                
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()
                
                col_order = list(range(rank))
                np.random.shuffle(col_order)
                
                max_elim = rank - effective_rank + 5
                for _ in range(max_elim):
                    if time.time() > deadline - 40:
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
                    
                    if null_sub.shape[0] == 0:
                        break
                    
                    nv_idx = np.random.randint(null_sub.shape[0])
                    nv = null_sub[nv_idx]
                    
                    local_best = np.argmax(np.abs(nv))
                    best_col = remaining[local_best]
                    
                    if abs(nv[local_best]) < 1e-10:
                        continue
                    
                    pivot_val = nv[local_best]
                    
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
                    
                    if verify_float(final_U, final_V, final_W, new_rank) and new_rank < best_random_rank:
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
    
    return best_result

# EVOLVE-BLOCK-END
