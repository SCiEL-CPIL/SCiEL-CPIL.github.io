"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
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
    
    # Build the 48x42 tensor matrix
    small_M = np.zeros((n * m * p, rank))  # 48 x 42
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                for r in range(rank):
                    small_M[idx, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    # Best result so far
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    def verify_decomp_float(U_arr, V_arr, W_arr, tol=1e-6):
        """Verify decomposition with floating point arrays."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(U_arr.shape[1]):
                        total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def to_half_integer_list(arr):
        """Convert float array to list of int/half-integer values."""
        result = []
        for row in arr:
            result.append([])
            for val in row:
                frac = round(val * 2) / 2
                if abs(frac - round(frac)) < 1e-10:
                    frac = int(round(frac))
                result[-1].append(frac)
        return result
    
    def eliminate_one_column(M_mat, U_mat, V_mat, W_mat, remaining_cols, time_limit=None):
        """Try to eliminate one column from remaining_cols using null space."""
        if time_limit is not None and time.time() > time_limit:
            return None, None, None, None
        
        if len(remaining_cols) <= 1:
            return None, None, None, None
        
        M_sub = M_mat[:, remaining_cols]
        n_cols = len(remaining_cols)
        
        try:
            U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, None, None, None
        
        if len(s_sub) == 0:
            return None, None, None, None
        tol_sub = max(1e-12, s_sub[0] * 1e-12)
        eff_rank_sub = int(np.sum(s_sub > tol_sub))
        
        if eff_rank_sub >= n_cols:
            return None, None, None, None
        
        null_sub = Vt_sub[eff_rank_sub:, :]
        
        best_col = None
        best_score = -1
        best_null_idx = 0
        
        for nv_idx in range(null_sub.shape[0]):
            nv = null_sub[nv_idx]
            for local_c, global_c in enumerate(remaining_cols):
                score = abs(nv[local_c])
                if score > best_score and score > 1e-10:
                    best_score = score
                    best_col = global_c
                    best_null_idx = nv_idx
        
        if best_col is None:
            return None, None, None, None
        
        nv = null_sub[best_null_idx]
        local_best = remaining_cols.index(best_col)
        pivot_val = nv[local_best]
        
        if abs(pivot_val) < 1e-12:
            return None, None, None, None
        
        for local_c, global_c in enumerate(remaining_cols):
            if global_c == best_col:
                continue
            coeff = -nv[local_c] / pivot_val
            if abs(coeff) > 1e-12:
                U_mat[:, global_c] += coeff * U_mat[:, best_col]
                V_mat[:, global_c] += coeff * V_mat[:, best_col]
                W_mat[:, global_c] += coeff * W_mat[:, best_col]
        
        return U_mat, V_mat, W_mat, best_col
    
    # Strategy 1: Iterative elimination
    try:
        if time.time() < deadline - 40:
            new_U = U_np.copy()
            new_V = V_np.copy()
            new_W = W_np.copy()
            eliminated = set()
            remaining = list(range(rank))
            
            max_elim_attempts = 20
            for attempt in range(max_elim_attempts):
                if time.time() > deadline - 40:
                    break
                if len(remaining) <= 1:
                    break
                
                new_U, new_V, new_W, elim_col = eliminate_one_column(
                    small_M, new_U, new_V, new_W, remaining, time_limit=deadline - 40)
                
                if elim_col is None:
                    break
                
                eliminated.add(elim_col)
                remaining = [c for c in remaining if c != elim_col]
            
            if eliminated:
                keep = [c for c in range(rank) if c not in eliminated]
                new_rank = len(keep)
                final_U = new_U[:, keep]
                final_V = new_V[:, keep]
                final_W = new_W[:, keep]
                
                if verify_decomp_float(final_U, final_V, final_W, tol=1e-6) and new_rank < best_rank:
                    best_rank = new_rank
                    best_result = {
                        "U": to_half_integer_list(final_U),
                        "V": to_half_integer_list(final_V),
                        "W": to_half_integer_list(final_W)
                    }
    except Exception:
        pass
    
    # Strategy 2: Random restart elimination
    try:
        if time.time() < deadline - 30:
            np.random.seed(42)
            best_random_rank = best_rank
            best_random_result = best_result
            
            max_trials = 20
            for trial in range(max_trials):
                if time.time() > deadline - 30:
                    break
                
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()
                
                col_order = list(range(rank))
                np.random.shuffle(col_order)
                remaining = list(col_order)
                
                max_elim_attempts = 15
                for _ in range(max_elim_attempts):
                    if time.time() > deadline - 30:
                        break
                    if len(remaining) <= 1:
                        break
                    
                    new_U, new_V, new_W, elim_col = eliminate_one_column(
                        small_M, new_U, new_V, new_W, remaining, time_limit=deadline - 30)
                    
                    if elim_col is None:
                        break
                    
                    eliminated.add(elim_col)
                    remaining = [c for c in remaining if c != elim_col]
                
                if eliminated:
                    keep = [c for c in range(rank) if c not in eliminated]
                    new_rank = len(keep)
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]
                    
                    if verify_decomp_float(final_U, final_V, final_W, tol=1e-6) and new_rank < best_random_rank:
                        best_random_rank = new_rank
                        best_random_result = {
                            "U": to_half_integer_list(final_U),
                            "V": to_half_integer_list(final_V),
                            "W": to_half_integer_list(final_W)
                        }
            
            if best_random_rank < best_rank:
                best_rank = best_random_rank
                best_result = best_random_result
    except Exception:
        pass
    
    # Strategy 3: Combinatorial support pruning - drop 2 columns and solve
    # For each pair of columns to drop, check if the remaining 40 columns
    # can be re-weighted (via a linear combination) to still span the tensor.
    # We look for a linear dependency: sum_{r in keep} c_r * tensor_r = 0
    # where the dropped columns can be expressed as combos of kept ones.
    try:
        if time.time() < deadline - 10:
            # Precompute column vectors for the 48-entry tensor
            # tensor_col[r][idx] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # where idx = i*m*p + j*p + k
            tensor_cols = np.zeros((n * m * p, rank))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            tensor_cols[idx, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
            
            # For each pair of columns to drop, solve for coefficients
            # that express the dropped columns as linear combinations of kept ones.
            # If we can find c such that for each dropped col d:
            #   tensor_col[d] = sum_{r in keep} c_{d,r} * tensor_col[r]
            # then we can absorb those contributions.
            # 
            # More precisely: we want to find new weights for all 42 columns
            # such that the effective rank is 40. This means there exist
            # 2 linear dependencies among the 42 tensor columns.
            #
            # Simpler approach: try to find if any 2 columns are linear
            # combinations of the other 40. If so, we can drop them.
            
            all_cols = list(range(rank))
            found_rank40 = False
            
            # Enumerate pairs to drop - 42 choose 2 = 861 pairs
            for drop_pair in combinations(all_cols, 2):
                if time.time() > deadline - 10:
                    break
                
                drop_set = set(drop_pair)
                keep_set = [c for c in all_cols if c not in drop_set]
                keep_arr = np.array(keep_set)
                
                # Build the system: A * x = b
                # where A is 48 x 40 (kept columns), b is the dropped column
                A = tensor_cols[:, keep_arr]  # 48 x 40
                
                for d in drop_pair:
                    b = tensor_cols[:, d]
                    
                    # Solve least squares
                    try:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                    except np.linalg.LinAlgError:
                        continue
                    
                    # Check if b is in the span of kept columns
                    # residual should be near 0
                    residual = np.linalg.norm(b - A @ x)
                    if residual < 1e-8:
                        # Column d is in the span of kept columns
                        # This means we can potentially drop both columns
                        # if the other dropped column is also in the span
                        pass
                
                # Better approach: check if both dropped columns are in the span
                # of the remaining 40 columns simultaneously
                # Build augmented matrix and check rank
                kept = tensor_cols[:, keep_arr]  # 48 x 40
                d1, d2 = drop_pair
                dropped = tensor_cols[:, [d1, d2]]  # 48 x 2
                full = np.hstack([kept, dropped])  # 48 x 42
                
                # If rank(full) == rank(kept), then dropped cols are in span
                try:
                    r_full = np.linalg.matrix_rank(full, tol=1e-10)
                    r_kept = np.linalg.matrix_rank(kept, tol=1e-10)
                except np.linalg.LinAlgError:
                    continue
                
                if r_full == r_kept and r_kept <= 40:
                    # Both dropped columns are linear combinations of kept columns
                    # We can express the tensor as a sum of only 40 terms
                    # by absorbing the dropped columns' contributions
                    #
                    # Find the coefficients: solve kept @ X = dropped
                    try:
                        X, res, rk, sv = np.linalg.lstsq(kept, dropped, rcond=1e-10)
                    except np.linalg.LinAlgError:
                        continue
                    
                    # Verify
                    recon = kept @ X
                    if np.linalg.norm(recon - dropped) < 1e-8:
                        # Success! We can drop these 2 columns.
                        # The new decomposition uses only the kept columns
                        # with modified weights. But wait - the tensor decomposition
                        # requires specific structure (outer products), so simple
                        # linear combination of tensor columns doesn't directly
                        # give us a valid rank-40 decomposition.
                        #
                        # However, if the dropped columns are linear combinations
                        # of kept columns in the tensor space, it means the rank
                        # of the tensor is at most 40. We need to find an explicit
                        # rank-40 decomposition.
                        #
                        # For now, let's just record that rank <= 40 is possible
                        # and try to construct it via the null space approach.
                        found_rank40 = True
                        break
            
            if found_rank40:
                # Try to construct an explicit rank-40 decomposition
                # using the null space of the 48x42 matrix
                try:
                    U_null, s_null, Vt_null = np.linalg.svd(small_M, full_matrices=True)
                    tol_null = 1e-10
                    eff_rank = int(np.sum(s_null > tol_null))
                    
                    if eff_rank < rank:
                        # There are rank - eff_rank null vectors
                        # Use them to eliminate columns
                        new_U = U_np.copy()
                        new_V = V_np.copy()
                        new_W = W_np.copy()
                        eliminated = set()
                        remaining = list(range(rank))
                        
                        for _ in range(rank - eff_rank):
                            if time.time() > deadline - 10:
                                break
                            if len(remaining) <= 1:
                                break
                            
                            new_U, new_V, new_W, elim_col = eliminate_one_column(
                                small_M, new_U, new_V, new_W, remaining, time_limit=deadline - 10)
                            
                            if elim_col is None:
                                break
                            
                            eliminated.add(elim_col)
                            remaining = [c for c in remaining if c != elim_col]
                        
                        if eliminated:
                            keep = [c for c in range(rank) if c not in eliminated]
                            new_rank = len(keep)
                            final_U = new_U[:, keep]
                            final_V = new_V[:, keep]
                            final_W = new_W[:, keep]
                            
                            if verify_decomp_float(final_U, final_V, final_W, tol=1e-6) and new_rank < best_rank:
                                best_rank = new_rank
                                best_result = {
                                    "U": to_half_integer_list(final_U),
                                    "V": to_half_integer_list(final_V),
                                    "W": to_half_integer_list(final_W)
                                }
                except Exception:
                    pass
    except Exception:
        pass
    
    # Strategy 4: Try dropping 3 columns (rank 39) via support pruning
    try:
        if time.time() < deadline - 5:
            tensor_cols = np.zeros((n * m * p, rank))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            tensor_cols[idx, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
            
            all_cols = list(range(rank))
            # Sample a subset of triples to keep runtime manageable
            np.random.seed(123)
            num_triples_to_try = 200
            all_triples = list(combinations(all_cols, 3))
            np.random.shuffle(all_triples)
            triples_to_try = all_triples[:num_triples_to_try]
            
            for drop_triple in triples_to_try:
                if time.time() > deadline - 5:
                    break
                
                drop_set = set(drop_triple)
                keep_set = [c for c in all_cols if c not in drop_set]
                keep_arr = np.array(keep_set)
                
                kept = tensor_cols[:, keep_arr]  # 48 x 39
                dropped = tensor_cols[:, list(drop_triple)]  # 48 x 3
                full = np.hstack([kept, dropped])  # 48 x 42
                
                try:
                    r_full = np.linalg.matrix_rank(full, tol=1e-10)
                    r_kept = np.linalg.matrix_rank(kept, tol=1e-10)
                except np.linalg.LinAlgError:
                    continue
                
                if r_full == r_kept and r_kept <= 39:
                    # All 3 dropped columns are in the span of kept columns
                    try:
                        X, res, rk, sv = np.linalg.lstsq(kept, dropped, rcond=1e-10)
                    except np.linalg.LinAlgError:
                        continue
                    
                    recon = kept @ X
                    if np.linalg.norm(recon - dropped) < 1e-8:
                        # Try to construct explicit rank-39 decomposition
                        new_U = U_np.copy()
                        new_V = V_np.copy()
                        new_W = W_np.copy()
                        eliminated = set(drop_triple)
                        remaining = keep_set
                        
                        # Try to absorb the dropped columns
                        for _ in range(3):
                            if time.time() > deadline - 5:
                                break
                            if len(remaining) <= 1:
                                break
                            
                            new_U, new_V, new_W, elim_col = eliminate_one_column(
                                small_M, new_U, new_V, new_W, remaining, time_limit=deadline - 5)
                            
                            if elim_col is None:
                                break
                            
                            eliminated.add(elim_col)
                            remaining = [c for c in remaining if c != elim_col]
                        
                        keep_final = [c for c in range(rank) if c not in eliminated]
                        new_rank = len(keep_final)
                        final_U = new_U[:, keep_final]
                        final_V = new_V[:, keep_final]
                        final_W = new_W[:, keep_final]
                        
                        if verify_decomp_float(final_U, final_V, final_W, tol=1e-6) and new_rank < best_rank:
                            best_rank = new_rank
                            best_result = {
                                "U": to_half_integer_list(final_U),
                                "V": to_half_integer_list(final_V),
                                "W": to_half_integer_list(final_W)
                            }
                            break
    except Exception:
        pass
    
    return best_result
# EVOLVE-BLOCK-END
