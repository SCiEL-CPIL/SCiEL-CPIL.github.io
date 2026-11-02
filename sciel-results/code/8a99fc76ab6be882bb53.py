"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Using GQ-style decomposition: split middle dim 4 = 2+2, and for each 
    2x2 x 2x5 sub-problem, split the 5 columns as 2+2+1:
    - 2x2 * 2x2: Strassen (rank 7)
    - 2x2 * 2x1: schoolbook (rank 4)
    Total per sub-problem: 7 + 7 + 4 = 18
    Total rank: 2 * 18 = 36
    
    Then search for rank reduction via linear algebra.
    """
    import time
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    rank_terms = []  # Each term is (u_dict, v_dict, w_dict)
    
    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1}, 
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1}, 
             {(j0, k0+1): 1, (j0+1, k0+1): -1}, 
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1}, 
             {(j0+1, k0): 1, (j0, k0): -1}, 
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1}, 
             {(j0+1, k0+1): 1}, 
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1}, 
             {(j0, k0): 1, (j0, k0+1): 1}, 
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1}, 
             {(j0+1, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
    
    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
    
    # Build the full decomposition
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
    
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))
    
    base_rank = len(rank_terms)  # 36
    
    import numpy as np
    
    # Build U, V, W as numpy arrays directly
    def build_np_matrices(terms):
        r = len(terms)
        U = np.zeros((n * m, r), dtype=np.float64)
        V = np.zeros((m * p, r), dtype=np.float64)
        W = np.zeros((p * n, r), dtype=np.float64)
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff
        return U, V, W
    
    U_base, V_base, W_base = build_np_matrices(rank_terms)
    
    # Build the constraint matrix: M[eq_idx, col]
    # eq_idx = i*m*p + j*p + k
    num_eq = n * m * p  # 40
    M_full = np.zeros((num_eq, base_rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for col in range(base_rank):
                    M_full[eq_idx, col] = (U_base[i * m + j, col] * 
                                           V_base[j * p + k, col] * 
                                           W_base[k * n + i, col])
    
    b_vec = np.ones(num_eq, dtype=np.float64)
    
    # Verify base decomposition
    base_check = M_full @ np.ones(base_rank)
    assert np.allclose(base_check, b_vec, atol=1e-10), "Base decomposition is invalid!"
    
    def check_half_integer_np(arr):
        """Check if all entries are integer or half-integer (numpy version)."""
        nonzero = arr[arr != 0]
        if len(nonzero) == 0:
            return True
        doubled = nonzero * 2
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=1e-8)
    
    def try_solve_subset(remaining_indices):
        """
        Try to solve M[:, remaining_indices] @ alpha = b.
        If solvable, try to find a valid factorization with half-integer entries.
        Returns (U, V, W) as numpy arrays or None.
        """
        r = len(remaining_indices)
        if r == 0:
            return None
        
        M_sub = M_full[:, remaining_indices]
        
        # Check if system is consistent using SVD
        try:
            U_svd, s_vals, Vt = np.linalg.svd(M_sub, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        
        tol = 1e-10
        rank_M = np.sum(s_vals > tol)
        
        if rank_M == 0:
            return None
        
        # Check consistency: project b onto column space
        # b should be in col(M_sub)
        proj_b = U_svd[:, :rank_M] @ (U_svd[:, :rank_M].T @ b_vec)
        if not np.allclose(proj_b, b_vec, atol=1e-8):
            return None
        
        # Find a solution - use least squares
        alpha, residuals, rank, sv = np.linalg.lstsq(M_sub, b_vec, rcond=tol)
        
        # Verify
        if not np.allclose(M_sub @ alpha, b_vec, atol=1e-8):
            return None
        
        # Check if alpha entries are "nice" (allowing half-integer factorization)
        # Strategy: try different ways to distribute the scaling
        # For each term, alpha[i] = U_scale * V_scale * W_scale
        # We want U_scale, V_scale, W_scale to be half-integers
        
        # Simple case: if all |alpha[i]| are 1, we're done
        if np.allclose(np.abs(alpha), 1, atol=1e-8):
            U_final = np.zeros((n * m, r), dtype=np.float64)
            V_final = np.zeros((m * p, r), dtype=np.float64)
            W_final = np.zeros((p * n, r), dtype=np.float64)
            
            for idx, col in enumerate(remaining_indices):
                a = alpha[idx]
                u_dict, v_dict, w_dict = rank_terms[col]
                for (i, j), coeff in u_dict.items():
                    U_final[i * m + j, idx] = a * coeff
                for (j, k), coeff in v_dict.items():
                    V_final[j * p + k, idx] = float(coeff)
                for (i, k), coeff in w_dict.items():
                    W_final[k * n + i, idx] = float(coeff)
            
            if check_half_integer_np(U_final):
                # Final verification
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U_final[i * m + j] * V_final[j * p + k] * W_final[k * n + i])
                            if abs(s - 1) > 1e-8:
                                return None
                return (U_final, V_final, W_final)
        
        # Try: scale all of alpha into U
        U_final = np.zeros((n * m, r), dtype=np.float64)
        V_final = np.zeros((m * p, r), dtype=np.float64)
        W_final = np.zeros((p * n, r), dtype=np.float64)
        
        for idx, col in enumerate(remaining_indices):
            a = alpha[idx]
            u_dict, v_dict, w_dict = rank_terms[col]
            for (i, j), coeff in u_dict.items():
                U_final[i * m + j, idx] = a * coeff
            for (j, k), coeff in v_dict.items():
                V_final[j * p + k, idx] = float(coeff)
            for (i, k), coeff in w_dict.items():
                W_final[k * n + i, idx] = float(coeff)
        
        if check_half_integer_np(U_final):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_final[i * m + j] * V_final[j * p + k] * W_final[k * n + i])
                        if abs(s - 1) > 1e-8:
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                return (U_final, V_final, W_final)
        
        # Try: scale into V
        U_final2 = np.zeros((n * m, r), dtype=np.float64)
        V_final2 = np.zeros((m * p, r), dtype=np.float64)
        W_final2 = np.zeros((p * n, r), dtype=np.float64)
        
        for idx, col in enumerate(remaining_indices):
            a = alpha[idx]
            u_dict, v_dict, w_dict = rank_terms[col]
            for (i, j), coeff in u_dict.items():
                U_final2[i * m + j, idx] = float(coeff)
            for (j, k), coeff in v_dict.items():
                V_final2[j * p + k, idx] = a * coeff
            for (i, k), coeff in w_dict.items():
                W_final2[k * n + i, idx] = float(coeff)
        
        if check_half_integer_np(V_final2):
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_final2[i * m + j] * V_final2[j * p + k] * W_final2[k * n + i])
                        if abs(s - 1) > 1e-8:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return (U_final2, V_final2, W_final2)
        
        # Try: scale into W
        U_final3 = np.zeros((n * m, r), dtype=np.float64)
        V_final3 = np.zeros((m * p, r), dtype=np.float64)
        W_final3 = np.zeros((p * n, r), dtype=np.float64)
        
        for idx, col in enumerate(remaining_indices):
            a = alpha[idx]
            u_dict, v_dict, w_dict = rank_terms[col]
            for (i, j), coeff in u_dict.items():
                U_final3[i * m + j, idx] = float(coeff)
            for (j, k), coeff in v_dict.items():
                V_final3[j * p + k, idx] = float(coeff)
            for (i, k), coeff in w_dict.items():
                W_final3[k * n + i, idx] = a * coeff
        
        if check_half_integer_np(W_final3):
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_final3[i * m + j] * V_final3[j * p + k] * W_final3[k * n + i])
                        if abs(s - 1) > 1e-8:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return (U_final3, V_final3, W_final3)
        
        # Try: split alpha = sqrt(|alpha|) * sign into U and V
        for split_power in [0.25, 0.33, 0.5, 0.66, 0.75]:
            U_test = np.zeros((n * m, r), dtype=np.float64)
            V_test = np.zeros((m * p, r), dtype=np.float64)
            W_test = np.zeros((p * n, r), dtype=np.float64)
            
            valid_split = True
            for idx, col in enumerate(remaining_indices):
                a = alpha[idx]
                if a < -1e-10:
                    # Negative: put sign in U
                    abs_a = -a
                    u_scale = abs_a ** split_power
                    v_scale = abs_a ** (1 - split_power)
                else:
                    abs_a = a
                    u_scale = abs_a ** split_power
                    v_scale = abs_a ** (1 - split_power)
                
                u_dict, v_dict, w_dict = rank_terms[col]
                sign = -1 if a < -1e-10 else 1
                
                for (i, j), coeff in u_dict.items():
                    U_test[i * m + j, idx] = sign * u_scale * coeff
                for (j, k), coeff in v_dict.items():
                    V_test[j * p + k, idx] = v_scale * coeff
                for (i, k), coeff in w_dict.items():
                    W_test[k * n + i, idx] = float(coeff)
            
            if check_half_integer_np(U_test) and check_half_integer_np(V_test):
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U_test[i * m + j] * V_test[j * p + k] * W_test[k * n + i])
                            if abs(s - 1) > 1e-8:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    return (U_test, V_test, W_test)
        
        return None
    
    # Search for best subset
    best_result = None
    best_rank = base_rank
    
    search_deadline = start_time + 135  # 5s safety margin
    
    # Strategy 1: Try removing 1 term at a time
    for t in range(base_rank):
        if time.time() > search_deadline:
            break
        remaining = [i for i in range(base_rank) if i != t]
        result = try_solve_subset(remaining)
        if result is not None:
            U_final, V_final, W_final = result
            if len(U_final[0]) < best_rank:
                best_rank = len(U_final[0])
                best_result = (U_final, V_final, W_final)
    
    # Strategy 2: Try removing 2 terms
    if best_rank <= 35 and time.time() < search_deadline:
        from itertools import combinations
        for t1, t2 in combinations(range(base_rank), 2):
            if time.time() > search_deadline:
                break
            remaining = [i for i in range(base_rank) if i not in (t1, t2)]
            result = try_solve_subset(remaining)
            if result is not None:
                U_final, V_final, W_final = result
                if len(U_final[0]) < best_rank:
                    best_rank = len(U_final[0])
                    best_result = (U_final, V_final, W_final)
    
    # Strategy 3: Try removing 3 terms
    if best_rank <= 34 and time.time() < search_deadline:
        from itertools import combinations
        for t1, t2, t3 in combinations(range(base_rank), 3):
            if time.time() > search_deadline:
                break
            remaining = [i for i in range(base_rank) if i not in (t1, t2, t3)]
            result = try_solve_subset(remaining)
            if result is not None:
                U_final, V_final, W_final = result
                if len(U_final[0]) < best_rank:
                    best_rank = len(U_final[0])
                    best_result = (U_final, V_final, W_final)
    
    # Strategy 4: Try removing 4 terms (if time allows)
    if best_rank <= 33 and time.time() < search_deadline:
        from itertools import combinations
        for t1, t2, t3, t4 in combinations(range(base_rank), 4):
            if time.time() > search_deadline:
                break
            remaining = [i for i in range(base_rank) if i not in (t1, t2, t3, t4)]
            result = try_solve_subset(remaining)
            if result is not None:
                U_final, V_final, W_final = result
                if len(U_final[0]) < best_rank:
                    best_rank = len(U_final[0])
                    best_result = (U_final, V_final, W_final)
    
    if best_result is not None:
        # Convert to list of lists for the interface
        return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}
    
    # Fallback: return the base decomposition
    return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
# EVOLVE-BLOCK-END
