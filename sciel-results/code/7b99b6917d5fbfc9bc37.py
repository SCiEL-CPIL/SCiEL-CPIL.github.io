import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = 2, 2, 6

    # Rank 21 decomposition using 3 blocks of Strassen (7 terms each)
    rank = 21
    U_arr = np.zeros((n * m, rank), dtype=int)
    V_arr = np.zeros((m * p, rank), dtype=int)
    W_arr = np.zeros((p * n, rank), dtype=int)

    # Strassen terms for A (2x2)
    # Indices: 0:A00, 1:A01, 2:A10, 3:A11
    U_terms = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]

    # Strassen terms for B (2x2 block)
    V_terms_block = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B0,c0 + B1,c1
        [(0, 0, 1)],                 # M2: B0,c0
        [(0, 1, 1), (1, 1, -1)],     # M3: B0,c1 - B1,c1
        [(1, 0, 1), (0, 0, -1)],     # M4: B1,c0 - B0,c0
        [(1, 1, 1)],                 # M5: B1,c1
        [(0, 0, 1), (0, 1, 1)],      # M6: B0,c0 + B0,c1
        [(1, 0, 1), (1, 1, 1)]       # M7: B1,c0 + B1,c1
    ]

    # Strassen terms for C (2x2 block)
    W_terms_block = [
        [(0, 0, 1), (1, 1, 1)],   # M1: C0,d0(1), C1,d1(1)
        [(1, 0, 1), (1, 1, -1)],  # M2: C1,d0(1), C1,d1(-1)
        [(0, 1, 1), (1, 1, 1)],   # M3: C0,d1(1), C1,d1(1)
        [(0, 0, 1), (1, 0, 1)],   # M4: C0,d0(1), C1,d0(1)
        [(0, 0, -1), (0, 1, 1)],  # M5: C0,d0(-1), C0,d1(1)
        [(1, 1, 1)],              # M6: C1,d1(1)
        [(0, 0, 1)]               # M7: C0,d0(1)
    ]

    for block in range(3):
        col_off = 2 * block
        for t in range(7):
            r = block * 7 + t
            for idx, val in U_terms[t].items():
                U_arr[idx, r] = val
            for j_loc, k_loc, val in V_terms_block[t]:
                gk = col_off + k_loc
                V_arr[j_loc * p + gk, r] = val
            for i, k_loc, val in W_terms_block[t]:
                gk = col_off + k_loc
                W_arr[gk * n + i, r] = val

    def verify_decomposition(U, V, W):
        """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
        n, m, p = 2, 2, 6
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True

    # Verify rank 21 works
    if not verify_decomposition(U_arr, V_arr, W_arr):
        # Shouldn't happen, but return as-is
        return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

    # Try to find rank-20 by eliminating one product
    # Strategy: for each product r, check if it can be expressed as a linear
    # combination of other products. This means there exist coefficients c_s
    # such that for all (i,j,k): U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] =
    # sum_{s!=r} c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
    
    # Build the tensor T of shape (4, 12, 12) where T[i*m+j, j*p+k, k*n+i] 
    # = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Actually, let's think differently.
    
    # For each rank r, define the "slice" contribution:
    # For product r, it contributes to entry (i,j,k) the value:
    #   U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # We want to find if product r can be removed, i.e., if there exist
    # coefficients c_s (s != r) such that:
    #   U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 
    #     sum_{s != r} c_s * U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,r]
    
    # This is a system of linear equations in the c_s's.
    # Number of equations: n*m*p = 2*2*6 = 24
    # Number of unknowns: rank-1 = 20
    
    # Let's set up the linear system for each candidate r to eliminate.
    
    # Build matrix A_r of shape (24, 20) and vector b_r of shape (24,)
    # where row (i,j,k) corresponds to equation:
    #   sum_{s != r} c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    best_result = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
    best_rank = 21
    
    if time.time() < deadline - 5.0:
        for r_elim in range(21):
            if time.time() > deadline - 5.0:
                break
            
            # Build linear system
            # Equations indexed by (i, j, k)
            # Unknowns indexed by s in [0..20] \ {r_elim}
            
            unknown_indices = [s for s in range(21) if s != r_elim]
            num_unknowns = len(unknown_indices)
            
            A_mat = np.zeros((24, num_unknowns), dtype=float)
            b_vec = np.zeros(24, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        b_vec[eq_idx] = U_arr[u_idx, r_elim] * V_arr[v_idx, r_elim] * W_arr[w_idx, r_elim]
                        
                        for u_idx_pos, s in enumerate(unknown_indices):
                            A_mat[eq_idx, u_idx_pos] = U_arr[u_idx, s] * V_arr[v_idx, s] * W_arr[w_idx, s]
                        
                        eq_idx += 1
            
            # Solve the linear system
            try:
                # Check if system is consistent
                # Use least squares first, then verify exactness
                if np.allclose(A_mat, 0):
                    if not np.allclose(b_vec, 0):
                        continue
                    # Trivial solution
                    coeffs = np.zeros(num_unknowns)
                else:
                    # Try to find exact rational solution
                    # First check rank
                    rank_A = np.linalg.matrix_rank(A_mat)
                    rank_aug = np.linalg.matrix_rank(np.column_stack([A_mat, b_vec]))
                    
                    if rank_A != rank_aug:
                        continue  # Inconsistent
                    
                    # Find a solution
                    coeffs, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    
                    # Verify the solution is exact
                    A_coeffs = A_mat @ coeffs
                    if not np.allclose(A_coeffs, b_vec, atol=1e-10):
                        continue
                    
                    # Check if coefficients are rational (integer or half-integer)
                    # For exact decomposition, we need integer/half-integer entries
                    # Check if 2*coeffs are integers
                    two_coeffs = 2 * coeffs
                    if not np.allclose(two_coeffs, np.round(two_coeffs), atol=1e-10):
                        continue
                    
                    # Great! We found a valid elimination.
                    # Construct the rank-20 decomposition
                    new_rank = 20
                    U_new = np.zeros((n * m, new_rank), dtype=int)
                    V_new = np.zeros((m * p, new_rank), dtype=int)
                    W_new = np.zeros((p * n, new_rank), dtype=int)
                    
                    for new_r, s in enumerate(unknown_indices):
                        U_new[:, new_r] = U_arr[:, s]
                        V_new[:, new_r] = V_arr[:, s]
                        W_new[:, new_r] = W_arr[:, s]
                    
                    # Verify
                    if verify_decomposition(U_new, V_new, W_new):
                        best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                        best_rank = 20
                        break  # Found rank 20, stop searching
                    
            except Exception:
                continue
    
    if time.time() > deadline:
        return best_result
    
    return best_result
