import numpy as np
import time
from itertools import combinations

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _build_rank93():
    n, m, p = 4, 4, 7
    total_rank = 93
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c
    
    def add_strassen_2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal current_rank
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = A_rows[li]
                j = A_cols[lj]
                U[i * m + j, r] += cx
                
            for (li, lj), cy in y_sub.items():
                i = B_rows[li]
                j = B_cols[lj]
                V[i * p + j, r] += cy
                
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = C_rows[li]
                k_final = C_cols[lj]
                W[k_final * n + i_final, r] += cc

    A00_r = [0, 1]; A00_c = [0, 1]
    A01_r = [0, 1]; A01_c = [2, 3]
    A10_r = [2, 3]; A10_c = [0, 1]
    A11_r = [2, 3]; A11_c = [2, 3]
    
    B_top_r = [0, 1]; B_top_c = [4, 5]
    B_bot_r = [2, 3]; B_bot_c = [4, 5]
    
    C_top_r = [0, 1]; C_top_c = [4, 5]
    C_bot_r = [2, 3]; C_bot_c = [4, 5]
    
    add_strassen_2x2(A00_r, A00_c, B_top_r, B_top_c, C_top_r, C_top_c)
    add_strassen_2x2(A01_r, A01_c, B_bot_r, B_bot_c, C_top_r, C_top_c)
    add_strassen_2x2(A10_r, A10_c, B_top_r, B_top_c, C_bot_r, C_bot_c)
    add_strassen_2x2(A11_r, A11_c, B_bot_r, B_bot_c, C_bot_r, C_bot_c)
    
    for i in range(n):
        for j in range(m):
            r = current_rank
            current_rank += 1
            U[i * m + j, r] = 1.0
            V[j * p + 6, r] = 1.0
            W[6 * n + i, r] = 1.0
    
    assert current_rank == total_rank, f"Expected {total_rank}, got {current_rank}"
    
    return U[:, :total_rank], V[:, :total_rank], W[:, :total_rank]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = U.shape[1]
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against the identity tensor (which is 1 when k==j, 0 otherwise for standard matmul)
    # Actually for matrix multiplication A(n,m) @ B(m,p) -> C(n,p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 if this contributes, which is always 1 for the multiplication tensor
    # The decomposition requires: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k
    # Wait, that's not right. The tensor for matmul has T[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # So we need this sum to equal 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_time=30.0):
    """Try to find a decomposition of target_rank by zeroing out terms and solving."""
    start_time = time.time()
    orig_rank = U.shape[1]
    
    # Strategy: try removing terms one at a time, check if remaining system is solvable
    # The tensor equation: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    # This is a system of n*m*p = 4*4*7 = 112 equations
    # With rank R, we have 3*R unknowns (entries of U, V, W)
    # We need 3*R >= 112, so R >= 38 (but that's a lower bound)
    
    # Better approach: use the structure. Try to identify redundant terms.
    # For each term r, check if its contribution can be absorbed by other terms.
    
    best_rank = orig_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try greedy elimination: for each term, check if removing it still allows exact solution
    # with the remaining terms (by solving a least-squares problem)
    
    active = list(range(orig_rank))
    
    while time.time() - start_time < max_time:
        found_removal = False
        for idx in range(len(active)):
            if time.time() - start_time > max_time:
                break
            r = active[idx]
            # Try removing term r
            new_active = active[:idx] + active[idx+1:]
            if len(new_active) < target_rank:
                break
            
            # Build the system: we have 3*len(new_active) unknowns
            # The equations are nonlinear (trilinear), so we can't directly use linear algebra
            # Instead, we can fix two of the three matrices and solve for the third.
            
            # Fix V and W, solve for U:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # For fixed (j,k): sum_r U[i*m+j,r] * (V[j*p+k,r]*W[k*n+i,r]) = 1
            # This is linear in U for each (i,j,k) but the coefficients depend on i through W[k*n+i,r]
            
            # Let's try: fix V and W, solve for U via least squares
            # Unknowns: U has n*m*len(new_active) entries
            # Equations: n*m*p = 112
            # For equation (i,j,k): sum_r U[i*m+j, r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            n_unknowns = n * m * len(new_active)
            n_eqs = n * m * p
            
            if n_unknowns < n_eqs:
                # Underdetermined, skip
                continue
            
            # Build A matrix and b vector
            # A is n_eqs x n_unknowns
            # This could be very large. n_eqs=112, n_unknowns up to 16*46=736
            # That's manageable.
            
            A = np.zeros((n_eqs, n_unknowns))
            b = np.ones(n_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r_idx, r in enumerate(new_active):
                            # U[i*m+j, r] is at position (i*m+j)*len(new_active) + r_idx
                            u_idx = (i * m + j) * len(new_active) + r_idx
                            A[eq_idx, u_idx] = V[j*p+k, r] * W[k*n+i, r]
            
            try:
                U_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if solution is exact
            if np.max(np.abs(A @ U_sol - b)) < 1e-9:
                # Great! We found a solution. Now check if entries are half-integers
                U_sol = U_sol.reshape(n * m, len(new_active))
                # Check if all entries are half-integers
                if np.all(np.abs(U_sol * 2 - np.round(U_sol * 2)) < 1e-6):
                    active = new_active
                    best_rank = len(new_active)
                    # Reconstruct full U, V, W
                    best_U = np.zeros((n*m, best_rank))
                    best_V = np.zeros((m*p, best_rank))
                    best_W = np.zeros((p*n, best_rank))
                    for new_r, old_r in enumerate(active):
                        best_U[:, new_r] = U_sol[:, new_r]
                        best_V[:, new_r] = V[:, old_r]
                        best_W[:, new_r] = W[:, old_r]
                    found_removal = True
                    break
        
        if not found_removal:
            break
    
    return best_rank, best_U, best_V, best_W

def construct():
    n, m, p = 4, 4, 7
    
    # Build the rank-93 baseline
    U93, V93, W93 = _build_rank93()
    
    # Verify baseline
    assert _verify_decomposition(U93, V93, W93, n, m, p), "Baseline verification failed"
    
    best_U, best_V, best_W = U93, V93, W93
    best_rank = 93
    
    # Try to reduce rank using available time
    time_budget = min(120.0, _time_remaining() - 10.0)  # Leave 10s safety margin
    if time_budget > 5.0:
        for target in range(92, 50, -1):
            if time.time() - _START_TIME > TIME_LIMIT - 15:
                break
            max_time_for_target = min(20.0, _time_remaining() - 15.0)
            if max_time_for_target < 3.0:
                break
            rank, U_t, V_t, W_t = _try_reduce_rank(U93, V93, W93, n, m, p, target, max_time=max_time_for_target)
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U_t, V_t, W_t
                # Verify
                if _verify_decomposition(best_U, best_V, best_W, n, m, p):
                    break  # Found improvement, stop searching
                else:
                    # Revert
                    best_U, best_V, best_W = U93, V93, W93
                    best_rank = 93
    
    # Final safety check
    if _time_remaining() < 10.0:
        best_U, best_V, best_W = U93, V93, W93
    
    U = best_U.tolist()
    V = best_V.tolist()
    W = best_W.tolist()
    
    return {"U": U, "V": V, "W": W}
