import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build rank-42 Strassen decomposition
    rank_full = 42
    U_full = np.zeros((nm, rank_full), dtype=float)
    V_full = np.zeros((mp, rank_full), dtype=float)
    W_full = np.zeros((pn, rank_full), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset, U, V, W):
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)
        
        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)
        
        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)
        
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0
        
        U[a10, offset + 1] += 1.0; U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0; W[w11, offset + 1] -= 1.0
        
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0; V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0; W[w11, offset + 2] += 1.0
        
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0; V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0; W[w10, offset + 3] += 1.0
        
        U[a00, offset + 4] += 1.0; U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0; W[w01, offset + 4] += 1.0
        
        U[a10, offset + 5] += 1.0; U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0; V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0
        
        U[a01, offset + 6] += 1.0; U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0; V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset, U_full, V_full, W_full)
                offset += 7

    def verify(U_mat, V_mat, W_mat, rank, tol=1e-8):
        T_computed = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        valid_mask = np.zeros((nm, mp, pn), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid_mask[a, b, c] = True
                    if abs(T_computed[a, b, c] - 1.0) > tol:
                        return False
        zero_entries = ~valid_mask
        if np.max(np.abs(T_computed[zero_entries])) > tol:
            return False
        return True

    # Initial best
    if verify(U_full, V_full, W_full, rank_full):
        best_rank = rank_full
        best_U, best_V, best_W = U_full.copy(), V_full.copy(), W_full.copy()
    else:
        # Fallback
        best_rank = n * m * p
        best_U = np.zeros((nm, best_rank))
        best_V = np.zeros((mp, best_rank))
        best_W = np.zeros((pn, best_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    best_U[i * m + j, r] = 1
                    best_V[j * p + k, r] = 1
                    best_W[k * n + i, r] = 1
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try to reduce rank using alternating minimization
    # We try to reduce to rank 38, 39, 40, 41
    target_ranks = [38, 39, 40, 41]
    
    # Precompute valid entries for faster verification if needed, 
    # but einsum is fast enough for small tensors.
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
            
        # We need to select 42 - target_rank columns to remove.
        # Let's try random subsets of columns to remove, or structured ones.
        # Given the block structure, removing columns from different blocks might be better.
        # But let's try a few random attempts first.
        
        num_remove = rank_full - target_rank
        attempts = 50 # Limit attempts per target rank to stay within time
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
                
            # Randomly select columns to remove
            remove_indices = np.random.choice(rank_full, num_remove, replace=False)
            keep_indices = np.setdiff1d(np.arange(rank_full), remove_indices)
            
            # Initialize with Strassen values
            U = U_full[:, keep_indices].copy()
            V = V_full[:, keep_indices].copy()
            W = W_full[:, keep_indices].copy()
            
            # Alternating minimization
            # We want to minimize || sum_t U[:,t] V[:,t]^T W[c,t] - T ||^2
            # For fixed V, W: Linear in U.
            # For fixed U, W: Linear in V.
            # For fixed U, V: Linear in W.
            
            converged = False
            for _ in range(50): # Max iterations
                if time.time() > deadline - 10:
                    break
                    
                # Solve for U
                # Eq: sum_t U[a,t] * (W[c,t] * V[b,t]) = T[a,b,c]
                # Vectorize: 
                # A_U[a, b, c, t] = W[c,t] * V[b,t]
                # This is too big.
                # Instead, for each c, we have a matrix equation:
                # sum_t W[c,t] U[:,t] V[:,t]^T = T[:,:,c]
                # Let M_c = T[:,:,c] (nm x mp)
                # We want to find U such that sum_t W[c,t] U[:,t] V[:,t]^T approx M_c
                # This is linear in U.
                # Let's stack the equations for all c.
                # Unknowns: U (nm x r)
                # For each c, a, b: sum_t U[a,t] W[c,t] V[b,t] = T[a,b,c]
                
                # Build sparse system for U
                rows_u = []
                cols_u = []
                vals_u = []
                rhs_u = []
                
                # We can subsample equations if too many, but 2304 is small.
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            # Coefficients for U[a, t]
                            for t_idx, t in enumerate(keep_indices):
                                coeff = W[c, t_idx] * V[b, t_idx]
                                if abs(coeff) > 1e-15:
                                    row_idx = len(rhs_u)
                                    col_idx = a * target_rank + t_idx
                                    rows_u.append(row_idx)
                                    cols_u.append(col_idx)
                                    vals_u.append(coeff)
                            # Determine target
                            i_a = a // m
                            j_a = a % m
                            j_b = b // p
                            k_b = b % p
                            k_c = c // n
                            i_c = c % n
                            if i_a == i_c and j_a == j_b and k_b == k_c:
                                rhs_u.append(1.0)
                            else:
                                rhs_u.append(0.0)
                                
                if len(rhs_u) > 0:
                    A_u = np.zeros((len(rhs_u), nm * target_rank))
                    for i in range(len(rows_u)):
                        A_u[rows_u[i], cols_u[i]] = vals_u[i]
                    y_u = np.array(rhs_u)
                    try:
                        U_flat, _, _, _ = np.linalg.lstsq(A_u, y_u, rcond=None)
                        U = U_flat.reshape(nm, target_rank)
                    except:
                        pass
                        
                # Solve for V
                rows_v = []
                cols_v = []
                vals_v = []
                rhs_v = []
                
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            for t_idx, t in enumerate(keep_indices):
                                coeff = W[c, t_idx] * U[a, t_idx]
                                if abs(coeff) > 1e-15:
                                    row_idx = len(rhs_v)
                                    col_idx = b * target_rank + t_idx
                                    rows_v.append(row_idx)
                                    cols_v.append(col_idx)
                                    vals_v.append(coeff)
                            i_a = a // m
                            j_a = a % m
                            j_b = b // p
                            k_b = b % p
                            k_c = c // n
                            i_c = c % n
                            if i_a == i_c and j_a == j_b and k_b == k_c:
                                rhs_v.append(1.0)
                            else:
                                rhs_v.append(0.0)
                                
                if len(rhs_v) > 0:
                    A_v = np.zeros((len(rhs_v), mp * target_rank))
                    for i in range(len(rows_v)):
                        A_v[rows_v[i], cols_v[i]] = vals_v[i]
                    y_v = np.array(rhs_v)
                    try:
                        V_flat, _, _, _ = np.linalg.lstsq(A_v, y_v, rcond=None)
                        V = V_flat.reshape(mp, target_rank)
                    except:
                        pass
                        
                # Solve for W
                rows_w = []
                cols_w = []
                vals_w = []
                rhs_w = []
                
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            for t_idx, t in enumerate(keep_indices):
                                coeff = V[b, t_idx] * U[a, t_idx]
                                if abs(coeff) > 1e-15:
                                    row_idx = len(rhs_w)
                                    col_idx = c * target_rank + t_idx
                                    rows_w.append(row_idx)
                                    cols_w.append(col_idx)
                                    vals_w.append(coeff)
                            i_a = a // m
                            j_a = a % m
                            j_b = b // p
                            k_b = b % p
                            k_c = c // n
                            i_c = c % n
                            if i_a == i_c and j_a == j_b and k_b == k_c:
                                rhs_w.append(1.0)
                            else:
                                rhs_w.append(0.0)
                                
                if len(rhs_w) > 0:
                    A_w = np.zeros((len(rhs_w), pn * target_rank))
                    for i in range(len(rows_w)):
                        A_w[rows_w[i], cols_w[i]] = vals_w[i]
                    y_w = np.array(rhs_w)
                    try:
                        W_flat, _, _, _ = np.linalg.lstsq(A_w, y_w, rcond=None)
                        W = W_flat.reshape(pn, target_rank)
                    except:
                        pass

            # Verify result
            if verify(U, V, W, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break # Found a valid decomposition for this target rank, try lower? No, target_ranks is sorted desc. So this is the best for this loop.
                # Actually, we want the lowest rank. So if we find 38, we can stop.
                # But the loop goes 38, 39, 40, 41. So if we find 38, we break the outer loop?
                # No, we break the inner loop. Then we should break the outer loop too if we found the lowest possible?
                # The lowest possible is 38 in this list.
                # So we can break.
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
