import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Build rank-49 decomposition via Kronecker product of Strassen factors
    i_vals = np.repeat(np.arange(4), 4)
    j_vals = np.tile(np.arange(4), 4)
    i_hi = (i_vals // 2).astype(int)
    i_lo = (i_vals % 2).astype(int)
    j_hi = (j_vals // 2).astype(int)
    j_lo = (j_vals % 2).astype(int)

    u_hi_vecs = U2[i_hi * 2 + j_hi]
    u_lo_vecs = U2[i_lo * 2 + j_lo]
    u_outer = u_hi_vecs[:, :, np.newaxis] * u_lo_vecs[:, np.newaxis, :]
    u_mat = u_outer.reshape(16, 49)

    j_vals_v = np.repeat(np.arange(4), 4)
    k_vals_v = np.tile(np.arange(4), 4)
    j_hi_v = (j_vals_v // 2).astype(int)
    j_lo_v = (j_vals_v % 2).astype(int)
    k_hi_v = (k_vals_v // 2).astype(int)
    k_lo_v = (k_vals_v % 2).astype(int)

    v_hi_vecs = V2[j_hi_v * 2 + k_hi_v]
    v_lo_vecs = V2[j_lo_v * 2 + k_lo_v]
    v_outer = v_hi_vecs[:, :, np.newaxis] * v_lo_vecs[:, np.newaxis, :]
    v_mat = v_outer.reshape(16, 49)

    k_vals_w = np.repeat(np.arange(4), 4)
    i_vals_w = np.tile(np.arange(4), 4)
    k_hi_w = (k_vals_w // 2).astype(int)
    k_lo_w = (k_vals_w % 2).astype(int)
    i_hi_w = (i_vals_w // 2).astype(int)
    i_lo_w = (i_vals_w % 2).astype(int)

    w_hi_vecs = W2[k_hi_w * 2 + i_hi_w]
    w_lo_vecs = W2[k_lo_w * 2 + i_lo_w]
    w_outer = w_hi_vecs[:, :, np.newaxis] * w_lo_vecs[:, np.newaxis, :]
    w_mat = w_outer.reshape(16, 49)

    # Verify correctness of rank-49
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                prod = np.sum(u_mat[i * 4 + j] * v_mat[j * 4 + k] * w_mat[k * 4 + i])
                if prod != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid or time.time() - start_time > deadline:
        n, m, p = 4, 4, 4
        rank = n * m * p
        U_f = [[0] * rank for _ in range(n * m)]
        V_f = [[0] * rank for _ in range(m * p)]
        W_f = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_f[ii * m + jj][col] = 1
                    V_f[jj * p + kk][col] = 1
                    W_f[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    # Pre-compute term vectors
    i_idx = np.repeat(np.arange(4), 4)
    j_idx = np.tile(np.arange(4), 4)
    k_idx = np.tile(np.arange(4), 16)
    
    i_hi = i_idx // 2
    i_lo = i_idx % 2
    j_hi = j_idx // 2
    j_lo = j_idx % 2
    k_hi = k_idx // 2
    k_lo = k_idx % 2
    
    high_comp = np.zeros((7, 64), dtype=int)
    low_comp = np.zeros((7, 64), dtype=int)
    
    for s in range(7):
        for idx in range(64):
            i = idx // 16
            j = (idx // 4) % 4
            k = idx % 4
            ih, il = i // 2, i % 2
            jh, jl = j // 2, j % 2
            kh, kl = k // 2, k % 2
            high_comp[s, idx] = U2[ih*2+jh, s] * V2[jh*2+kh, s] * W2[kh*2+ih, s]
            low_comp[s, idx] = U2[il*2+jl, s] * V2[jl*2+kl, s] * W2[kl*2+il, s]
    
    terms = np.zeros((49, 64), dtype=int)
    for a in range(7):
        for b in range(7):
            terms[a*7+b] = high_comp[a] * low_comp[b]
    
    T = np.ones(64, dtype=int)
    
    # Find the rank of the terms matrix and a minimal spanning set
    # Use QR decomposition with column pivoting to find a basis
    Q, R, piv = np.linalg.qr(terms.T, mode='complete', pivoting=True)
    
    # The first r columns (where r = rank) form a basis
    # Find the rank
    rank_terms = np.linalg.matrix_rank(terms)
    
    # Get the pivot indices for the first rank_terms columns
    # These are the indices of the linearly independent terms
    basis_indices = piv[:rank_terms]
    
    # Solve for coefficients using only the basis terms
    M_basis = terms[basis_indices].T  # (64, rank_terms)
    lambda_sol, residuals, rank_M, s_vals = np.linalg.lstsq(M_basis, T, rcond=None)
    recon = M_basis @ lambda_sol
    
    if np.max(np.abs(recon - T)) < 1e-9:
        # Check if lambda values are integers or half-integers
        valid_lambda = True
        for val in lambda_sol:
            val_round = round(val)
            val_half = round(val * 2) / 2.0
            if abs(val - val_round) > 1e-6 and abs(val - val_half) > 1e-6:
                valid_lambda = False
                break
        
        if valid_lambda:
            new_U = u_mat[:, basis_indices].copy()
            new_V = v_mat[:, basis_indices].copy()
            new_W = w_mat[:, basis_indices].copy()
            for idx, lam in enumerate(lambda_sol):
                new_U[:, idx] *= lam
                new_V[:, idx] *= lam
                new_W[:, idx] *= lam
            
            # Final verification
            valid_final = True
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        prod = np.dot(new_U[i*4+j], new_V[j*4+k] * new_W[k*4+i])
                        if abs(prod - 1) > 1e-6:
                            valid_final = False
                            break
                    if not valid_final:
                        break
                if not valid_final:
                    break
            
            if valid_final:
                return {
                    "U": new_U.tolist(),
                    "V": new_V.tolist(),
                    "W": new_W.tolist()
                }
    
    # If the basis approach didn't work, try the original one-at-a-time removal
    norms = np.linalg.norm(terms, axis=1)
    order = np.argsort(norms)
    
    for c_to_remove in order:
        if time.time() - start_time > deadline:
            break
        
        remaining = [c for c in range(49) if c != c_to_remove]
        M = terms[remaining].T  # Shape (64, 48)
        
        try:
            lambda_sol, residuals, rank_M, s_vals = np.linalg.lstsq(M, T, rcond=None)
            recon = M @ lambda_sol
            if np.max(np.abs(recon - T)) < 1e-9:
                valid_lambda = True
                for val in lambda_sol:
                    val_round = round(val)
                    val_half = round(val * 2) / 2.0
                    if abs(val - val_round) > 1e-6 and abs(val - val_half) > 1e-6:
                        valid_lambda = False
                        break
                
                if valid_lambda:
                    new_U = u_mat[:, remaining].copy()
                    new_V = v_mat[:, remaining].copy()
                    new_W = w_mat[:, remaining].copy()
                    for idx, lam in enumerate(lambda_sol):
                        new_U[:, idx] *= lam
                        new_V[:, idx] *= lam
                        new_W[:, idx] *= lam
                    
                    valid_final = True
                    for i in range(4):
                        for j in range(4):
                            for k in range(4):
                                prod = np.dot(new_U[i*4+j], new_V[j*4+k] * new_W[k*4+i])
                                if abs(prod - 1) > 1e-6:
                                    valid_final = False
                                    break
                            if not valid_final:
                                break
                        if not valid_final:
                            break
                    
                    if valid_final:
                        return {
                            "U": new_U.tolist(),
                            "V": new_V.tolist(),
                            "W": new_W.tolist()
                        }
        except Exception:
            pass
    
    # If no reduced rank found, return rank-49
    return {
        "U": u_mat.tolist(),
        "V": v_mat.tolist(),
        "W": w_mat.tolist()
    }
