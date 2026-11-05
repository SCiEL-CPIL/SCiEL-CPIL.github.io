import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Build rank-28 baseline using Strassen + schoolbook
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28))
    V28 = np.zeros((m * p, rank_28))
    W28 = np.zeros((p * n, rank_28))

    col = 0
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A[i1,j1]+A[i2,j2])(B[j1,k1]+B[j2,k2])
        U28[i1*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        # M2 = (A[i2,j1]+A[i2,j2]) B[j1,k1]
        U28[i2*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1
        W28[k1*n+i2, col] = 1; W28[k2*n+i2, col] = -1
        col += 1
        # M3 = A[i1,j1] (B[j1,k2]-B[j2,k2])
        U28[i1*m+j1, col] = 1
        V28[j1*p+k2, col] = 1; V28[j2*p+k2, col] = -1
        W28[k2*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        # M4 = A[i2,j2] (B[j2,k1]-B[j1,k1])
        U28[i2*m+j2, col] = 1
        V28[j2*p+k1, col] = 1; V28[j1*p+k1, col] = -1
        W28[k1*n+i1, col] = 1; W28[k1*n+i2, col] = 1
        col += 1
        # M5 = (A[i1,j1]+A[i1,j2]) B[j2,k2]
        U28[i1*m+j1, col] = 1; U28[i1*m+j2, col] = 1
        V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = -1; W28[k2*n+i1, col] = 1
        col += 1
        # M6 = (A[i2,j1]-A[i1,j1])(B[j1,k1]+B[j1,k2])
        U28[i2*m+j1, col] = 1; U28[i1*m+j1, col] = -1
        V28[j1*p+k1, col] = 1; V28[j1*p+k2, col] = 1
        W28[k2*n+i2, col] = 1
        col += 1
        # M7 = (A[i1,j2]-A[i2,j2])(B[j2,k1]+B[j2,k2])
        U28[i1*m+j2, col] = 1; U28[i2*m+j2, col] = -1
        V28[j2*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U28[i*m+j, col] = 1
                    V28[j*p+k, col] = 1
                    W28[k*n+i, col] = 1
                    col += 1

    # Build the rank-28 decomposition
    add_strassen(0, 1, 0, 1, 0, 1)    # 7 factors
    add_strassen(0, 1, 0, 1, 2, 3)    # 7 factors
    add_schoolbook([0, 1], [0, 1], [4])    # 4 factors
    add_schoolbook([0, 1], [2], [0, 1])    # 4 factors
    add_schoolbook([0, 1], [2], [2, 3])    # 4 factors
    add_schoolbook([0, 1], [2], [4])       # 2 factors
    # Total: 7+7+4+4+4+2 = 28

    def verify(U, V, W, rank):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        res = np.zeros((n, m, p))
        for c in range(rank):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            # W is indexed [k*n+i], so Wc.T gives [n, p] = [i, k]
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[np.newaxis, :, :]
        return np.allclose(res, 1.0, atol=1e-9)

    assert verify(U28, V28, W28, rank_28), "Rank-28 baseline failed verification"

    # Attempt to reduce rank by finding linear dependencies among factors
    # Strategy: try removing one factor and see if the remaining factors can be adjusted
    # to still decompose the tensor (this is hard, so we use a simpler approach)
    
    # Simpler approach: check if any two factors are "mergeable" via linear combination
    # If factor c2 = a*factor_c1 + b*factor_c0 for some existing factor c0,
    # we might be able to eliminate it.
    
    # Most practical: try all pairs and check collinearity in all three spaces simultaneously
    best_rank = rank_28
    best_U, best_V, best_W = U28.copy(), V28.copy(), W28.copy()

    # Try eliminating one factor at a time by checking if it's redundant
    for target in range(rank_28):
        if time.time() > deadline - 40:
            break
        
        # Try to see if factor 'target' can be expressed as combination of others
        # Build matrix of other factors
        others = [c for c in range(rank_28) if c != target]
        
        # Check U space
        U_others = U28[:, others]  # (6, 27)
        U_target = U28[:, target]  # (6,)
        
        # Check if U_target is in span of U_others
        try:
            # Use least squares
            coeffs_u, residuals_u, rank_u, _ = np.linalg.lstsq(U_others, U_target, rcond=1e-10)
            if residuals_u[0] > 1e-8:
                continue
        except:
            continue
        
        # Check V space
        V_others = V28[:, others]  # (15, 27)
        V_target = V28[:, target]  # (15,)
        
        try:
            coeffs_v, residuals_v, rank_v, _ = np.linalg.lstsq(V_others, V_target, rcond=1e-10)
            if residuals_v[0] > 1e-8:
                continue
        except:
            continue
        
        # Check W space
        W_others = W28[:, others]  # (10, 27)
        W_target = W28[:, target]  # (10,)
        
        try:
            coeffs_w, residuals_w, rank_w, _ = np.linalg.lstsq(W_others, W_target, rcond=1e-10)
            if residuals_w[0] > 1e-8:
                continue
        except:
            continue
        
        # If target is in span of others in all three spaces, it might be redundant
        # But we need the SAME coefficients to work... which is unlikely.
        # Skip this approach as it's too restrictive.
        pass

    # Alternative: try direct pair merging (as in current program but more thorough)
    for c1 in range(rank_28 - 1):
        if time.time() > deadline - 40:
            break
        for c2 in range(c1 + 1, rank_28):
            if time.time() > deadline - 40:
                break
            
            u1, u2 = U28[:, c1], U28[:, c2]
            v1, v2 = V28[:, c1], V28[:, c2]
            w1, w2 = W28[:, c1], W28[:, c2]
            
            if np.allclose(u1, 0) or np.allclose(u2, 0):
                continue
            if np.allclose(v1, 0) or np.allclose(v2, 0):
                continue
            if np.allclose(w1, 0) or np.allclose(w2, 0):
                continue
            
            # Check if u2 = alpha_u * u1
            try:
                dot_u1 = np.dot(u1, u1)
                if abs(dot_u1) < 1e-12:
                    continue
                alpha_u = np.dot(u2, u1) / dot_u1
                if not np.allclose(u2, alpha_u * u1, atol=1e-9):
                    continue
            except:
                continue
            
            try:
                dot_v1 = np.dot(v1, v1)
                if abs(dot_v1) < 1e-12:
                    continue
                alpha_v = np.dot(v2, v1) / dot_v1
                if not np.allclose(v2, alpha_v * v1, atol=1e-9):
                    continue
            except:
                continue
            
            try:
                dot_w1 = np.dot(w1, w1)
                if abs(dot_w1) < 1e-12:
                    continue
                alpha_w = np.dot(w2, w1) / dot_w1
                if not np.allclose(w2, alpha_w * w1, atol=1e-9):
                    continue
            except:
                continue
            
            factor = 1 + alpha_u * alpha_v * alpha_w
            if abs(factor) < 1e-9:
                # Perfect cancellation: factor c1 and c2 cancel each other
                # Remove c2, keep c1 unchanged
                new_rank = rank_28 - 1
                U_new = np.zeros((n * m, new_rank))
                V_new = np.zeros((m * p, new_rank))
                W_new = np.zeros((p * n, new_rank))
                
                new_col = 0
                for c in range(rank_28):
                    if c == c2:
                        continue
                    U_new[:, new_col] = U28[:, c]
                    V_new[:, new_col] = V28[:, c]
                    W_new[:, new_col] = W28[:, c]
                    new_col += 1
                
                if verify(U_new, V_new, W_new, new_rank):
                    best_rank = new_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    break
            else:
                # Merged factor
                merged_U = factor * u1
                # Check if merged_U has nice values (integer or half-integer)
                if not np.allclose(merged_U, np.round(merged_U * 2) / 2, atol=1e-9):
                    continue
                
                new_rank = rank_28 - 1
                U_new = np.zeros((n * m, new_rank))
                V_new = np.zeros((m * p, new_rank))
                W_new = np.zeros((p * n, new_rank))
                
                new_col = 0
                for c in range(rank_28):
                    if c == c2:
                        continue
                    if c == c1:
                        U_new[:, new_col] = merged_U
                        V_new[:, new_col] = v1
                        W_new[:, new_col] = w1
                    else:
                        U_new[:, new_col] = U28[:, c]
                        V_new[:, new_col] = V28[:, c]
                        W_new[:, new_col] = W28[:, c]
                    new_col += 1
                
                if verify(U_new, V_new, W_new, new_rank):
                    best_rank = new_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    break
        if best_rank < rank_28:
            break

    if best_rank < rank_28:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
