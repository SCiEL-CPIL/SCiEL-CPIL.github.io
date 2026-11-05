import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Target tensor T flattened
    # T[i, j, k] is 1 if we are computing C[i, k] from A[i, j] and B[j, k]
    # Flattened indices:
    # a = i * m + j
    # b = j * p + k
    # c = k * n + i
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        R = U.shape[1]
        try:
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        except Exception:
            return False, 1e10
           
            
        diff = np.max(np.abs(T_calc - T))
        return diff < 1e-9, diff

    def rationalize(M, tol=1e-6):
        # Round to nearest half-integer
        M_half = np.round(M * 2) / 2.0
        return M_half

    # Fallback: Schoolbook
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    start_time = time.time()
    deadline = start_time + 100 # 100s limit
    
    # Pre-compute reshaped T for efficient updates
    # For U update: T_flat_a = T.reshape(nm, -1) -> T[a, :, :] flattened
    T_flat_a = T.reshape(n * m, -1)
    
    # For V update: T_flat_b = T.transpose(1, 0, 2).reshape(mp, -1) -> T[:, b, :] flattened
    T_flat_b = T.transpose(1, 0, 2).reshape(m * p, -1)
    
    # For W update: T_flat_c = T.transpose(2, 0, 1).reshape(pn, -1) -> T[:, :, c] flattened
    T_flat_c = T.transpose(2, 0, 1).reshape(p * n, -1)

    # Try a few ranks. Given the difficulty, we try a few low ranks quickly.
    # If they fail, we return schoolbook.
    for R in [50, 45, 40, 35, 30]:
        if time.time() > deadline - 10: # Leave 10s for final check/return
            break
            
        np.random.seed(R)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # ALS iterations
        for it in range(20): # Reduced iterations to save time
            if time.time() > deadline - 5:
                break
                
            # Update U
            # M_mat columns r: (V[:, r] * W[:, r]).flatten() -> size mp*pn
            M_mat_U = np.empty((m * p * p * n, R))
            for r in range(R):
                M_mat_U[:, r] = (V[:, r] * W[:, r]).flatten()
            
            MtM_U = M_mat_U.T @ M_mat_U + 1e-6 * np.eye(R)
            MtT_U = M_mat_U.T @ T_flat_a.T # (R, nm)
            try:
                U_new = np.linalg.solve(MtM_U, MtT_U).T
            except np.linalg.LinAlgError:
                break

            # Update V
            # M_mat_V columns r: (U_new[:, r] * W[:, r]).flatten() -> size nm*pn
            M_mat_V = np.empty((n * m * p * n, R))
            for r in range(R):
                M_mat_V[:, r] = (U_new[:, r] * W[:, r]).flatten()
            
            MtM_V = M_mat_V.T @ M_mat_V + 1e-6 * np.eye(R)
            MtT_V = M_mat_V.T @ T_flat_b.T # (R, mp)
            try:
                V_new = np.linalg.solve(MtM_V, MtT_V).T
            except np.linalg.LinAlgError:
                break

            # Update W
            # M_mat_W columns r: (U_new[:, r] * V_new[:, r]).flatten() -> size nm*mp
            M_mat_W = np.empty((n * m * m * p, R))
            for r in range(R):
                M_mat_W[:, r] = (U_new[:, r] * V_new[:, r]).flatten()
            
            MtM_W = M_mat_W.T @ M_mat_W + 1e-6 * np.eye(R)
            MtT_W = M_mat_W.T @ T_flat_c.T # (R, pn)
            try:
                W_new = np.linalg.solve(MtM_W, MtT_W).T
            except np.linalg.LinAlgError:
                break
                
            U, V, W = U_new, V_new, W_new
            
        # Check error
        is_valid, err = verify(U, V, W)
        if is_valid:
            U_r = rationalize(U)
            V_r = rationalize(V)
            W_r = rationalize(W)
            is_valid_r, err_r = verify(U_r, V_r, W_r)
            if is_valid_r:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        else:
            if err < 1e-4:
                U_r = rationalize(U)
                V_r = rationalize(V)
                W_r = rationalize(W)
                is_valid_r, err_r = verify(U_r, V_r, W_r)
                if is_valid_r:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
