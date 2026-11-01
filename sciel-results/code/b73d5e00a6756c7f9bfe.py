import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    K = mp * pn
    
    # Schoolbook fallback
    rank_school = nm * mp * pn
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    start_time = time.time()
    time_limit = 140.0 # Leave 10s margin
    
    # Target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' and i=i', j=j', k=k'
    # Actually, T is the tensor for matrix multiplication.
    # T_{(i,j), (j',k), (k',i')} = 1 if i=i', j=j', k=k' else 0.
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_flat = T.reshape(nm, K)
    
    def als_step(U, V, W, R):
        # Update U
        # Z[r, idx] = V[b, r] * W[c, r]
        # V is (mp, R), W is (pn, R)
        # V[:, r] is (mp, 1), W[:, r] is (pn, 1)
        # Outer product is (mp, pn) -> flatten to (K,)
        # Z is (R, K)
        Z = np.einsum('br,cr->rbc', V, W).reshape(R, K)
        # U = T_flat @ Z.T @ inv(Z @ Z.T)
        # Z.T is (K, R)
        # Z @ Z.T is (R, R)
        try:
            ZZT = Z @ Z.T
            # Add small regularization to avoid singularity
            ZZT += 1e-6 * np.eye(R)
            U_new = T_flat @ Z.T @ np.linalg.inv(ZZT)
        except np.linalg.LinAlgError:
            U_new = U
        
        # Update V
        # Fix U, W.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For fixed b, V[b, :] is the variable.
        # T[:, b, :] = sum_r V[b, r] (U[:, r] * W[:, r])
        # Let Y[r, idx(ac)] = U[a, r] * W[c, r]
        # Y is (R, nm * pn)
        # T[:, b, :] reshaped is (nm * pn,)
        # T_b = sum_r V[b, r] Y[r, :]
        # This is a linear system for V[b, :]
        # V[b, :] = T_b @ Y.T @ inv(Y @ Y.T) ?
        # Y is (R, K2) where K2 = nm * pn
        # T_b is (K2,)
        # V[b, :] is (R,)
        # T_b = V[b, :] @ Y
        # So V[b, :] = T_b @ Y.T @ inv(Y @ Y.T)
        
        K2 = nm * pn
        Y = np.einsum('ar,cr->rac', U, W).reshape(R, K2)
        try:
            YYT = Y @ Y.T
            YYT += 1e-6 * np.eye(R)
            V_new = np.zeros((mp, R))
            for b in range(mp):
                T_b = T[:, b, :].reshape(K2)
                V_new[b, :] = T_b @ Y.T @ np.linalg.inv(YYT)
        except np.linalg.LinAlgError:
            V_new = V
            
        # Update W
        # Fix U, V.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For fixed c, W[c, :] is the variable.
        # T[:, :, c] = sum_r W[c, r] (U[:, r] * V[:, r])
        # Let X[r, idx(ab)] = U[a, r] * V[b, r]
        # X is (R, nm * mp)
        # T[:, :, c] reshaped is (nm * mp,)
        # W[c, :] = T_c @ X.T @ inv(X @ X.T)
        
        K3 = nm * mp
        X = np.einsum('ar,br->rab', U, V).reshape(R, K3)
        try:
            XXT = X @ X.T
            XXT += 1e-6 * np.eye(R)
            W_new = np.zeros((pn, R))
            for c in range(pn):
                T_c = T[:, :, c].reshape(K3)
                W_new[c, :] = T_c @ X.T @ np.linalg.inv(XXT)
        except np.linalg.LinAlgError:
            W_new = W
            
        return U_new, V_new, W_new

    def check_solution(U, V, W):
        # Round to nearest half-integer
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Reconstruct T
        T_recon = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        
        # Check exactness
        if np.allclose(T_recon, T, atol=1e-9):
            return True, U_int, V_int, W_int
        return False, None, None, None

    # Try ranks from 20 to 23
    for R in [20, 21, 22, 23]:
        if time.time() - start_time > time_limit:
            break
            
        max_restarts = 1000
        max_iter = 50
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            # Scale 1.0
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Run ALS
            for it in range(max_iter):
                U, V, W = als_step(U, V, W, R)
                
                # Check error periodically
                if it % 5 == 0:
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(T_recon - T)
                    if err < 1e-4:
                        break
                        
            # Check if solution is close to integer/half-integer
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if err < 1e-3:
                ok, U_sol, V_sol, W_sol = check_solution(U, V, W)
                if ok:
                    best_rank = R
                    best_U = U_sol.tolist()
                    best_V = V_sol.tolist()
                    best_W = W_sol.tolist()
                    # Found a solution, can break if R is minimal possible?
                    # We don't know if 19 is possible, so we continue to lower R?
                    # We are iterating R from 20 upwards.
                    # If we find 20, we can stop if we are confident 19 is not possible or if we want to save time.
                    # Given time limits, finding 20 is a good win.
                    # Let's break out of restart loop and try lower R?
                    # No, we are going 20, 21...
                    # If we find 20, we should break the outer loop too?
                    # Actually, if we find 20, it's better than 24.
                    # If we have time, we could try 19.
                    # Let's break to 19 if we find 20 quickly.
                    break
                    
        if best_rank < rank_school and best_rank == 20:
             # Try 19 if time permits
             if time.time() - start_time < time_limit - 20:
                 pass # Logic to try 19 would go here, but let's keep it simple first.
                 # Actually, let's just return 20 if found.
                 break
                 
    return {"U": best_U, "V": best_V, "W": best_W}
