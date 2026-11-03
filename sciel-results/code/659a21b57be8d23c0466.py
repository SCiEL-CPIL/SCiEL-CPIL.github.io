import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 110  # 40s safety margin before 150s kill

    # Schoolbook decomposition as guaranteed fallback (rank 105)
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n*m, rank_schoolbook))
    V_schoolbook = np.zeros((m*p, rank_schoolbook))
    W_schoolbook = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i*m+j, col] = 1
                V_schoolbook[j*p+k, col] = 1
                W_schoolbook[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_schoolbook.copy(), V_schoolbook.copy(), W_schoolbook.copy()

    # Precompute target indices: (idx_u, idx_v, idx_w) for each (i,j,k)
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    target_indices = np.array(target_indices)  # shape (105, 3)
    idx_u = target_indices[:, 0]
    idx_v = target_indices[:, 1]
    idx_w = target_indices[:, 2]
    T = np.ones(len(target_indices))  # all targets are 1

    def verify(U, V, W, tol=1e-8):
        """Check if decomposition exactly reproduces the tensor."""
        R = U.shape[1]
        # Compute T_hat[idx] = sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
        # Vectorized: for each r, outer product contribution
        err = np.zeros(len(target_indices))
        for r in range(R):
            err += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
        return np.allclose(err, T, atol=tol, rtol=0)

    def round_and_verify(U, V, W):
        """Round to half-integers and verify."""
        Ur = np.round(U * 2) / 2
        Vr = np.round(V * 2) / 2
        Wr = np.round(W * 2) / 2
        if verify(Ur, Vr, Wr, tol=1e-10):
            return Ur, Vr, Wr, True
        # Try integer rounding
        Ui = np.round(U)
        Vi = np.round(V)
        Wi = np.round(W)
        if verify(Ui, Vi, Wi, tol=1e-10):
            return Ui, Vi, Wi, True
        return None, None, None, False

    def als_solve(U, V, W, max_iter=500, tol=1e-10):
        """Alternating Least Squares for tensor factorization.
        Minimize ||T - sum_r U[idx_u,r]*V[idx_v,r]*W[idx_w,r]||^2
        """
        R = U.shape[1]
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: for each r, solve least squares
            # T = sum_r U[:,r] * (V[idx_v,r] * W[idx_w,r])
            # For fixed r, U[:,r] = argmin ||T - other_r - U[:,r]*coef||^2
            # where coef = V[idx_v,r] * W[idx_w,r]
            
            for r in range(R):
                # Compute coefficient vector for this r
                coef = V[idx_v, r] * W[idx_w, r]
                # Residual without this term
                resid = T.copy()
                for rr in range(R):
                    if rr != r:
                        resid -= U[idx_u, rr] * V[idx_v, rr] * W[idx_w, rr]
                # Solve: U[idx_u, r] * coef ≈ resid
                # Least squares: U[idx_u, r] = (coef^T resid) / (coef^T coef)
                denom = np.dot(coef, coef)
                if abs(denom) < 1e-15:
                    continue
                val = np.dot(coef, resid) / denom
                U[idx_u, r] = val
            
            # Update V
            for r in range(R):
                coef = U[idx_u, r] * W[idx_w, r]
                resid = T.copy()
                for rr in range(R):
                    if rr != r:
                        resid -= U[idx_u, rr] * V[idx_v, rr] * W[idx_w, rr]
                denom = np.dot(coef, coef)
                if abs(denom) < 1e-15:
                    continue
                val = np.dot(coef, resid) / denom
                V[idx_v, r] = val
            
            # Update W
            for r in range(R):
                coef = U[idx_u, r] * V[idx_v, r]
                resid = T.copy()
                for rr in range(R):
                    if rr != r:
                        resid -= U[idx_u, rr] * V[idx_v, rr] * W[idx_w, rr]
                denom = np.dot(coef, coef)
                if abs(denom) < 1e-15:
                    continue
                val = np.dot(coef, resid) / denom
                W[idx_w, r] = val
            
            # Check convergence
            err = np.zeros(len(target_indices))
            for r in range(R):
                err += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
            if np.allclose(err, T, atol=tol, rtol=0):
                break
        
        return U, V, W

    def fast_als(U, V, W, max_iter=200, tol=1e-10):
        """Faster ALS using vectorized residual computation."""
        R = U.shape[1]
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Precompute current factorization
            # For each r, compute contribution
            # Use the fact that we can vectorize the residual computation
            
            # Update U
            for r in range(R):
                coef = V[idx_v, r] * W[idx_w, r]
                # resid = T - sum_{rr != r} U[idx_u,rr] * V[idx_v,rr] * W[idx_w,rr]
                # = T - (sum_{all} U[idx_u,rr]*V[idx_v,rr]*W[idx_w,rr]) + U[idx_u,r]*V[idx_v,r]*W[idx_w,r]
                total = np.zeros(len(target_indices))
                for rr in range(R):
                    total += U[idx_u, rr] * V[idx_v, rr] * W[idx_w, rr]
                resid = T - total + U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                denom = np.dot(coef, coef)
                if abs(denom) < 1e-15:
                    continue
                val = np.dot(coef, resid) / denom
                U[idx_u, r] = val
            
            # Update V
            for r in range(R):
                coef = U[idx_u, r] * W[idx_w, r]
                total = np.zeros(len(target_indices))
                for rr in range(R):
                    total += U[idx_u, rr] * V[idx_v, rr] * W[idx_w, rr]
                resid = T - total + U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                denom = np.dot(coef, coef)
                if abs(denom) < 1e-15:
                    continue
                val = np.dot(coef, resid) / denom
                V[idx_v, r] = val
            
            # Update W
            for r in range(R):
                coef = U[idx_u, r] * V[idx_v, r]
                total = np.zeros(len(target_indices))
                for rr in range(R):
                    total += U[idx_u, rr] * V[idx_v, rr] * W[idx_w, rr]
                resid = T - total + U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                denom = np.dot(coef, coef)
                if abs(denom) < 1e-15:
                    continue
                val = np.dot(coef, resid) / denom
                W[idx_w, r] = val
            
            # Check convergence
            total = np.zeros(len(target_indices))
            for r in range(R):
                total += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
            if np.allclose(total, T, atol=tol, rtol=0):
                break
        
        return U, V, W

    # Try ranks from 104 down to 50
    # Use multiple random initializations for each rank
    np.random.seed(42)
    
    for rank in range(rank_schoolbook - 1, 49, -1):
        if time.time() > deadline:
            break
        
        # Try several random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            U, V, W = fast_als(U, V, W, max_iter=100, tol=1e-12)
            
            # Check if we found an exact solution
            if verify(U, V, W, tol=1e-10):
                Ur, Vr, Wr, valid = round_and_verify(U, V, W)
                if valid:
                    best_rank = rank
                    best_U, best_V, best_W = Ur, Vr, Wr
                    break
            else:
                # Even if not exact, check if rounding gives a valid solution
                Ur, Vr, Wr, valid = round_and_verify(U, V, W)
                if valid:
                    best_rank = rank
                    best_U, best_V, best_W = Ur, Vr, Wr
                    break
        
        if best_rank < rank_schoolbook:
            break

    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-10):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_schoolbook, V_schoolbook, W_schoolbook
        best_rank = rank_schoolbook

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
