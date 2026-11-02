import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15
    
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify(U, V, W, tol=1e-9):
        """Check if T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err += (s - 1.0) ** 2
        return err
    
    def als_step(U, V, W, target_rank):
        """One ALS iteration: update W, then U, then V"""
        # Update W: for each (k,i), W[k*n+i,:] = A_{ki} / ||A_{ki}||^2
        # where A_{ki} = sum_j U[i*m+j,:] * V[j*p+k,:]
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros(target_rank)
                for j in range(m):
                    A += U[i * m + j, :] * V[j * p + k, :]
                norm_sq = np.dot(A, A)
                if norm_sq > 1e-12:
                    W_new[k * n + i, :] = A / norm_sq
                else:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        # Update U: for each (i,j), solve B * U[i*m+j,:]^T = 1
        # where B[k,r] = V[j*p+k,r] * W_new[k*n+i,r]
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                B = np.zeros((p, target_rank))
                for k in range(p):
                    for r in range(target_rank):
                        B[k, r] = V[j * p + k, r] * W_new[k * n + i, r]
                # Solve B * x = ones(p) for x (length target_rank)
                # Minimum norm solution: x = B^T (B B^T)^{-1} ones
                try:
                    BBT = B @ B.T
                    BBT += 1e-10 * np.eye(p)
                    x = np.linalg.solve(BBT, np.ones(p))
                    U_new[i * m + j, :] = B.T @ x
                except np.linalg.LinAlgError:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Update V: for each (j,k), solve C * V[j*p+k,:]^T = 1
        # where C[i,r] = U_new[i*m+j,r] * W_new[k*n+i,r]
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                C = np.zeros((n, target_rank))
                for i in range(n):
                    for r in range(target_rank):
                        C[i, r] = U_new[i * m + j, r] * W_new[k * n + i, r]
                # Solve C * x = ones(n) for x (length target_rank)
                try:
                    CCT = C @ C.T
                    CCT += 1e-10 * np.eye(n)
                    x = np.linalg.solve(CCT, np.ones(n))
                    V_new[j * p + k, :] = C.T @ x
                except np.linalg.LinAlgError:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        return U_new, V_new, W_new
    
    def try_rank_als(r, max_iters=200):
        """Try to find a rank-r decomposition using ALS"""
        start_time = time.time()
        
        np.random.seed(42 + r)
        
        # Initialize with random values
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        # Normalize columns
        for c in range(r):
            norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
            if norm > 1e-10:
                U[:, c] /= norm
                V[:, c] /= norm
                W[:, c] /= norm
        
        current_err = compute_error(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            U, V, W = als_step(U, V, W, r)
            
            new_err = compute_error(U, V, W)
            
            if new_err < 1e-8:
                if verify(U, V, W, tol=1e-9):
                    return U, V, W, r
                if verify(U, V, W, tol=1e-10):
                    return U, V, W, r
            
            if it % 50 == 0:
                pass  # Could add restart logic here
        
        return None
    
    # Try ranks from 40 down to 30
    for r in range(40, 29, -1):
        if time.time() > deadline - 20:
            break
        result = try_rank_als(r)
        if result is not None:
            U, V, W, rk = result
            best_rank = rk
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
