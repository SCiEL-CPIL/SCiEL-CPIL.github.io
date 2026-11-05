import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill

    n, m, p = 3, 3, 6
    dim_U = n * m  # 9
    dim_V = m * p  # 18
    dim_W = p * n  # 18

    # Build target tensor: T_target[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def compute_residual(U, V, W):
        """Compute sum_r U[:,r] V[:,r] W[:,r] - T_target squared norm."""
        R = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((R - T_target) ** 2)

    def snap_and_verify(U, V, W):
        """Snap to nearest half-integer and verify exactness."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        if np.allclose(R, T_target, atol=1e-9):
            return True, U_s.tolist(), V_s.tolist(), W_s.tolist()
        return False, None, None, None

    def als_optimize(U, V, W, max_iter=200, tol=1e-12):
        """Alternating Least Squares for tensor decomposition."""
        dim_U, rank = U.shape
        dim_V, _ = V.shape
        dim_W, _ = W.shape
        
        # Precompute target reshapes for ALS
        # For updating U: fix V, W. We need to solve for each column r of U.
        # R_target = T_target. R = sum_r U[:,r] V[:,r] W[:,r]
        # For a fixed r, the contribution is U[:,r] V[:,r] W[:,r] (outer product)
        # To update U[:,r], we want to minimize ||T_target - sum_s U[:,s] V[:,s] W[:,s]||^2
        # This is equivalent to minimizing over U[:,r]:
        # ||T_target - U[:,r] V[:,r] W[:,r] - sum_{s!=r} U[:,s] V[:,s] W[:,s]||^2
        # Let R_other = sum_{s!=r} U[:,s] V[:,s] W[:,s]
        # Then we minimize ||(T_target - R_other) - U[:,r] V[:,r] W[:,r]||^2
        # This is a rank-1 approximation problem.
        
        # More efficient: use the fact that for fixed V[:,r] and W[:,r],
        # the optimal U[:,r] can be found by solving a linear system.
        # Let A = V[:,r] (dim_V,), B = W[:,r] (dim_W,)
        # We want to minimize ||T_target - U[:,r] outer(A, B) - R_other||^2
        # This is equivalent to: for each (i,j,k), the residual is
        # T_target[i,j,k] - U[i,r]*A[j]*B[k] - R_other[i,j,k]
        # Taking derivative w.r.t. U[i,r]:
        # sum_{j,k} A[j]*B[k]*(T_target[i,j,k] - U[i,r]*A[j]*B[k] - R_other[i,j,k]) = 0
        # U[i,r] * sum_{j,k} A[j]^2 * B[k]^2 = sum_{j,k} A[j]*B[k]*(T_target[i,j,k] - R_other[i,j,k])
        
        # Precompute A_sq_dot_B_sq = sum_{j,k} A[j]^2 * B[k]^2 = (sum_j A[j]^2) * (sum_k B[k]^2)
        # And for each i: rhs[i] = sum_{j,k} A[j]*B[k]*(T_target[i,j,k] - R_other[i,j,k])
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            old_res = compute_residual(U, V, W)
            
            # Update U
            for r in range(rank):
                A = V[:, r]
                B = W[:, r]
                A_sq = np.sum(A ** 2)
                B_sq = np.sum(B ** 2)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                
                # Compute R_other = sum_{s!=r} U[:,s] V[:,s] W[:,s]
                # Efficient: compute full R and subtract contribution of r
                R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
                R_r = np.einsum('i,j,k->ijk', U[:, r], A, B)
                R_other = R_full - R_r
                
                # rhs[i] = sum_{j,k} A[j]*B[k]*(T_target[i,j,k] - R_other[i,j,k])
                # = sum_{j,k} A[j]*B[k]*T_target[i,j,k] - sum_{j,k} A[j]*B[k]*R_other[i,j,k]
                # = einsum('j,k,ijk->i', A, B, T_target) - einsum('j,k,ijk->i', A, B, R_other)
                rhs = np.einsum('j,k,ijk->i', A, B, T_target) - np.einsum('j,k,ijk->i', A, B, R_other)
                U[:, r] = rhs / denom
            
            # Update V
            for r in range(rank):
                A = U[:, r]
                B = W[:, r]
                A_sq = np.sum(A ** 2)
                B_sq = np.sum(B ** 2)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                
                R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
                R_r = np.einsum('i,j,k->ijk', A, V[:, r], B)
                R_other = R_full - R_r
                
                rhs = np.einsum('i,k,ijk->j', A, B, T_target) - np.einsum('i,k,ijk->j', A, B, R_other)
                V[:, r] = rhs / denom
            
            # Update W
            for r in range(rank):
                A = U[:, r]
                B = V[:, r]
                A_sq = np.sum(A ** 2)
                B_sq = np.sum(B ** 2)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                
                R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
                R_r = np.einsum('i,j,k->ijk', A, B, W[:, r])
                R_other = R_full - R_r
                
                rhs = np.einsum('i,j,ijk->k', A, B, T_target) - np.einsum('i,j,ijk->k', A, B, R_other)
                W[:, r] = rhs / denom
            
            new_res = compute_residual(U, V, W)
            if abs(old_res - new_res) < tol:
                break
        
        return U, V, W

    def get_schoolbook(rank):
        """Standard schoolbook decomposition with rank = n*m*p."""
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Try ranks from 46 to 54, using ALS optimization
    # Start from lower ranks and work up
    for rank in range(46, 55):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        num_attempts = 8 if rank < 54 else 3
        for attempt in range(num_attempts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(rank * 100 + attempt)
            
            # Initialize with small random values
            U = np.random.randn(dim_U, rank) * 0.5
            V = np.random.randn(dim_V, rank) * 0.5
            W = np.random.randn(dim_W, rank) * 0.5
            
            # Run ALS
            U_opt, V_opt, W_opt = als_optimize(U, V, W, max_iter=150, tol=1e-12)
            
            # Check if we got close to zero residual
            res = compute_residual(U_opt, V_opt, W_opt)
            if res < 1e-8:
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return {"U": U_s, "V": V_s, "W": W_s}
            
            # Try a second polish pass with tighter tolerance if close
            if res < 1e-4:
                U_opt2, V_opt2, W_opt2 = als_optimize(U_opt, V_opt, W_opt, max_iter=100, tol=1e-14)
                res2 = compute_residual(U_opt2, V_opt2, W_opt2)
                if res2 < 1e-8:
                    success, U_s, V_s, W_s = snap_and_verify(U_opt2, V_opt2, W_opt2)
                    if success:
                        return {"U": U_s, "V": V_s, "W": W_s}
        
        # If we're at rank 54, just return schoolbook
        if rank == 54:
            break

    # Fallback to schoolbook (rank 54)
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
