import numpy as np

def construct():
    n, m, p = 3, 3, 6
    dim_U = n * m   # 9
    dim_V = m * p   # 18
    dim_W = p * n   # 18

    # Build target tensor: T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def snap_and_verify(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        if np.allclose(R, T_target, atol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None, None

    def als_optimize(U, V, W, max_iter=300, tol=1e-13):
        dim_U, rank = U.shape
        dim_V, _ = V.shape
        dim_W, _ = W.shape

        def compute_residual():
            R = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.sum((R - T_target) ** 2)

        for iteration in range(max_iter):
            old_res = compute_residual()

            # Update U
            for r in range(rank):
                A = V[:, r]
                B = W[:, r]
                A_sq = np.sum(A ** 2)
                B_sq = np.sum(B ** 2)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
                R_r = np.einsum('i,j,k->ijk', U[:, r], A, B)
                R_other = R_full - R_r
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

            new_res = compute_residual()
            if abs(old_res - new_res) < tol:
                break

        return U, V, W

    def get_schoolbook(rank):
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

    # Try ranks from 30 to 54, using ALS optimization
    # Start from lower ranks and work up
    for rank in range(30, 55):
        # Try multiple random initializations
        num_attempts = 12 if rank < 40 else 6
        for attempt in range(num_attempts):
            np.random.seed(rank * 100 + attempt)
            
            # Initialize with small random values
            U = np.random.randn(dim_U, rank) * 0.5
            V = np.random.randn(dim_V, rank) * 0.5
            W = np.random.randn(dim_W, rank) * 0.5
            
            # Run ALS
            U_opt, V_opt, W_opt = als_optimize(U, V, W, max_iter=200, tol=1e-13)
            
            # Check if we got close to zero residual
            R = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt)
            res = np.sum((R - T_target) ** 2)
            if res < 1e-8:
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            
            # Try a second polish pass with tighter tolerance if close
            if res < 1e-4:
                U_opt2, V_opt2, W_opt2 = als_optimize(U_opt, V_opt, W_opt, max_iter=150, tol=1e-14)
                R2 = np.einsum('ir,jr,kr->ijk', U_opt2, V_opt2, W_opt2)
                res2 = np.sum((R2 - T_target) ** 2)
                if res2 < 1e-8:
                    success, U_s, V_s, W_s = snap_and_verify(U_opt2, V_opt2, W_opt2)
                    if success:
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        # If we're at rank 54, just return schoolbook
        if rank == 54:
            break

    # Fallback to schoolbook (rank 54)
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
