import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    n, m, p = 3, 3, 6
    dim_U = n * m      # 9
    dim_V = m * p      # 18
    dim_W = p * n      # 18

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
        """Optimized ALS with precomputed contractions."""
        rank = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 20:
                break

            old_res = compute_residual(U, V, W)

            # === Update U ===
            R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
            for r in range(rank):
                A = V[:, r]
                B = W[:, r]
                A_sq = np.dot(A, A)
                B_sq = np.dot(B, B)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                R_r = np.einsum('i,j,k->ijk', U[:, r], A, B)
                R_other = R_full - R_r
                rhs = np.einsum('j,k,ijk->i', A, B, T_target) - np.einsum('j,k,ijk->i', A, B, R_other)
                U[:, r] = rhs / denom

            # === Update V ===
            R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
            for r in range(rank):
                A = U[:, r]
                B = W[:, r]
                A_sq = np.dot(A, A)
                B_sq = np.dot(B, B)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                R_r = np.einsum('i,j,k->ijk', A, V[:, r], B)
                R_other = R_full - R_r
                rhs = np.einsum('i,k,ijk->j', A, B, T_target) - np.einsum('i,k,ijk->j', A, B, R_other)
                V[:, r] = rhs / denom

            # === Update W ===
            R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
            for r in range(rank):
                A = U[:, r]
                B = V[:, r]
                A_sq = np.dot(A, A)
                B_sq = np.dot(B, B)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                R_r = np.einsum('i,j,k->ijk', A, B, W[:, r])
                R_other = R_full - R_r
                rhs = np.einsum('i,j,ijk->k', A, B, T_target) - np.einsum('i,j,ijk->k', A, B, R_other)
                W[:, r] = rhs / denom

            new_res = compute_residual(U, V, W)
            if abs(old_res - new_res) < tol:
                break

        return U, V, W

    def polish_pass(U, V, W):
        """High-precision polish with more iterations and tighter tolerance."""
        U, V, W = als_optimize(U, V, W, max_iter=300, tol=1e-14)
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            R = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = R - T_target
            res = np.sum(diff ** 2)
            if res < 1e-20:
                break
            for r in range(U.shape[1]):
                A = V[:, r]
                B = W[:, r]
                grad_U = np.einsum('j,k,ijk->i', A, B, diff)
                A_sq = np.dot(A, A)
                B_sq = np.dot(B, B)
                denom = A_sq * B_sq
                if denom > 1e-15:
                    U[:, r] -= grad_U / denom * 0.5
                grad_V = np.einsum('i,k,ijk->j', U[:, r], B, diff)
                if denom > 1e-15:
                    V[:, r] -= grad_V / denom * 0.5
                grad_W = np.einsum('i,j,ijk->k', U[:, r], A, diff)
                if denom > 1e-15:
                    W[:, r] -= grad_W / denom * 0.5
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

    def structured_init_p6(rank, seed):
        """Initialize exploiting p=6=2*3 structure.
        
        Decompose k = 2*l + s where l in {0,1,2}, s in {0,1}
        This gives a block structure that may allow lower rank.
        """
        np.random.seed(seed)
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        
        # Try to exploit the 2x3 block structure in k dimension
        # For each (i,j) pair, we need to cover k=0..5
        # Group k into pairs: (0,1), (2,3), (4,5)
        
        for r in range(rank):
            # Assign structure based on rank
            if r < 18:
                # First 18: one per (i, j, k_pair)
                idx = r
                i = idx // 6
                rem = idx % 6
                j = rem // 2
                k_pair = rem % 2  # 0 or 1
                
                # U: (i,j)
                U[i*m+j, r] = 1.0
                
                # V: (j, k) for k in {2*k_pair, 2*k_pair+1}
                V[j*p + 2*k_pair, r] = 1.0
                V[j*p + 2*k_pair + 1, r] = 1.0
                
                # W: (k, i) for k in {2*k_pair, 2*k_pair+1}
                W[(2*k_pair)*n + i, r] = 1.0
                W[(2*k_pair+1)*n + i, r] = 1.0
            elif r < 36:
                # Next 18: offset structure
                idx = r - 18
                i = idx // 6
                rem = idx % 6
                j = rem // 2
                k_pair = rem % 2
                
                U[i*m+j, r] = 1.0
                V[j*p + 2*k_pair, r] = 1.0
                V[j*p + 2*k_pair + 1, r] = -1.0
                W[(2*k_pair)*n + i, r] = 1.0
                W[(2*k_pair+1)*n + i, r] = -1.0
            else:
                # Random fill for higher ranks
                for idx in range(dim_U):
                    U[idx, r] = np.random.choice([-1.0, 0.0, 1.0])
                for idx in range(dim_V):
                    V[idx, r] = np.random.choice([-1.0, 0.0, 1.0])
                for idx in range(dim_W):
                    W[idx, r] = np.random.choice([-1.0, 0.0, 1.0])
        
        return U, V, W

    def structured_init_block(rank, seed):
        """Initialize with block structure exploiting m=3, p=6=2*3."""
        np.random.seed(seed)
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        
        for r in range(rank):
            # Use modular structure
            for idx in range(dim_U):
                i, j = divmod(idx, m)
                if (i + j + r) % 3 == 0:
                    U[idx, r] = np.random.choice([-1.0, 1.0])
                else:
                    U[idx, r] = np.random.choice([-1.0, 0.0, 1.0])
            
            for idx in range(dim_V):
                j, k = divmod(idx, p)
                if (j + k + r) % 3 == 0:
                    V[idx, r] = np.random.choice([-1.0, 1.0])
                else:
                    V[idx, r] = np.random.choice([-1.0, 0.0, 1.0])
            
            for idx in range(dim_W):
                k, i = divmod(idx, n)
                if (k + i + r) % 3 == 0:
                    W[idx, r] = np.random.choice([-1.0, 1.0])
                else:
                    W[idx, r] = np.random.choice([-1.0, 0.0, 1.0])
        
        return U, V, W

    def try_rank(rank, num_attempts=5):
        """Try to find exact decomposition for given rank."""
        best_res = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 25:
                break
            
            seed = rank * 1000 + attempt * 137 + 42
            
            if attempt == 0:
                U, V, W = structured_init_p6(rank, seed)
            elif attempt == 1:
                U, V, W = structured_init_block(rank, seed)
            elif attempt == 2:
                # Sparse random
                U = np.zeros((dim_U, rank))
                V = np.zeros((dim_V, rank))
                W = np.zeros((dim_W, rank))
                for r in range(rank):
                    mask_U = np.random.random(dim_U) < 0.3
                    mask_V = np.random.random(dim_V) < 0.2
                    mask_W = np.random.random(dim_W) < 0.2
                    U[mask_U, r] = np.random.choice([-1, 0, 1], size=mask_U.sum())
                    V[mask_V, r] = np.random.choice([-1, 0, 1], size=mask_V.sum())
                    W[mask_W, r] = np.random.choice([-1, 0, 1], size=mask_W.sum())
            elif attempt == 3:
                # Half-integer random
                U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(dim_U, rank))
                V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(dim_V, rank))
                W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(dim_W, rank))
            else:
                # Standard random
                U = np.random.randn(dim_U, rank) * 0.5
                V = np.random.randn(dim_V, rank) * 0.5
                W = np.random.randn(dim_W, rank) * 0.5
            
            U_opt, V_opt, W_opt = als_optimize(U, V, W, max_iter=100, tol=1e-12)
            res = compute_residual(U_opt, V_opt, W_opt)
            
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
            if res < 1e-8:
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return True, U_s, V_s, W_s
        
        # Try polish on best found
        if best_U is not None and best_res < 1e-4:
            U_pol, V_pol, W_pol = polish_pass(best_U, best_V, best_W)
            res_pol = compute_residual(U_pol, V_pol, W_pol)
            if res_pol < 1e-8:
                success, U_s, V_s, W_s = snap_and_verify(U_pol, V_pol, W_pol)
                if success:
                    return True, U_s, V_s, W_s
        
        return False, None, None, None

    # Try ranks from 27 upward (half of schoolbook)
    for rank in range(27, 55):
        if time.time() > deadline - 30:
            break
        
        num_attempts = 8 if rank <= 35 else 6 if rank <= 45 else 4
        
        success, U_s, V_s, W_s = try_rank(rank, num_attempts)
        if success:
            return {"U": U_s, "V": V_s, "W": W_s}
    
    # Fallback to schoolbook (rank 54)
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
