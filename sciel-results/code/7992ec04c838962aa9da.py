import numpy as np
import time
from itertools import product

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
        """Optimized ALS with precomputed contractions."""
        dim_U, rank = U.shape
        dim_V, _ = V.shape
        dim_W, _ = W.shape

        for iteration in range(max_iter):
            if time.time() > deadline - 15:
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
        # Additional refinement: try to drive residual even lower
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            R = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = R - T_target
            res = np.sum(diff ** 2)
            if res < 1e-20:
                break
            # Gradient-based step for each factor
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

    def structured_search(rank, values=(-1, 0, 1)):
        """Try structured search for small rank with integer/half-integer values."""
        # This is a heuristic search - try random structured patterns
        for attempt in range(20):
            if time.time() > deadline - 20:
                return None
            
            np.random.seed(42 + rank * 100 + attempt)
            
            # Try different patterns
            if attempt % 3 == 0:
                # Sparse random
                U = np.zeros((dim_U, rank))
                V = np.zeros((dim_V, rank))
                W = np.zeros((dim_W, rank))
                for r in range(rank):
                    # Random sparse support
                    mask_U = np.random.random(dim_U) < 0.3
                    mask_V = np.random.random(dim_V) < 0.2
                    mask_W = np.random.random(dim_W) < 0.2
                    U[mask_U, r] = np.random.choice(values, size=mask_U.sum())
                    V[mask_V, r] = np.random.choice(values, size=mask_V.sum())
                    W[mask_W, r] = np.random.choice(values, size=mask_W.sum())
            elif attempt % 3 == 1:
                # Block structure: exploit that p=6=2*3
                U = np.zeros((dim_U, rank))
                V = np.zeros((dim_V, rank))
                W = np.zeros((dim_W, rank))
                for r in range(rank):
                    # Assign based on index patterns
                    for idx in range(dim_U):
                        i, j = divmod(idx, m)
                        if (i + j + r) % 3 == 0:
                            U[idx, r] = np.random.choice(values)
                    for idx in range(dim_V):
                        j, k = divmod(idx, p)
                        if (j + k + r) % 3 == 0:
                            V[idx, r] = np.random.choice(values)
                    for idx in range(dim_W):
                        k, i = divmod(idx, n)
                        if (k + i + r) % 3 == 0:
                            W[idx, r] = np.random.choice(values)
            else:
                # Symmetric pattern
                U = np.zeros((dim_U, rank))
                V = np.zeros((dim_V, rank))
                W = np.zeros((dim_W, rank))
                for r in range(rank):
                    for idx in range(dim_U):
                        i, j = divmod(idx, m)
                        U[idx, r] = np.random.choice(values)
                    for idx in range(dim_V):
                        j, k = divmod(idx, p)
                        V[idx, r] = np.random.choice(values)
                    for idx in range(dim_W):
                        k, i = divmod(idx, n)
                        W[idx, r] = np.random.choice(values)
            
            # Polish and check
            U_opt, V_opt, W_opt = als_optimize(U, V, W, max_iter=100, tol=1e-12)
            res = compute_residual(U_opt, V_opt, W_opt)
            if res < 1e-8:
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return (U_s, V_s, W_s)
        
        return None

    # First, try structured search for ranks 2-10 (very small)
    for rank in range(2, 11):
        if time.time() > deadline - 30:
            break
        result = structured_search(rank)
        if result is not None:
            U_s, V_s, W_s = result
            return {"U": U_s, "V": V_s, "W": W_s}

    # Try ALS with various initializations for ranks 10-54
    best_result = None
    best_rank = 54

    for rank in range(10, 55):
        if time.time() > deadline - 25:
            break
        
        # More attempts for lower ranks
        num_attempts = 15 if rank <= 20 else 10 if rank <= 35 else 6
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(rank * 1000 + attempt * 37 + 13)
            
            if attempt == 0:
                # Schoolbook-like initialization with noise
                U = np.zeros((dim_U, rank))
                V = np.zeros((dim_V, rank))
                W = np.zeros((dim_W, rank))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < rank:
                                U[i*m+j, col] = 1.0
                                V[j*p+k, col] = 1.0
                                W[k*n+i, col] = 1.0
                            col += 1
                U += np.random.randn(dim_U, rank) * 0.05
                V += np.random.randn(dim_V, rank) * 0.05
                W += np.random.randn(dim_W, rank) * 0.05
            elif attempt == 1:
                # Scaled random
                U = np.random.randn(dim_U, rank) * 0.4
                V = np.random.randn(dim_V, rank) * 0.4
                W = np.random.randn(dim_W, rank) * 0.4
            elif attempt == 2:
                # Binary-like init
                U = np.random.choice([-1.0, 0.0, 1.0], size=(dim_U, rank))
                V = np.random.choice([-1.0, 0.0, 1.0], size=(dim_V, rank))
                W = np.random.choice([-1.0, 0.0, 1.0], size=(dim_W, rank))
            elif attempt == 3:
                # Half-integer init
                U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(dim_U, rank))
                V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(dim_V, rank))
                W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(dim_W, rank))
            else:
                # Standard random
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
            
            # Polish pass if close
            if res < 1e-4:
                U_opt2, V_opt2, W_opt2 = polish_pass(U_opt, V_opt, W_opt)
                res2 = compute_residual(U_opt2, V_opt2, W_opt2)
                if res2 < 1e-8:
                    success, U_s, V_s, W_s = snap_and_verify(U_opt2, V_opt2, W_opt2)
                    if success:
                        return {"U": U_s, "V": V_s, "W": W_s}

    # Fallback to schoolbook (rank 54)
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
