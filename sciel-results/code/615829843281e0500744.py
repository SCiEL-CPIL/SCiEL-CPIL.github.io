import numpy as np
import time
import random

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8

    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # Strassen 7 terms for 2x2x2
    terms = [
        ([(idx_u(0,0), 1), (idx_u(1,1), 1)],
         [(idx_v(0,0), 1), (idx_v(1,1), 1)],
         [(idx_w(0,0), 1), (idx_w(1,1), 1)]),
        ([(idx_u(1,0), 1), (idx_u(1,1), 1)],
         [(idx_v(0,0), 1)],
         [(idx_w(0,1), 1), (idx_w(1,1), -1)]),
        ([(idx_u(0,0), 1)],
         [(idx_v(0,1), 1), (idx_v(1,1), -1)],
         [(idx_w(1,0), 1), (idx_w(1,1), 1)]),
        ([(idx_u(1,1), 1)],
         [(idx_v(1,0), 1), (idx_v(0,0), -1)],
         [(idx_w(0,0), 1), (idx_w(0,1), 1)]),
        ([(idx_u(0,0), 1), (idx_u(0,1), 1)],
         [(idx_v(1,1), 1)],
         [(idx_w(1,0), 1), (idx_w(0,0), -1)]),
        ([(idx_u(1,0), 1), (idx_u(0,0), -1)],
         [(idx_v(0,0), 1), (idx_v(0,1), 1)],
         [(idx_w(1,1), 1)]),
        ([(idx_u(0,1), 1), (idx_u(1,1), -1)],
         [(idx_v(1,0), 1), (idx_v(1,1), 1)],
         [(idx_w(0,0), 1)]),
    ]

    def build_rank14():
        rank = 14
        U = np.zeros((u_dim, rank))
        V = np.zeros((v_dim, rank))
        W = np.zeros((w_dim, rank))
        
        for r in range(7):
            u_vec, v_vec, w_vec = terms[r]
            
            # Block 1: columns 0, 1 of B
            col = r
            for idx, val in u_vec:
                U[idx, col] += val
            for idx, val in v_vec:
                V[idx, col] += val
            for idx, val in w_vec:
                W[idx, col] += val
                
            # Block 2: columns 2, 3 of B
            col = r + 7
            for idx, val in u_vec:
                U[idx, col] += val
            for idx, val in v_vec:
                V[idx + 2, col] += val
            for idx, val in w_vec:
                W[idx + 4, col] += val
        
        return U, V, W

    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify tensor identity: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all (i,j,k)"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(U_mat.shape[1]):
                        total += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Build and verify rank-14
    U14, V14, W14 = build_rank14()
    
    if not verify(U14, V14, W14):
        # This should never happen, but fallback
        return {"U": U14.tolist(), "V": V14.tolist(), "W": W14.tolist()}

    best_rank = 14
    best_U, best_V, best_W = U14.copy(), V14.copy(), W14.copy()

    # Attempt rank-13 search
    # Strategy: Try to merge two of the 14 factors by finding a linear combination
    # that preserves the tensor identity.
    # 
    # The tensor identity is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # 
    # If we can find coefficients alpha, beta such that replacing factors r1, r2 with
    # a single factor works, we get rank 13.
    #
    # More practically: try random rank-13 initializations and use gradient descent
    # on the objective F = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2

    def objective(U_mat, V_mat, W_mat):
        """Compute sum of squared errors over all tensor entries."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U_mat.shape[1]):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    err += (s - 1.0) ** 2
        return err

    def try_rank13_search(seed, max_iter=2000, lr=0.01):
        """Try to find a rank-13 decomposition via gradient descent."""
        rng = np.random.RandomState(seed)
        rank = 13
        
        # Initialize near the rank-14 solution: randomly merge two factors
        # by averaging and keeping 13 columns
        U_init = U14.copy()
        V_init = V14.copy()
        W_init = W14.copy()
        
        # Try different merging strategies
        for merge_pair in range(14):
            if time.time() > deadline:
                return None
            
            # Merge factor merge_pair with factor (merge_pair+1) % 14
            other = (merge_pair + 1) % 14
            if other == merge_pair:
                continue
            
            # Create rank-13 by removing one factor and adjusting
            cols = [c for c in range(14) if c != other]
            U_r13 = U14[:, cols].copy()
            V_r13 = V14[:, cols].copy()
            W_r13 = W14[:, cols].copy()
            
            # Add small random perturbation
            U_r13 += rng.randn(u_dim, rank) * 0.01
            V_r13 += rng.randn(v_dim, rank) * 0.01
            W_r13 += rng.randn(w_dim, rank) * 0.01
            
            # Gradient descent
            U_cur = U_r13.copy()
            V_cur = V_r13.copy()
            W_cur = W_r13.copy()
            
            best_obj = objective(U_cur, V_cur, W_cur)
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute gradients numerically (central differences)
                # This is expensive but we have limited iterations
                eps = 1e-5
                grad_U = np.zeros_like(U_cur)
                grad_V = np.zeros_like(V_cur)
                grad_W = np.zeros_like(W_cur)
                
                # Sample a subset of parameters for gradient estimation
                # to keep it fast
                num_samples = min(20, u_dim * rank)
                indices = rng.choice(u_dim * rank, size=num_samples, replace=False)
                
                for idx in indices:
                    i, r = divmod(idx, rank)
                    
                    # Gradient wrt U[i, r]
                    U_cur[i, r] += eps
                    f_plus = objective(U_cur, V_cur, W_cur)
                    U_cur[i, r] -= 2 * eps
                    f_minus = objective(U_cur, V_cur, W_cur)
                    U_cur[i, r] += eps
                    grad_U[i, r] = (f_plus - f_minus) / (2 * eps)
                    
                    # Gradient wrt V[i, r]
                    V_cur[i, r] += eps
                    f_plus = objective(V_cur, U_cur, W_cur)
                    V_cur[i, r] -= 2 * eps
                    f_minus = objective(V_cur, U_cur, W_cur)
                    V_cur[i, r] += eps
                    grad_V[i, r] = (f_plus - f_minus) / (2 * eps)
                    
                    # Gradient wrt W[i, r]
                    W_cur[i, r] += eps
                    f_plus = objective(W_cur, U_cur, V_cur)
                    W_cur[i, r] -= 2 * eps
                    f_minus = objective(W_cur, U_cur, V_cur)
                    W_cur[i, r] += eps
                    grad_W[i, r] = (f_plus - f_minus) / (2 * eps)
                
                # Update
                U_cur -= lr * grad_U
                V_cur -= lr * grad_V
                W_cur -= lr * grad_W
                
                obj = objective(U_cur, V_cur, W_cur)
                if obj < best_obj:
                    best_obj = obj
                    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                
                if best_obj < 1e-8:
                    break
                
                # Adaptive learning rate
                if obj > best_obj * 1.5:
                    lr *= 0.5
            
            if best_obj < 1e-8:
                return best_U, best_V, best_W
        
        return None

    # Try several seeds for rank-13 search
    for seed in range(20):
        if time.time() > deadline:
            break
        
        result = try_rank13_search(seed, max_iter=500, lr=0.005)
        if result is not None:
            U13, V13, W13 = result
            if verify(U13, V13, W13, tol=1e-8):
                best_rank = 13
                best_U, best_V, best_W = U13, V13, W13
                break
    
    # If rank-13 not found, try a different approach: structured search
    # based on the block structure with shared factors
    
    if best_rank == 14 and time.time() < deadline:
        # Try to find a rank-13 by exploiting the fact that the two blocks
        # might share some factors if we allow cross-block coupling
        # 
        # Alternative: use the known lower bound analysis
        # The tensor rank of 2x2x4 matmul is known to be 14 (two independent 2x2x2 blocks)
        # So rank-13 likely doesn't exist. But let's try one more approach.
        
        # Try random rank-13 with different initialization
        rng = np.random.RandomState(42)
        rank = 13
        U_try = rng.randn(u_dim, rank) * 0.3
        V_try = rng.randn(v_dim, rank) * 0.3
        W_try = rng.randn(w_dim, rank) * 0.3
        
        # Simple gradient descent with a few iterations
        lr = 0.01
        for it in range(100):
            if time.time() > deadline:
                break
            
            eps = 1e-4
            # Compute a few gradient components
            for _ in range(5):
                idx = rng.randint(0, u_dim * rank - 1)
                i, r = divmod(idx, rank)
                
                U_try[i, r] += eps
                f_plus = objective(U_try, V_try, W_try)
                U_try[i, r] -= 2 * eps
                f_minus = objective(U_try, V_try, W_try)
                U_try[i, r] += eps
                grad = (f_plus - f_minus) / (2 * eps)
                U_try[i, r] -= lr * grad
            
            # Similarly for V and W (simplified)
            for _ in range(5):
                idx = rng.randint(0, v_dim * rank - 1)
                j, r = divmod(idx, rank)
                
                V_try[j, r] += eps
                f_plus = objective(U_try, V_try, W_try)
                V_try[j, r] -= 2 * eps
                f_minus = objective(U_try, V_try, W_try)
                V_try[j, r] += eps
                grad = (f_plus - f_minus) / (2 * eps)
                V_try[j, r] -= lr * grad
            
            for _ in range(5):
                idx = rng.randint(0, w_dim * rank - 1)
                k, r = divmod(idx, rank)
                
                W_try[k, r] += eps
                f_plus = objective(U_try, V_try, W_try)
                W_try[k, r] -= 2 * eps
                f_minus = objective(U_try, V_try, W_try)
                W_try[k, r] += eps
                grad = (f_plus - f_minus) / (2 * eps)
                W_try[k, r] -= lr * grad
        
        if verify(U_try, V_try, W_try, tol=1e-6):
            best_rank = 13
            best_U, best_V, best_W = U_try, V_try, W_try

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
