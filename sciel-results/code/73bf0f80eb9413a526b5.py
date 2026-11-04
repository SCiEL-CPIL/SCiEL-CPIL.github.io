import numpy as np
import time

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = N, M, P
    
    # Dimensions of the factor matrices
    dim_U = n * m  # 15
    dim_V = m * p  # 35
    dim_W = p * n  # 21
    
    # Target Tensor T*
    T_star = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T_star[a, b, d] = 1.0
    
    def run_als(rank, max_iter=200, tol=1e-12, seed=42, verbose=False):
        np.random.seed(seed)
        
        # Multiple initialization strategies
        best_result = None
        best_error = float('inf')
        
        for init_strategy in range(3):
            if init_strategy == 0:
                # Random Gaussian
                U = np.random.randn(dim_U, rank)
                V = np.random.randn(dim_V, rank)
                W = np.random.randn(dim_W, rank)
            elif init_strategy == 1:
                # Sparse random
                U = np.random.randn(dim_U, rank)
                V = np.random.randn(dim_V, rank)
                W = np.random.randn(dim_W, rank)
                mask_u = np.random.rand(dim_U, rank) < 0.3
                mask_v = np.random.rand(dim_V, rank) < 0.3
                mask_w = np.random.rand(dim_W, rank) < 0.3
                U *= mask_u
                V *= mask_v
                W *= mask_w
            else:
                # Structured init based on partial sums
                U = np.zeros((dim_U, rank))
                V = np.zeros((dim_V, rank))
                W = np.zeros((dim_W, rank))
                for c in range(rank):
                    # Each component covers a subset of (i,j,k) triples
                    i0 = (c * n) // rank
                    j0 = (c * m) // rank
                    k0 = (c * p) // rank
                    for i in range(i0, min(i0+1, n)):
                        for j in range(j0, min(j0+1, m)):
                            U[i*m+j, c] = 1.0
                    for j in range(j0, min(j0+1, m)):
                        for k in range(k0, min(k0+1, p)):
                            V[j*p+k, c] = 1.0
                    for k in range(k0, min(k0+1, p)):
                        for i in range(i0, min(i0+1, n)):
                            W[k*n+i, c] = 1.0
                # Add random perturbation
                U += 0.01 * np.random.randn(dim_U, rank)
                V += 0.01 * np.random.randn(dim_V, rank)
                W += 0.01 * np.random.randn(dim_W, rank)
            
            # Normalize
            for arr in [U, V, W]:
                norms = np.linalg.norm(arr, axis=0, keepdims=True)
                norms[norms == 0] = 1
                arr /= norms
            
            current_error = float('inf')
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                Z = np.einsum('jc,kc->jkc', V, W).reshape(rank, -1).T
                T_flat_U = T_star.reshape(dim_U, -1)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(Z.T, T_flat_U.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V
                Y = np.einsum('ic,kc->ikc', U, W).reshape(rank, -1).T
                T_flat_V = np.moveaxis(T_star, 1, 0).reshape(dim_V, -1)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(Y.T, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Update W
                X = np.einsum('ic,jc->ijc', U, V).reshape(rank, -1).T
                T_flat_W = np.moveaxis(T_star, 2, 0).reshape(dim_W, -1)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(X.T, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                # Calculate Error
                T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
                error = np.linalg.norm(T_approx - T_star)
                
                if error < current_error:
                    current_error = error
                    if error < best_error:
                        best_error = error
                        best_result = (U.copy(), V.copy(), W.copy())
                        
                if error < tol:
                    break
            
            if current_error < best_error:
                best_error = current_error
                best_result = (U.copy(), V.copy(), W.copy())
        
        return best_result, best_error
    
    def check_and_round(U, V, W, tol=1e-8):
        """Check if decomposition is exact and try to round to half-integers."""
        T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
        if not np.allclose(T_approx, T_star, atol=tol):
            return None, None, None, False
            
        # Try rounding to half-integers
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        T_approx_round = np.einsum('ac,bc,dc->abd', U_round, V_round, W_round)
        if np.allclose(T_approx_round, T_star, atol=1e-10):
            return U_round, V_round, W_round, True
            
        # Try integer rounding
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        T_approx_int = np.einsum('ac,bc,dc->abd', U_int, V_int, W_int)
        if np.allclose(T_approx_int, T_star, atol=1e-10):
            return U_int, V_int, W_int, True
            
        # If rounding fails but error is tiny, return floats
        if np.max(np.abs(T_approx - T_star)) < 1e-10:
            return U, V, W, True
        return None, None, None, False
    
    best_result = None
    best_rank = float('inf')
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((dim_U, rank_sb))
    V_sb = np.zeros((dim_V, rank_sb))
    W_sb = np.zeros((dim_W, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    
    # Try lower ranks with multiple seeds
    # Lower bound is max(nm, mp, pn) = 35
    ranks_to_try = [35, 36, 37, 38, 39, 40, 42, 45, 50, 55, 60, 70, 80, 90]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
            
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                    break
            result, err = run_als(rank, max_iter=100, seed=seed)
            if result is not None and err < 1e-10:
                U_r, V_r, W_r, ok = check_and_round(result[0], result[1], result[2])
                if ok:
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U_r, V_r, W_r, rank)
                        found = True
                        break
            elif result is not None and err < 1e-6:
                # Close but not exact - try more iterations
                U_r, V_r, W_r, ok = check_and_round(result[0], result[1], result[2], tol=1e-6)
                if ok:
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U_r, V_r, W_r, rank)
                        found = True
                        break
        
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
    
    # Convert best result to list of lists for output
    if best_result is not None:
        U, V, W, r = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
