import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def compute_slice_rank_lower_bound():
    """
    Compute the slice-rank lower bound for the matrix multiplication tensor T_{n,m,p}.
    
    For a rank-r decomposition, each rank-1 slice U_c ⊗ V_c ⊗ W_c has:
    - U_c (dim_u x 1) with at most 1 nonzero per row-block
    - V_c (dim_v x 1) with at most 1 nonzero per row-block  
    - W_c (dim_w x 1) with at most 1 nonzero per row-block
    
    The slice rank lower bound: r >= max over all (I,J,K) slices of the number
    of "active" positions. More precisely, for each fixed K, the matrix T[:,:,K]
    must be covered by rank-1 terms where U_c has support only where W_c[K]!=0.
    
    A simpler bound: r >= max{rank(T[:,:,K]) for K} = max{rank(T[I,:,J]) for I,J}
    """
    n, m, p = DIMENSIONS
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    # Slice rank bound: for each K, rank of T[:,:,K]
    max_rank_K = 0
    for K in range(dim_w):
        r = np.linalg.matrix_rank(T[:, :, K], tol=1e-10)
        max_rank_K = max(max_rank_K, r)
    
    # For each I, rank of T[I,:,:]
    max_rank_I = 0
    for I in range(dim_u):
        r = np.linalg.matrix_rank(T[I, :, :], tol=1e-10)
        max_rank_I = max(max_rank_I, r)
    
    # For each J, rank of T[:,J,:]
    max_rank_J = 0
    for J in range(dim_v):
        r = np.linalg.matrix_rank(T[:, J, :], tol=1e-10)
        max_rank_J = max(max_rank_J, r)
    
    lower_bound = max(max_rank_K, max_rank_I, max_rank_J)
    
    # Strassen-style bound: the tensor has a "structured" lower bound
    # For 2x3x5: schoolbook=30, known best is 28 (from splitting into 2x2+2x2+2x1 blocks)
    # The slice rank bound gives a rigorous lower bound
    return lower_bound

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    # T[I, J, K] = 1 if I=i*m+j, J=j*p+k, K=k*n+i for valid (i,j,k)
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    # Pre-compute RHS vectors for ALS updates
    RHS_U = T.reshape(dim_u, dim_v * dim_w)  # (dim_u, dim_v*dim_w)
    RHS_V = np.zeros((dim_v, dim_u * dim_w))
    for j in range(dim_v):
        RHS_V[j] = T[:, j, :].flatten()
    RHS_W = np.zeros((dim_w, dim_u * dim_v))
    for k in range(dim_w):
        RHS_W[k] = T[:, :, k].flatten()

    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety

    def verify_decomposition(U, V, W, r):
        """Verify exact decomposition and check half-integer constraint."""
        T_approx = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_approx += U[:, c][:, np.newaxis, np.newaxis] * V[:, c][np.newaxis, :, np.newaxis] * W[:, c][np.newaxis, np.newaxis, :]
        
        if not np.allclose(T_approx, T, atol=1e-10):
            return False
        
        # Check if entries are half-integers (within tolerance)
        U_check = np.round(U * 2) / 2
        V_check = np.round(V * 2) / 2
        W_check = np.round(W * 2) / 2
        
        if not np.allclose(U, U_check, atol=1e-8):
            return False
        if not np.allclose(V, V_check, atol=1e-8):
            return False
        if not np.allclose(W, W_check, atol=1e-8):
            return False
        
        # Final exact verification with rounded values
        T_approx2 = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_approx2 += U_check[:, c][:, np.newaxis, np.newaxis] * V_check[:, c][np.newaxis, :, np.newaxis] * W_check[:, c][np.newaxis, np.newaxis, :]
        
        if not np.allclose(T_approx2, T, atol=1e-10):
            return False
        
        return True

    def try_rank(r, max_iters=150, tol=1e-9, num_restarts=40):
        """
        Attempts to find a decomposition of rank r using vectorized ALS.
        Returns (U, V, W) if successful, else None.
        """
        best_error = np.inf
        best_result = None
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(777 + restart * 13)
            U = np.random.randn(dim_u, r) * 0.5
            V = np.random.randn(dim_v, r) * 0.5
            W = np.random.randn(dim_w, r) * 0.5
            
            prev_error = np.inf
            stagnant_count = 0
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U: For each I, solve M_U @ U[I,:] = RHS_U[I,:]
                # M_U[(J,K), c] = V[J,c] * W[K,c]
                M_U = np.einsum('jc,kc->jkc', V, W).reshape(dim_v * dim_w, r)
                try:
                    M_U_pinv = np.linalg.pinv(M_U)
                    U_new = (M_U_pinv @ RHS_U.T).T
                except np.linalg.LinAlgError:
                    continue
                
                # Update V: For each J, solve M_V @ V[J,:] = RHS_V[J,:]
                # M_V[(I,K), c] = U_new[I,c] * W[K,c]
                M_V = np.einsum('ic,kc->ikc', U_new, W).reshape(dim_u * dim_w, r)
                try:
                    M_V_pinv = np.linalg.pinv(M_V)
                    V_new = (M_V_pinv @ RHS_V.T).T
                except np.linalg.LinAlgError:
                    continue
                
                # Update W: For each K, solve M_W @ W[K,:] = RHS_W[K,:]
                # M_W[(I,J), c] = U_new[I,c] * V_new[J,c]
                M_W = np.einsum('ic,jc->ijc', U_new, V_new).reshape(dim_u * dim_v, r)
                try:
                    M_W_pinv = np.linalg.pinv(M_W)
                    W_new = (M_W_pinv @ RHS_W.T).T
                except np.linalg.LinAlgError:
                    continue
                
                U, V, W = U_new, V_new, W_new
                
                # Compute error
                T_approx = np.zeros((dim_u, dim_v, dim_w))
                for c in range(r):
                    T_approx += U[:, c][:, np.newaxis, np.newaxis] * V[:, c][np.newaxis, :, np.newaxis] * W[:, c][np.newaxis, np.newaxis, :]
                
                error = np.linalg.norm(T_approx - T)
                
                if error < best_error:
                    best_error = error
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if error < tol:
                    # Final polish pass with tighter tolerance and more iterations
                    for polish_it in range(100):
                        M_U = np.einsum('jc,kc->jkc', V, W).reshape(dim_v * dim_w, r)
                        try:
                            M_U_pinv = np.linalg.pinv(M_U)
                            U = (M_U_pinv @ RHS_U.T).T
                        except:
                            break
                        
                        M_V = np.einsum('ic,kc->ikc', U, W).reshape(dim_u * dim_w, r)
                        try:
                            M_V_pinv = np.linalg.pinv(M_V)
                            V = (M_V_pinv @ RHS_V.T).T
                        except:
                            break
                        
                        M_W = np.einsum('ic,jc->ijc', U, V).reshape(dim_u * dim_v, r)
                        try:
                            M_W_pinv = np.linalg.pinv(M_W)
                            W = (M_W_pinv @ RHS_W.T).T
                        except:
                            break
                        
                        T_approx = np.zeros((dim_u, dim_v, dim_w))
                        for c in range(r):
                            T_approx += U[:, c][:, np.newaxis, np.newaxis] * V[:, c][np.newaxis, :, np.newaxis] * W[:, c][np.newaxis, np.newaxis, :]
                        error = np.linalg.norm(T_approx - T)
                        if error < 1e-14:
                            break
                    
                    if verify_decomposition(U, V, W, r):
                        return U, V, W
                    
                    if best_result is None:
                        best_result = (U.copy(), V.copy(), W.copy())
                    break
                
                # Check for stagnation
                if error > prev_error * 0.999:
                    stagnant_count += 1
                else:
                    stagnant_count = 0
                prev_error = error
                
                if stagnant_count > 15:
                    break
        
        # If we have a best result, try to verify it
        if best_result is not None and best_error < 1e-6:
            U, V, W = best_result
            # Try rounding to half-integers and verify
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            
            T_approx = np.zeros((dim_u, dim_v, dim_w))
            for c in range(r):
                T_approx += U_half[:, c][:, np.newaxis, np.newaxis] * V_half[:, c][np.newaxis, :, np.newaxis] * W_half[:, c][np.newaxis, np.newaxis, :]
            
            if np.allclose(T_approx, T, atol=1e-8):
                return U_half, V_half, W_half
            
            # Also try integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            T_approx2 = np.zeros((dim_u, dim_v, dim_w))
            for c in range(r):
                T_approx2 += U_int[:, c][:, np.newaxis, np.newaxis] * V_int[:, c][np.newaxis, :, np.newaxis] * W_int[:, c][np.newaxis, np.newaxis, :]
            
            if np.allclose(T_approx2, T, atol=1e-8):
                return U_int, V_int, W_int
        
        return None

    # Compute slice rank lower bound to guide search
    lower_bound = compute_slice_rank_lower_bound()
    
    # Known results: 2x3x5 has rank 28 (via Strassen on 2x2 blocks)
    # Start from the best known and search downward
    best_known_rank = 28
    
    # Try ranks from best known down to the lower bound
    for r in range(best_known_rank, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback: Use the known 28-rank decomposition from Strassen splitting
    # Split B (3x5) into B1 (3x2), B2 (3x2), B3 (3x1)
    # For each 2x2 block, use Strassen (rank 7), for 2x1 use schoolbook (rank 4)
    # Total: 7 + 7 + 4 = 18... but we need to account for the full structure
    
    # Actually, let's use the verified 28-rank construction:
    # A is 2x3, B is 3x5, C is 2x5
    # Split B into B[:,0:2], B[:,2:4], B[:,4:5]
    # For 2x2*3x2: rank 7 (Strassen), for 2x2*3x2: rank 7, for 2x2*3x1: rank 4
    # But we need to handle the 3-row dimension properly
    
    # Simpler: use the standard 28-rank decomposition
    # This is the known optimal for 2x3x5
    
    # Construct using the block approach:
    # A = [A1 A2 A3] where Ai is 2x1
    # B = [B1 B2 B3 B4 B5] where Bj is 3x1
    # C = A*B = sum of Ai*Bj
    
    # The 28-rank construction uses:
    # - For columns 0,1 of B: treat as 2x2 block with Strassen (7 multiplications)
    # - For columns 2,3 of B: treat as 2x2 block with Strassen (7 multiplications)  
    # - For column 4 of B: schoolbook (4 multiplications)
    # But we need to be careful about how the 3-row dimension interacts
    
    # Let me just use the schoolbook as a safe fallback with rank 30
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
