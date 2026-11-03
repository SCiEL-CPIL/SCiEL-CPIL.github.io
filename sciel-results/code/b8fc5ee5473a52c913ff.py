"""Matrix multiplication 2x4*4x5 rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition (rank = n*m*p = 40)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # The tensor T is all-ones: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want to find minimal rank r and matrices U (nm x r), V (mp x r), W (pn x r)
    # such that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Use alternating least squares / structured search
    # For a given rank r, we can try to find a decomposition.
    # Key insight: This is a bilinear problem in each factor when the other two are fixed.
    
    # Let's try to find decompositions of decreasing rank.
    # Known lower bound: rank >= max(n*m, m*p, p*n) / something... 
    # Actually for matrix mult, rank >= n*m*p / max(n,m,p) is not a valid bound.
    # The slice rank lower bound: rank >= max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But actual rank might be higher.
    
    # Let's use a randomized approach with ALS (Alternating Least Squares)
    # For each trial:
    #   1. Initialize U, V, W randomly
    #   2. Iterate: fix V,W -> solve for U; fix U,W -> solve for V; fix U,V -> solve for W
    #   3. Check if the reconstruction is exact (within tolerance)
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # Efficient check: compute the full tensor and compare
        # T_recon[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We can compute this as: for each c, outer product U[:,c] (reshaped) * V[:,c] * W[:,c]
        # But that's expensive. Instead, use einsum or manual loop.
        
        # Reshape for computation
        U_3d = U.reshape(n, m, -1)  # (n, m, r)
        V_3d = V.reshape(m, p, -1)  # (m, p, r)
        W_3d = W.reshape(p, n, -1)  # (p, n, r)
        
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ijk'
        r = U.shape[1]
        T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T_recon, 1.0, atol=tol, rtol=0)
    
    def als_decomposition(target_rank, max_iter=200, tol=1e-10):
        """Try to find a rank-`target_rank` decomposition using ALS."""
        # Random initialization with small values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Fix V and W, solve for U
            # For each (i,j), we need: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
            # This is: U[i*m+j,:] @ M_ij = ones(p) where M_ij[k,c] = V[j*p+k,c]*W[k*n+i,c]
            # So U[i*m+j,:] is the solution to M_ij.T @ U[i*m+j,:] = ones(p)
            # i.e., solve M_ij.T @ x = ones(p) for x
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # M_ij has shape (p, target_rank)
                    M_ij = np.zeros((p, target_rank))
                    for k in range(p):
                        M_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve M_ij.T @ x = ones(p)
                    # This is an underdetermined system (target_rank >= p)
                    # Use least squares: x = pinv(M_ij.T) @ ones(p)
                    # Or: minimize ||M_ij.T @ x - ones(p)||^2
                    rhs = np.ones(p)
                    A_mat = M_ij.T  # shape (target_rank, p)
                    # x = A_mat.T @ inv(A_mat @ A_mat.T) @ rhs  if A_mat @ A_mat.T is invertible
                    # Use lstsq
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    except:
                        x = np.zeros(target_rank)
                    U_new[i * m + j, :] = x
            
            # Fix U and W, solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # Need: sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
                    # M_jk[i,c] = U[i*m+j,c] * W[k*n+i,c], shape (n, target_rank)
                    M_jk = np.zeros((n, target_rank))
                    for i in range(n):
                        M_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    rhs = np.ones(n)
                    A_mat = M_jk.T  # shape (target_rank, n)
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    except:
                        x = np.zeros(target_rank)
                    V_new[j * p + k, :] = x
            
            # Fix U and V, solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # Need: sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
                    # M_ki[j,c] = U[i*m+j,c] * V[j*p+k,c], shape (m, target_rank)
                    M_ki = np.zeros((m, target_rank))
                    for j in range(m):
                        M_ki[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    rhs = np.ones(m)
                    A_mat = M_ki.T  # shape (target_rank, m)
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    except:
                        x = np.zeros(target_rank)
                    W_new[k * n + i, :] = x
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            if check_decomposition(U, V, W, tol=1e-8):
                # Verify with tighter tolerance
                if check_decomposition(U, V, W, tol=1e-12):
                    return (U, V, W)
        
        # Final check
        if check_decomposition(U, V, W, tol=1e-9):
            return (U, V, W)
        return None
    
    def try_rank_with_restarts(target_rank, n_restarts=50):
        """Try multiple random restarts for a given target rank."""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            np.random.seed(restart * 1000 + target_rank)
            result = als_decomposition(target_rank, max_iter=150, tol=1e-10)
            if result is not None:
                return result
        return None
    
    # Try ranks from low to high
    # Lower bound is 20 (slice rank), but let's start from 20
    # The schoolbook is 40, so we have room to improve.
    # Known results: rank(2x4*4x5) is likely around 24-30
    
    for target_rank in range(20, rank_school + 1):
        if time.time() > deadline:
            break
        
        # Determine number of restarts based on remaining time
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # Estimate time per ALS attempt
        n_restarts = min(30, max(5, int(remaining / 1.5)))
        
        result = try_rank_with_restarts(target_rank, n_restarts=n_restarts)
        if result is not None:
            U, V, W = result
            # Round to clean values if possible
            best_rank = target_rank
            best_U = U
            best_V = V
            best_W = W
            # Continue trying lower ranks? No, we went from low to high, so this is the best.
            # Actually we should break since we found the minimum.
            break
    
    # If we found a better decomposition, use it; otherwise use schoolbook
    if best_rank < rank_school:
        # Try to clean up the values - round to nearest half-integer or integer
        # Check if rounding works
        def clean_matrix(M):
            """Try to round matrix to clean values."""
            M_clean = np.round(M * 2) / 2  # round to nearest 0.5
            return M_clean
        
        U_clean = clean_matrix(best_U)
        V_clean = clean_matrix(best_V)
        W_clean = clean_matrix(best_W)
        
        if check_decomposition(U_clean, V_clean, W_clean, tol=1e-9):
            best_U = U_clean
            best_V = V_clean
            best_W = W_clean
        else:
            # Try rounding to integers
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            if check_decomposition(U_int, V_int, W_int, tol=1e-9):
                best_U = U_int
                best_V = V_int
                best_W = W_int
            # Otherwise keep the float values
    
    # Convert to lists for the required interface
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
