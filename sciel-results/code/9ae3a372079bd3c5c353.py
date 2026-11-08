"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_fb = np.zeros((n * m, rank_school))
    V_fb = np.zeros((m * p, rank_school))
    W_fb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_fb
    best_V = V_fb
    best_W = W_fb
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Reshape for efficient computation
        # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        # This is expensive for large rank, so do it in chunks
        rank = U.shape[1]
        T_approx = np.zeros_like(T)
        # Use einsum for efficiency
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_approx, T, atol=tol, rtol=0)
    
    def round_half(U, V, W):
        """Round entries to nearest half-integer and verify."""
        U_r = np.round(2 * U) / 2
        V_r = np.round(2 * V) / 2
        W_r = np.round(2 * W) / 2
        if verify(U_r, V_r, W_r, tol=1e-6):
            return U_r, V_r, W_r
        return None
    
    # Try to find lower rank decompositions using random initialization + iterative refinement
    # Strategy: random rank-r initialization, then use a tensor decomposition approach
    
    # The matrix multiplication tensor for (n,m,p) has known lower bounds.
    # For 4x4x7, schoolbook is 112. We try to find lower rank.
    
    # Use a simple approach: random rank-r factors, then optimize
    # to minimize ||T - sum_r U_r V_r W_r||
    
    # Actually, let's use a different approach: 
    # Randomly initialize rank-r factors and use alternating minimization
    # to fit the tensor.
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from low to high, stopping when we find a valid decomposition
    # or run out of time
    
    for rank in range(1, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize columns to help convergence
            for r in range(rank):
                norm_u = np.linalg.norm(U[:, r])
                norm_v = np.linalg.norm(V[:, r])
                norm_w = np.linalg.norm(W[:, r])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    scale = (norm_u * norm_v * norm_w) ** (1/3)
                    U[:, r] /= (norm_u / scale)
                    V[:, r] /= (norm_v / scale)
                    W[:, r] /= (norm_w / scale)
            
            # Alternating minimization: fix two, solve for third
            # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
            # For fixed V, W: U[:,r] = argmin ||T - sum_r U[:,r] (V[:,r] outer W[:,r])||
            # This is a linear system
            
            converged = False
            for iteration in range(200):
                if time.time() > deadline - 3:
                    break
                
                # Solve for U given V, W
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each r: U[:,r] satisfies sum_r U[a,r] * (V[:,r] * W[:,r] as vector in bc-space) = T[:,b,c]
                # Reshape T to (n*m, m*p*p*n)
                T_mat = T.reshape(n * m, m * p * p * n)
                
                # Build the design matrix: M[a, b*c] = sum_r V[b,r]*W[c,r] * delta(r==target)
                # Actually: T_mat[a, bc] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # So: T_mat = U @ M where M[r, bc] = V[b,r] * W[c,r]
                M = np.zeros((rank, m * p * p * n))
                for b in range(m * p):
                    for c in range(p * n):
                        M[:, b * p * n + c] = V[b, :] * W[c, :]
                
                # Solve U @ M = T_mat => U = T_mat @ M^+ (pseudo-inverse)
                try:
                    U_new = T_mat @ np.linalg.pinv(M)
                except:
                    break
                
                # Solve for V given U, W
                T_mat2 = T.reshape(m * p, n * m * p * n)
                M2 = np.zeros((rank, n * m * p * n))
                for a in range(n * m):
                    for c in range(p * n):
                        M2[:, a * p * n + c] = U[a, :] * W[c, :]
                try:
                    V_new = T_mat2 @ np.linalg.pinv(M2)
                except:
                    break
                
                # Solve for W given U, V
                T_mat3 = T.reshape(p * n, n * m * m * p)
                M3 = np.zeros((rank, n * m * m * p))
                for a in range(n * m):
                    for b in range(m * p):
                        M3[:, a * m * p + b] = U[a, :] * V[b, :]
                try:
                    W_new = T_mat3 @ np.linalg.pinv(M3)
                except:
                    break
                
                # Check convergence
                diff = (np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W))
                U, V, W = U_new, V_new, W_new
                
                if diff < 1e-10:
                    converged = True
                    break
            
            # Verify the solution
            if verify(U, V, W, tol=1e-8):
                # Try to round to half-integers
                rounded = round_half(U, V, W)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    if U_r.shape[1] < best_rank:
                        best_rank = rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                else:
                    # Keep as-is if it's exact
                    if U.shape[1] < best_rank:
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W
                break  # Found valid decomposition for this rank, move to next lower rank
            
            # Check time
            if time.time() > deadline - 3:
                break
    
    # Convert to lists
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
