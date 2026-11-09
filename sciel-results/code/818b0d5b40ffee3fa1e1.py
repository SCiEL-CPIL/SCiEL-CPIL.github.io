"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Tensor T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook rank
    rank_sb = n * m * p
    
    def make_schoolbook(rk):
        U = np.zeros((n*m, rk))
        V = np.zeros((m*p, rk))
        W = np.zeros((p*n, rk))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook(rank_sb)
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U ⊙ V ⊙ W equals the target tensor."""
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def snap_to_halfint(M, tol=0.01):
        """Snap values to nearest half-integer if close."""
        M2 = M.copy()
        for idx in np.ndindex(M2.shape):
            v = M2[idx]
            nearest = round(2*v) / 2
            if abs(v - nearest) < tol:
                M2[idx] = nearest
        return M2
    
    # Try to find lower rank decompositions
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = time.time() + 130  # Safety margin
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n)
    lower_bound = max(n*m, m*p, p*n)
    
    # We'll try a few specific ranks with random search
    ranks_to_try = []
    # Start from lower bound + some, go up
    for rk in range(lower_bound, rank_sb):
        ranks_to_try.append(rk)
    
    # Sort by likelihood of success (try middle ranks first, then extremes)
    # Actually, try from high to low near the schoolbook rank first
    # since those are more likely to converge
    # Let's try: rank_sb - 1, rank_sb - 2, ..., down to lower_bound
    
    # But we need to be time-efficient. Let's try a subset.
    # Prioritize: try rank 47, 46, 45, ... down to maybe 30
    
    np.random.seed(42)
    
    for target_rank in range(rank_sb - 1, lower_bound - 1, -1):
        if time.time() > deadline - 20:
            break
            
        # Multiple random attempts
        max_attempts = 5
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Use gradient-free optimization: iterative scaling / ALS-like
            # For tensor decomposition T = sum_r U_r ⊗ V_r ⊗ W_r
            # where U_r is a vector of size n*m, etc.
            # We want sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 for all i,j,k
            
            # Use a simple iterative method:
            # Fix two matrices, solve for the third (least squares)
            # Repeat until convergence
            
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
            
            # Precompute target tensor
            T = np.ones((n, m, p))
            
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                    
                # Update U: for each (i,j), U[i*m+j, :] should satisfy
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: U[i*m+j, :] * diag(V[j*p+k,:]*W[k*n+i,:]) = 1 for all k
                # Actually: for fixed (i,j), we need:
                # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # Let B_k[r] = V[j*p+k,r] * W[k*n+i,r]
                # Then U[i*m+j,:] @ B_k = 1 for all k
                # So U[i*m+j,:] is a solution to B @ U[i*m+j,:]^T = ones
                # where B is p x target_rank with B[k,r] = V[j*p+k,r]*W[k*n+i,r]
                
                for i in range(n):
                    for j in range(m):
                        B_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            B_mat[k, :] = V_curr[j*p+k, :] * W_curr[k*n+i, :]
                        # Solve B_mat @ x = ones(p)
                        try:
                            x, _, _, _ = np.linalg.lstsq(B_mat, np.ones(p), rcond=None)
                            U_curr[i*m+j, :] = x
                        except:
                            pass
                
                # Update V: for each (j,k), V[j*p+k, :] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        B_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            B_mat[i, :] = U_curr[i*m+j, :] * W_curr[k*n+i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(B_mat, np.ones(n), rcond=None)
                            V_curr[j*p+k, :] = x
                        except:
                            pass
                
                # Update W: for each (k,i), W[k*n+i, :] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        B_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            B_mat[j, :] = U_curr[i*m+j, :] * V_curr[j*p+k, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(B_mat, np.ones(m), rcond=None)
                            W_curr[k*n+i, :] = x
                        except:
                            pass
            
            # Check result
            if check_decomposition(U_curr, V_curr, W_curr, tol=1e-4):
                # Try to snap to half-integers
                U_snap = snap_to_halfint(U_curr, tol=0.05)
                V_snap = snap_to_halfint(V_curr, tol=0.05)
                W_snap = snap_to_halfint(W_curr, tol=0.05)
                
                if check_decomposition(U_snap, V_snap, W_snap, tol=1e-3):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    found = True
                    break
                else:
                    # Use the un-snapped version if it works
                    best_rank = target_rank
                    best_U, best_V, best_W = U_curr, V_curr, W_curr
                    found = True
                    break
        
        if found:
            # Try one more rank down
            continue
    
    # Convert to lists
    U_list = [[float(v) for v in row] for row in best_U]
    V_list = [[float(v) for v in row] for row in best_V]
    W_list = [[float(v) for v in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
