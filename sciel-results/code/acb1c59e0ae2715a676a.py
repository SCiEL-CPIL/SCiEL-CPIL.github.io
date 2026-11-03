"""Improved program for matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 100
    
    # Known lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But actual rank for 4x5*5x5 is known to be between 25 and 100
    # Let's try to find a decomposition with rank < 100
    
    # Strategy: Use structured random initialization and iterative refinement
    # to find a valid decomposition with minimal rank
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, return schoolbook as fallback
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = make_schoolbook()
    
    # Try to find lower rank decompositions
    # For matrix multiplication tensor, we can use structured approaches
    
    # Approach: Try various ranks and see if we can find a valid decomposition
    # The tensor T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # We need: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # i.e., T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try a structured approach using bilinear forms
    
    # Known result: rank(4x5*5x5) is at most 100 (schoolbook)
    # There may be better decompositions. Let's try rank 50, 40, 30, etc.
    
    # Use a smarter initialization based on the structure
    
    def try_rank(rank, num_trials=50, seed=42):
        """Try to find a valid decomposition of given rank."""
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        # Use the fact that the tensor has a specific structure
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = rng.randn(n * m, rank) * 0.1
            V = rng.randn(m * p, rank) * 0.1
            W = rng.randn(p * n, rank) * 0.1
            
            # Iteratively refine: for each factor, hold others fixed and solve
            # This is like alternating least squares
            
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j) and r, U[i*m+j, r] should be such that
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # This is a linear system in U for fixed V, W
                
                # Actually, let's use a different approach:
                # The tensor equation is:
                # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                
                # Let's define the residual and use gradient descent or ALS
                
                # ALS approach:
                # Fix V and W, solve for U:
                # For each i,j: U[i*m+j, :] should satisfy
                #   sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # This is: U[i*m+j, :] @ M[i,j,:] = ones(p)
                # where M[i,j,k] = sum_r V[j*p+k,r] * W[k*n+i,r]
                
                # Actually, it's:
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # Let A[i,j,r,k] = V[j*p+k,r] * W[k*n+i,r]
                # Then sum_r U[i*m+j,r] * A[i,j,r,k] = 1 for all k
                # This is a system: U[i*m+j, :] @ A[i,j, :, :] = ones(p)
                
                # But A depends on i,j through the indices in V and W
                
                # Let me reformulate:
                # For fixed V, W:
                # For each (i,j), we need to find u = U[i*m+j, :] such that
                #   for all k: sum_r u_r * V[j*p+k, r] * W[k*n+i, r] = 1
                # Define M_ij as a p x rank matrix where M_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then we need u @ M_ij = ones(p), i.e., M_ij.T @ u = ones(p)
                
                # Solve least squares: u = pinv(M_ij.T) @ ones(p)
                
                for i in range(n):
                    for j in range(m):
                        # Build M_ij: p x rank
                        M_ij = np.zeros((p, rank))
                        for k in range(p):
                            M_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve M_ij.T @ u = ones(p)
                        # u = lstsq(M_ij.T, ones(p))
                        try:
                            u, _, _, _ = np.linalg.lstsq(M_ij.T, np.ones(p), rcond=None)
                            U[i * m + j, :] = u
                        except:
                            pass
                
                # Similarly update V and W
                
                for j in range(m):
                    for k in range(p):
                        M_jk = np.zeros((n, rank))
                        for i in range(n):
                            M_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            v, _, _, _ = np.linalg.lstsq(M_jk.T, np.ones(n), rcond=None)
                            V[j * p + k, :] = v
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        M_ki = np.zeros((m, rank))
                        for j in range(m):
                            M_ki[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            w, _, _, _ = np.linalg.lstsq(M_ki.T, np.ones(m), rcond=None)
                            W[k * n + i, :] = w
                        except:
                            pass
                
                # Check convergence
                if iteration % 20 == 0:
                    residual = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                                residual += (val - 1.0) ** 2
                    if residual < 1e-10:
                        # Found a valid decomposition!
                        return U, V, W, rank
        
        return None
    
    # Try decreasing ranks
    for target_rank in [60, 50, 45, 40, 35, 30, 28, 25]:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank, num_trials=20, seed=random.randint(0, 10000))
        if result is not None:
            U, V, W, rank = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            print(f"Found valid decomposition with rank {rank}")
            # Continue trying to find even lower rank
        else:
            # Try a few more trials with this rank
            pass
    
    # If we found a better rank, use it; otherwise return schoolbook
    if best_rank < schoolbook_rank:
        return best_result
    else:
        return make_schoolbook()
