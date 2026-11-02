"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    rank = U.shape[1]
    
    # The tensor T[i,j,k] = delta(i,k) for matrix multiplication
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check a few random entries
    import random
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        # Expected value: 1 if i==k, else 0 (for the multiplication tensor)
        expected = 1.0 if (i == k) else 0.0
        
        # Compute from decomposition
        actual = 0.0
        for r in range(rank):
            actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        if abs(actual - expected) > tol:
            return False
    
    return True

def construct():
    """
    Constructs a tensor decomposition for 2x3 matrix multiplication by 3x9.
    Uses schoolbook as baseline, attempts to find lower rank.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline with safety margin
    
    # Start with schoolbook decomposition (rank 54)
    rank_schoolbook = n * m * p
    U_best = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_best = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_best = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j, col] = 1.0
                V_best[j * p + k, col] = 1.0
                W_best[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using random initialization
    # The tensor for 2x3x9 multiplication has 2*3*9 = 54 non-zero entries
    # We need to find a rank-r decomposition
    
    # Try ranks from 30 to 53 (lower than schoolbook)
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Simple iterative refinement
            for iteration in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), U[i*m+j, r] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta(i,k)
                # This is a linear system in U for fixed V, W
                
                for i in range(n):
                    for j in range(m):
                        # Build linear system for U[i*m+j, :]
                        # For each k: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = delta(i,k)
                        # This is: U[i*m+j, :] @ diag(V[j*p+k,:] * W[k*n+i,:]) = delta(i,k)
                        
                        # Build A matrix: A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        A = np.zeros((p, target_rank))
                        b = np.zeros(p)
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                            b[k] = 1.0 if (i == k) else 0.0
                        
                        # Solve A @ U[i*m+j, :]^T = b, i.e., A.T @ U[i*m+j, :] = b.T
                        # Use least squares
                        try:
                            U_new = np.linalg.lstsq(A.T, b, rcond=None)[0]
                            U[i*m+j, :] = U_new
                        except:
                            pass
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        # For each i: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta(i,k)
                        # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        A = np.zeros((n, target_rank))
                        b = np.zeros(n)
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                            b[i] = 1.0 if (i == k) else 0.0
                        
                        try:
                            V_new = np.linalg.lstsq(A.T, b, rcond=None)[0]
                            V[j*p+k, :] = V_new
                        except:
                            pass
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        # For each j: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta(i,k)
                        # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        A = np.zeros((m, target_rank))
                        b = np.zeros(m)
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                            b[j] = 1.0 if (i == k) else 0.0
                        
                        try:
                            W_new = np.linalg.lstsq(A.T, b, rcond=None)[0]
                            W[k*n+i, :] = W_new
                        except:
                            pass
            
            # Verify the decomposition
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                best_rank = target_rank
                U_best = U.copy()
                V_best = V.copy()
                W_best = W.copy()
                break  # Found valid decomposition at this rank, try lower
    
    # Return the best decomposition found
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
# EVOLVE-BLOCK-END
