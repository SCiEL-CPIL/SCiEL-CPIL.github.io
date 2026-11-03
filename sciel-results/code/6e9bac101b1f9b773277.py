"""Matrix multiplication tensor decomposition for 3x4 * 4x7."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    R = U.shape[1]
    C_comp = np.zeros((n, p))
    
    U_reshaped = U.reshape(n, m, R)
    S_A = np.einsum('ijr,ij->ir', U_reshaped, A)
    
    V_reshaped = V.reshape(m, p, R)
    S_B = np.einsum('jkr,jk->kr', V_reshaped, B)
    
    W_reshaped = W.reshape(p, n, R)
    C_comp = np.einsum('kir,ir,kr->ik', W_reshaped, S_A, S_B)
    
    return np.allclose(C_true, C_comp, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 84
    
    # The tensor T_{i,j,k} = 1 for all i,j,k
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Known lower bounds for matrix multiplication rank:
    # For 3x4 * 4x7, the trivial rank is 84.
    # Strassen-style improvements may exist but are hard to find.
    # Let's try a few approaches:
    # 1. Random search with local optimization
    # 2. Try known small-rank constructions if applicable
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety
    
    # First, let's try to verify if there's a known lower rank
    # For 3x4 * 4x7, the rank is at least max(3*4, 4*7, 3*7) = max(12, 28, 21) = 28
    # But that's a very weak lower bound. The true rank is likely close to 84.
    
    # Let's try random search with least squares for ranks slightly below 84
    # This is computationally expensive, so we'll try a limited number of attempts
    
    best_rank = rank_schoolbook
    best_solution = None
    
    # Try ranks from 83 down to 78, with a few random restarts each
    for R in range(rank_schoolbook - 1, max(75, rank_schoolbook - 15), -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # Use alternating least squares approach
            # Fix V and W, solve for U
            # Fix U and W, solve for V
            # Fix U and V, solve for W
            
            # For each iteration, we can solve linear systems
            # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # Let's vectorize: for fixed V and W, U[i*m+j, :] appears in
            # equations for all k: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
            # This is a linear system in U[i*m+j, :]
            
            # Let's implement ALS
            max_iter = 50
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), solve linear system
                for i in range(n):
                    for j in range(m):
                        # A matrix: for each r, coefficient is sum_k V[j*p+k, r] * W[k*n+i, r]
                        # Actually, the equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each k
                        # So for fixed (i,j), we have p equations:
                        # For each k: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                        # This is a linear system: M @ U[i*m+j, :] = b
                        # where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # and b[k] = 1
                        
                        M = np.zeros((p, R))
                        for k in range(p):
                            for r in range(R):
                                M[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        b = np.ones(p)
                        
                        # Solve M @ x = b in least squares sense
                        try:
                            x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # Update V: for each (j,k), solve linear system
                for j in range(m):
                    for k in range(p):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each i
                        # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        # b[i] = 1
                        M = np.zeros((n, R))
                        for i in range(n):
                            for r in range(R):
                                M[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        b = np.ones(n)
                        
                        try:
                            x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass
                
                # Update W: for each (k,i), solve linear system
                for k in range(p):
                    for i in range(n):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each j
                        # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # b[j] = 1
                        M = np.zeros((m, R))
                        for j in range(m):
                            for r in range(R):
                                M[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        b = np.ones(m)
                        
                        try:
                            x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass
            
            # Check if solution is valid
            if time.time() > deadline:
                break
                
            if _verify(U, V, W):
                best_rank = R
                best_solution = (U, V, W)
                break
        
        if best_solution is not None:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    rank = rank_schoolbook
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
