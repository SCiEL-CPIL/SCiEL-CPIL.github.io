"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 140.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def max_error(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    def try_rank(target_rank, max_attempts=200, max_iters=1000):
        """Try to find a valid decomposition with given rank."""
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline:
                return None
            
            seed = attempt * 104729 + target_rank * 7919
            np.random.seed(seed)
            
            # Initialize
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # ALS iterations
            for it in range(max_iters):
                if time.time() - start_time > deadline:
                    return None
                
                # Solve for U: for each (i,j), find U[i*m+j,:] such that
                # dot(U[i*m+j,:], V[j*p+k,:]*W[k*n+i,:]) = 1 for all k
                for i in range(n):
                    for j in range(m):
                        M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        try:
                            U[i*m+j, :], _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        except:
                            return None
                
                # Solve for V: for each (j,k), find V[j*p+k,:] such that
                # dot(V[j*p+k,:], U[i*m+j,:]*W[k*n+i,:]) = 1 for all i
                for j in range(m):
                    for k in range(p):
                        M = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        try:
                            V[j*p+k, :], _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        except:
                            return None
                
                # Solve for W: for each (k,i), find W[k*n+i,:] such that
                # dot(W[k*n+i,:], U[i*m+j,:]*V[j*p+k,:]) = 1 for all j
                for k in range(p):
                    for i in range(n):
                        M = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        try:
                            W[k*n+i, :], _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        except:
                            return None
                
                err = max_error(U, V, W)
                if err < 1e-7:
                    # Polish
                    for _ in range(500):
                        for i in range(n):
                            for j in range(m):
                                M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                                try:
                                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                                except:
                                    pass
                        for j in range(m):
                            for k in range(p):
                                M = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                                try:
                                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                                except:
                                    pass
                        for k in range(p):
                            for i in range(n):
                                M = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                                try:
                                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                                except:
                                    pass
                        if max_error(U, V, W) < 1e-10:
                            break
                    
                    if max_error(U, V, W) < 1e-7:
                        return (U, V, W)
        
        return None
    
    # Try ranks from low to high
    for target_rank in range(27, best_rank):
        if time.time() - start_time > deadline:
            break
        
        result = try_rank(target_rank, max_attempts=50, max_iters=200)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    return best_result
# EVOLVE-BLOCK-END
