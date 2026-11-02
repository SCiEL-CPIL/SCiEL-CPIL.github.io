"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Lower bound on rank is max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Schoolbook rank is n*m*p = 96
    
    # Try to find a lower rank decomposition using iterative refinement
    # Start from the theoretical lower bound and work up
    
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    for target_rank in range(32, 97):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + attempt * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Alternating minimization
            for it in range(100):
                if time.time() > deadline - 15:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                        # This is a linear system: M @ U[i*m+j, :] = 1
                        # where M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            for c in range(target_rank):
                                M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        
                        # Solve M @ x = ones(p)
                        try:
                            x = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                            U[i*m+j, :] = x
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            for c in range(target_rank):
                                M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        
                        try:
                            x = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                            V[j*p+k, :] = x
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        
                        try:
                            x = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                            W[k*n+i, :] = x
                        except:
                            pass
                
                # Check convergence
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            error += (s - 1.0) ** 2
                
                if error < 1e-12:
                    found = True
                    break
            
            if found:
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if found:
            break
    
    # Verify the result
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
