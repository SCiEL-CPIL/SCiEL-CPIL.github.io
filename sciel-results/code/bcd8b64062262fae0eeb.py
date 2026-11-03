"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_start_time = time.time()
_DEADLINE = 145  # seconds, safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the decomposition: for each (i,j,k), T[i,j,k] = sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
    # We check that this equals 1 if (i,j,k) is a valid multiplication index, else 0
    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition must satisfy: for all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # (since the matrix multiplication tensor has all-ones entries for the relevant indices)
    
    # More precisely: the tensor T has T[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # The rank-r decomposition means: T[i,j,k] = sum_{t=0}^{r-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each t, the contribution is U[:,t] outer V[:,t] outer W[:,t]
    # We need: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all i,j,k
    
    # Check a sample of entries or all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, max_iter=500, seed=None):
    """Try to find a rank-r decomposition using iterative refinement."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize with random matrices
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        # Update U: for each (i,j) and t, U[i*m+j, t] should satisfy
        # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all k
        # This is a linear system in U[i*m+j, :] for fixed (i,j)
        # Let a_t = V[j*p+k,t] * W[k*n+i,t] for each k
        # We need: U[i*m+j,:] dot a_k = 1 for all k
        # So we solve: A_k @ U[i*m+j,:] = 1 where A_k[t] = V[j*p+k,t]*W[k*n+i,t]
        
        for i in range(n):
            for j in range(m):
                # Build system: for each k, sum_t U[i*m+j,t] * (V[j*p+k,t] * W[k*n+i,t]) = 1
                A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b = np.ones(p)
                # Solve A @ x = b, where x = U[i*m+j, :]
                try:
                    x, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k) and t
        for j in range(m):
            for k in range(p):
                A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b = np.ones(n)
                try:
                    x, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i) and t
        for k in range(p):
            for i in range(n):
                A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b = np.ones(m)
                try:
                    x, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Check convergence
        if iteration % 10 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for t in range(r):
                            val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                        max_err = max(max_err, abs(val - 1.0))
            if max_err < 1e-8:
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Start with schoolbook rank
    schoolbook_rank = n * m * p
    
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Try ranks from schoolbook_rank - 1 downward
    for r in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
        if time.time() > _DEADLINE - 10:
            break
        
        # Try a few random seeds
        for seed in range(5):
            if time.time() > _DEADLINE - 10:
                break
            
            result = _try_rank(r, n, m, p, max_iter=200, seed=seed)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                    best_rank = r
                    break  # Found a valid decomposition for this rank, move to lower
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
