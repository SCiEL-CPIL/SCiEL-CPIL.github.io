import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition_full(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W form an exact decomposition of the matrix multiplication tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    
    Non-zero entries (1s) are at:
    a = i*m + j
    b = j*p + k
    c = k*n + i
    for all i, j, k.
    
    All other entries must be 0.
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    
    # Check 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a] * V[b], W[c])
                if abs(val - 1.0) > tol:
                    return False
                    
    # Check 0s
    # We need to check all (a, b, c) that are NOT of the form above.
    # Total entries: (n*m) * (m*p) * (p*n) = n^2 m^2 p^2.
    # For 3,5,7: 9 * 25 * 49 = 11025 entries.
    # This is small enough to check all.
    
    # Precompute valid indices
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.add((a, b, c))
                
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid_indices:
                    val = np.dot(U[a] * V[b], W[c])
                    if abs(val) > tol:
                        return False
                        
    return True

def _try_rank_r_full(n, m, p, r, max_attempts=500, seed=42):
    """
    Try to find a rank-r decomposition.
    """
    np.random.seed(seed)
    
    for attempt in range(max_attempts):
        # Generate random sparse U and V
        # Use small integers to increase chance of integer W
        U = np.random.randint(-1, 2, size=(n * m, r))
        V = np.random.randint(-1, 2, size=(m * p, r))
        
        W = np.zeros((p * n, r))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Construct linear system for W[k*n+i, :]
                # Variables: w_0, ..., w_{r-1}
                # Equations: for j in 0..m-1:
                #   sum_c U[i*m+j, c] * V[j*p+k, c] * w_c = 1
                
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Solve A_mat * w_vec = ones(m)
                try:
                    w_vec, residuals, rank, s = lstsq(A_mat, np.ones(m), cond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if solution is close to integers
                w_int = np.round(w_vec)
                if np.linalg.norm(A_mat @ w_int - np.ones(m)) > 1e-9:
                    valid = False
                    break
                
                W[k * n + i, :] = w_int
        
        if not valid:
            continue
            
        # Final verification including zeros
        if _verify_decomposition_full(U, V, W, n, m, p, tol=1e-9):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Trivial solution fallback
    rank_trivial = n * m * p
    U_triv = np.zeros((n * m, rank_trivial))
    V_triv = np.zeros((m * p, rank_trivial))
    W_triv = np.zeros((p * n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, col] = 1.0
                V_triv[j * p + k, col] = 1.0
                W_triv[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_trivial
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Lower bound
    lb = max(n * m, m * p, p * n)
    
    # Search from lb upwards
    # We will try a few ranks.
    # If we find one, we return it.
    # To be safe, we might try lb, lb+1, ... up to some limit.
    
    for r in range(lb, rank_trivial + 1):
        if time.time() > deadline:
            break
            
        # More attempts for lower ranks
        attempts = 100 if r < lb + 20 else 50
        
        found = False
        for seed in range(attempts):
            if time.time() > deadline:
                break
                
            res = _try_rank_r_full(n, m, p, r, max_attempts=1, seed=seed + r * 1000)
            if res is not None:
                U, V, W = res
                best_rank = r
                best_U, best_V, best_W = U, V, W
                found = True
                break
                
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
