"""Matrix multiplication 4x5x5 - structured low-rank search.
Uses the bilinear constraint structure of matrix multiplication to search
for rank < 100 decompositions via random init + iterative refinement.
"""

import time
import numpy as np

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 140  # seconds, leave 10s safety margin

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W form a valid decomposition of the matmul tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # The matmul tensor T[i,j,k] = delta[i,j,k] where:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
    # i.e., for each (a,b,c) with a=i*m+j, b=j'*p+k, c=k'*n+i':
    #   sum_r U[a,r]*V[b,r]*W[c,r] = 1 if a=i*m+j, b=j*p+k, c=k*n+i with matching j
    #   = 0 otherwise
    
    # More precisely: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = delta_{i,j,k} meaning T[i*m+j, j*p+k, k*n+i] = 1
    # But actually the standard form is:
    #   sum_j A[i,j] B[j,k] = C[i,k]
    # As a tensor over (i,j) x (j,k) x (k,i):
    #   T[(i,j),(j,k),(k,i)] = 1 if all indices consistent, 0 otherwise
    #   Specifically: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Check: for each term r, U[a,r]*V[b,r]*W[c,r]
    # The sum over r must equal 1 when (a,b,c) corresponds to (i*m+j, j*p+k, k*n+i)
    # and 0 otherwise.
    
    # Efficient check: compute the implied C matrix for random A, B
    # C_implied[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal sum_j A[i,j]*B[j,k] for ALL A, B
    # So: sum_j U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] summed over r must be
    #   delta_{j,j'} for the j index... 
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k] / (U*V*W)
    # This is getting circular. Let me use the direct tensor check.
    
    # Direct approach: build the tensor and verify
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # must equal 1 if exists i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    # and 0 otherwise
    
    # For efficiency, check only the non-zero entries and a sample of zero entries
    # Non-zero: for each i in [0,n), j in [0,m), k in [0,p):
    #   a = i*m+j, b = j*p+k, c = k*n+i, T[a,b,c] should be 1
    # Zero: all other (a,b,c) triples should give 0
    
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check some zero entries (sample)
    rng = np.random.RandomState(42)
    num_checks = min(500, (n*m)*(m*p)*(p*n) - n*m*p)
    for _ in range(num_checks):
        a = rng.randint(n*m)
        b = rng.randint(m*p)
        c = rng.randint(p*n)
        # Check if this is a "valid" triple
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if j == j2 and k == k2 and i == i2:
            continue  # This is a non-zero entry, skip
        val = np.sum(U[a] * V[b] * W[c])
        if abs(val) > tol:
            return False
    
    return True

def _try_rank(rank, n, m, p, max_iters=200, num_trials=3, tol=1e-6):
    """Try to find a rank-rank decomposition via iterative refinement."""
    if time.time() - START_TIME > TIME_LIMIT:
        return None
    
    for trial in range(num_trials):
        if time.time() - START_TIME > TIME_LIMIT:
            return None
        
        rng = np.random.RandomState(trial * 12345 + 67890)
        
        # Initialize with random values
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Normalize columns to help convergence
        for r in range(rank):
            norm = np.sqrt(np.sum(U[:, r]**2) + np.sum(V[:, r]**2) + np.sum(W[:, r]**2))
            if norm > 1e-10:
                U[:, r] /= norm
                V[:, r] /= norm
                W[:, r] /= norm
        
        best_err = np.inf
        for it in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT:
                return None
            
            # Compute error on non-zero entries
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U[a] * V[b] * W[c])
                        err += (val - 1.0) ** 2
            
            # Compute error on a sample of zero entries
            rng2 = np.random.RandomState(it + trial * 1000)
            zero_err = 0.0
            num_zero_checks = 200
            for _ in range(num_zero_checks):
                a = rng2.randint(n * m)
                b = rng2.randint(m * p)
                c = rng2.randint(p * n)
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and k == k2 and i == i2:
                    continue
                val = np.sum(U[a] * V[b] * W[c])
                zero_err += val ** 2
            
            total_err = err + zero_err
            
            if total_err < best_err:
                best_err = total_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if total_err < tol:
                break
            
            # Gradient-based update (simplified)
            # For each r, we want to minimize the error
            # This is complex, so use a simple perturbation approach
            
            # Perturb one term at a time
            r = it % rank
            lr = 0.01 / (1 + it * 0.001)
            
            # Compute gradient-like direction for U[:, r]
            # dE/dU[a,r] = sum over (b,c) of error_term * V[b,r] * W[c,r]
            # This is expensive, so use a simpler approach
            
            # Try small random perturbations
            for coord in range(min(5, n * m)):
                a = rng2.randint(n * m)
                U[a, r] += lr * rng2.randn()
            
            # Renormalize
            norm = np.sqrt(np.sum(U[:, r]**2) + np.sum(V[:, r]**2) + np.sum(W[:, r]**2))
            if norm > 1e-10:
                U[:, r] /= norm
                V[:, r] /= norm
                W[:, r] /= norm
        
        if best_err < tol:
            # Verify
            if _verify(best_U, best_V, best_W, n, m, p, tol):
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 100
    
    # First, return schoolbook as fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Known lower bounds suggest rank is at least ~75-80 for 4x5x5
    # Try ranks from 80 down to 70
    for rank in range(80, 69, -1):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        result = _try_rank(rank, n, m, p, max_iters=100, num_trials=2, tol=1e-6)
        if result is not None:
            return result
    
    # Return schoolbook fallback
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
