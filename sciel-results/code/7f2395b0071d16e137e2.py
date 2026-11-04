"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_result = None
    
    # Helper to create schoolbook decomposition
    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Helper to verify decomposition
    def verify(U, V, W, n, m, p, tol=1e-6):
        rank = U.shape[1]
        # Check that U^T * tensor * V^T = W^T for all entries
        # The multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # and 0 otherwise
        
        # Build the multiplication tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        
        # Check decomposition
        # For each (a, b) pair, compute sum_r U[a,r] * V[b,r] and check against T
        # Actually, we need to check: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == T[i*m+j, j*p+k, k*n+i]
        
        # Efficient check: for each r, compute outer product contribution
        # T_approx[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can check this by computing for each (a,b) the vector sum_r U[a,r]*V[b,r]*W[:,r]
        # and comparing to T[a,b,:]
        
        # More efficient: for each r, the contribution to T is U[:,r] * V[:,r] * W[:,r] (outer product)
        # T_approx = sum_r np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        
        # Check a sample of entries for speed
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # Compute sum_r U[a,r] * V[b,r] * W[c,r]
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Also check that entries that should be 0 are close to 0
        # Sample some zero entries
        for _ in range(100):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            # Check if this should be zero
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            if not (j == j2 and i == i2 and k == k2):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        
        return True
    
    # Initialize with schoolbook
    U, V, W, rank = make_schoolbook(n, m, p)
    best_result = (U.copy(), V.copy(), W.copy())
    best_rank = rank
    
    # Try iterative refinement to reduce rank
    # Start from schoolbook and try to zero out columns
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Try to reduce rank by attempting to set random columns to zero
        # and see if we can adjust other columns to compensate
        
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_rank = current_U.shape[1]
        
        if current_rank <= best_rank:
            break
        
        # Try to find a column that can be eliminated
        # A column r can be eliminated if there exist adjustments to other columns
        # such that the tensor decomposition is preserved
        
        # Simple approach: try setting each column to zero and check if the
        # resulting "hole" can be filled by adjusting other columns
        
        eliminated = False
        for r in range(current_rank):
            if time.time() > deadline:
                break
            
            # Try to eliminate column r
            # Compute the tensor contribution of column r
            # T_r[a,b,c] = U[a,r] * V[b,r] * W[c,r]
            
            # Check if this can be represented as a combination of other columns
            # This is a complex optimization problem, so use a heuristic:
            # Try to find small adjustments to other columns that compensate
            
            # For now, just check if the column has small entries
            col_norm = np.sqrt(np.sum(current_U[:, r]**2) * np.sum(current_V[:, r]**2) * np.sum(current_W[:, r]**2))
            
            if col_norm < 0.1:
                # Try to eliminate this column
                new_U = np.delete(current_U, r, axis=1)
                new_V = np.delete(current_V, r, axis=1)
                new_W = np.delete(current_W, r, axis=1)
                
                # Verify
                if verify(new_U, new_V, new_W, n, m, p, tol=1e-4):
                    U, V, W = new_U, new_V, new_W
                    best_rank = U.shape[1]
                    best_result = (U.copy(), V.copy(), W.copy())
                    eliminated = True
                    break
        
        if not eliminated and current_rank == best_rank:
            # No progress, try a different approach
            # Random perturbation and re-optimization
            if attempt % 5 == 4:
                # Try a fresh random initialization with lower rank
                target_rank = max(1, best_rank - 1)
                if target_rank < best_rank:
                    np.random.seed(attempt * 1000)
                    U_rand = np.random.randn(n * m, target_rank) * 0.1
                    V_rand = np.random.randn(m * p, target_rank) * 0.1
                    W_rand = np.random.randn(p * n, target_rank) * 0.1
                    
                    # Simple gradient-free optimization: try to minimize error
                    # For each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                    # and for other (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = 0
                    
                    # This is hard to optimize directly, so skip for now
                    pass
    
    # Convert to list of lists with integer/half-integer entries
    def to_fraction_list(matrix):
        result = []
        for row in matrix:
            frac_row = []
            for val in row:
                # Try to find a simple fraction
                if abs(val) < 1e-10:
                    frac_row.append(0)
                else:
                    # Try denominators 1, 2, 4, 8
                    for denom in [1, 2, 4, 8]:
                        numer = round(val * denom)
                        if abs(val - numer / denom) < 1e-6:
                            frac_row.append(numer / denom)
                            break
                    else:
                        frac_row.append(val)
            result.append(frac_row)
        return result
    
    U_final, V_final, W_final = best_result
    return {"U": to_fraction_list(U_final), "V": to_fraction_list(V_final), "W": to_fraction_list(W_final)}
# EVOLVE-BLOCK-END
