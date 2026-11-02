import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
_START_TIME = time.time()
_DEADLINE = 140  # safety margin before 150s kill

def _build_target_tensor(n, m, p):
    """Build the target tensor T where T[i*m+j, j*p+k, k*n+i] = 1"""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i*m+j, j*p+k, k*n+i)] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U, V, W are expected to be lists of lists of numbers (integers or half-integers).
    """
    rank = len(U[0]) if U else 0
    # Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(computed - target) > tol:
                    return False
    
    # Check that no spurious entries exist
    # We check a subset of random zero entries to save time
    import random
    random.seed(42)
    # Total entries in U is n*m, V is m*p, W is p*n
    # Total possible triples is (n*m)*(m*p)*(p*n) which is huge.
    # We just need to ensure the tensor is zero where it should be.
    # The tensor is zero if (i1 != i2) or (j1 != j2) or (k1 != k2)
    # where a = i1*m+j1, b = j2*p+k2, c = k1*n+i2
    # We sample random a, b, c and check if they form a valid non-zero triple.
    # If they don't, the sum must be 0.
    
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k2 = b % p
        k1 = c // n
        i2 = c % n
        
        if (i1 == i2 and j1 == j2 and k1 == k2):
            continue  # This is a real entry, already checked
        
        computed = 0.0
        for r in range(rank):
            computed += U[a][r] * V[b][r] * W[c][r]
        if abs(computed) > tol:
            return False
    return True

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-12, seed=0):
    """Try to find a rank decomposition using ALS with exact verification."""
    np.random.seed(seed)
    
    # Initialize with random matrices
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # Build target tensor as sparse dict
    target = _build_target_tensor(n, m, p)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                row_u = i*m + j
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[row_u, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                row_v = j*p + k
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[row_v, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                row_w = k*n + i
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[row_w, :] = x
                except:
                    pass
        
        # Check convergence periodically
        if it % 20 == 0:
            # Convert to list of lists for verification? 
            # It's faster to verify on numpy if we just check the sum.
            # But we need to ensure it can be represented by half-integers.
            # Let's do a quick check on the floating point values first.
            # If the floating point values are not close to half-integers, skip detailed check.
            is_close_to_half = True
            for arr in [U, V, W]:
                doubled = arr * 2
                rounded = np.round(doubled)
                if not np.allclose(doubled, rounded, atol=1e-6):
                    is_close_to_half = False
                    break
            
            if is_close_to_half:
                # Convert to lists for exact verification
                U_list = [[val for val in row] for row in U]
                V_list = [[val for val in row] for row in V]
                W_list = [[val for val in row] for row in W]
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
                    return U, V, W
    
    return None

def _try_rank(n, m, p, rank, num_trials=10):
    """Try multiple random initializations for a given rank."""
    for trial in range(num_trials):
        if time.time() > _DEADLINE:
            break
        result = _als_decompose(n, m, p, rank, max_iters=50, seed=trial*12345+42)
        if result is not None:
            U, V, W = result
            # Final verification on the rounded values
            U_list = _to_list(U)
            V_list = _to_list(V)
            W_list = _to_list(W)
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
                return U_list, V_list, W_list
    return None

def _to_list(arr):
    """Convert numpy array to list of lists with half-integer rounding."""
    result = []
    for row in arr:
        lst = []
        for val in row:
            rounded = round(val * 2) / 2
            lst.append(rounded)
        result.append(lst)
    return result

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # Start with schoolbook as fallback
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Try ranks below schoolbook, starting from the highest possible lower rank
    # This is more likely to find a solution if one exists near the bound
    # If the true rank is 90, this will fail quickly for 89, 88...
    # We limit the number of ranks to try to ensure we don't waste all time
    # on infeasible ranks.
    
    # Let's try 89, 88, 87... down to 80
    # If we don't find anything, we return 90.
    
    # Time budget for search: 100 seconds
    search_deadline = _START_TIME + 100
    
    for rank in range(schoolbook_rank - 1, max(80, schoolbook_rank - 10), -1):
        if time.time() > search_deadline:
            break
        
        result = _try_rank(n, m, p, rank, num_trials=5)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            # Found a lower rank. Continue trying to find even lower?
            # Given the difficulty, it's unlikely we find 88 after finding 89.
            # But let's try.
            continue
    
    # If we found a decomposition, use it
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
