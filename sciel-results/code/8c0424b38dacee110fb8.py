import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Build the tensor from U, V, W
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _try_rank(r, n, m, p, max_trials=200, seed=42):
    """Try to find a rank-r decomposition using structured random search."""
    rng = np.random.default_rng(seed)
    num_eq = n * m * p
    
    for trial in range(max_trials):
        # Generate structured random U, V
        # Use values from {-1, 0, 1} but with some structure
        U_rand = rng.integers(-1, 2, size=(n * m, r))
        V_rand = rng.integers(-1, 2, size=(m * p, r))
        
        # Build the linear system A W' = 1
        # A[row, col] where row = i*m*p + j*p + k, col = k*n*r + i*r + c
        # A[row, col] = U[i*m+j, c] * V[j*p+k, c]
        A = np.zeros((num_eq, p * n * r), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i * m * p + j * p + k
                    for c in range(r):
                        col = k * n * r + i * r + c
                        A[row, col] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
        
        # Solve A W' = 1 over rationals using exact integer arithmetic
        # Use Gaussian elimination with fractions for small systems
        from fractions import Fraction
        import sys
        sys.setrecursionlimit(10000)
        
        # Convert to Fraction-based matrix for exact solving
        # This is slow but exact for small systems
        try:
            # Use sympy-like approach with fractions
            # For r up to ~25 and num_eq=36, this is feasible
            
            # Actually, let's use a different approach: 
            # Check if the system is consistent and has integer solution
            # Use SVD to check rank and then verify
            
            # Compute SVD
            U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
            rank_A = np.sum(s > 1e-10)
            
            # If rank_A < num_eq, no solution
            if rank_A < num_eq:
                continue
            
            # Solve using lstsq
            W_prime, residuals, rank_lstsq, sv = np.linalg.lstsq(A, np.ones(num_eq), rcond=None)
            
            if residuals.size > 0 and residuals[0] > 1e-8:
                continue
            
            # Check if solution is close to integers
            W_rounded = np.round(W_prime)
            if np.allclose(A @ W_rounded, np.ones(num_eq), atol=1e-8):
                # Found a valid solution!
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = (W_rounded / 2.0).tolist()
                
                # Verify
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Try ranks from 21 upwards
    for r in range(21, 37):
        if time.time() > deadline:
            break
        
        # More trials for lower ranks
        trials = 500 if r <= 23 else 200
        if time.time() > 120.0:
            trials = 50
        
        result = _try_rank(r, n, m, p, max_trials=trials, seed=r * 1000 + 42)
        if result is not None:
            return result
        
        # Try with different seed
        if time.time() < deadline - 10:
            result = _try_rank(r, n, m, p, max_trials=trials // 2, seed=r * 1000 + 123)
            if result is not None:
                return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
