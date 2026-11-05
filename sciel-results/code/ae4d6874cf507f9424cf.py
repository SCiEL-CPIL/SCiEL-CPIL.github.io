"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 115s to be safe within 150s
    
    # Index mappings
    # U index: i*m + j, where i in [0,n), j in [0,m)
    # V index: j*p + k, where j in [0,m), k in [0,p)
    # W index: k*n + i, where k in [0,p), i in [0,n)
    
    # The tensor equation: for all (i,j,k) in [0,n)x[0,m)x[0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    best_rank = n * m * p  # schoolbook = 84
    best_solution = None
    
    # Generate schoolbook solution as baseline
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    for col in range(best_rank):
        i = col // (m*p)
        j = (col // p) % m
        k = col % p
        U_sb[i*m+j, col] = 1.0
        V_sb[j*p+k, col] = 1.0
        W_sb[k*n+i, col] = 1.0
    best_solution = (U_sb, V_sb, W_sb)
    
    def check_solution(U, V, W, tol=1e-6):
        """Check if the solution is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_rational_matrix(mat, max_denom=2):
        """Convert a matrix to rationals with small denominators."""
        result = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                # Try to find best rational with small denominator
                best_frac = Fraction(0)
                best_err = float('inf')
                for denom in range(1, max_denom + 1):
                    for numer in range(int(-2*denom), int(2*denom) + 1):
                        frac = Fraction(numer, denom)
                        err = abs(float(frac) - val)
                        if err < best_err:
                            best_err = err
                            best_frac = frac
                if best_err < 1e-4:
                    result[i, j] = float(best_frac)
                else:
                    result[i, j] = val
        return result
    
    def is_valid_half_integer(arr, tol=1e-9):
        """Check if all entries are integers or half-integers."""
        for val in arr.flatten():
            if abs(val * 2 - round(val * 2)) > tol:
                return False
        return True
    
    def als_decompose(rank, max_iter=100, tol=1e-8):
        """ALS decomposition with random initialization."""
        # Build the tensor T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
        # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Initialize with small random values
        np.random.seed(rank)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # Build system: U[i*m+j, :] * (V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    # Least squares
                    U[i*m+j, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    V[j*p+k, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    W[k*n+i, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0)**2
            err = np.sqrt(err / (n*m*p))
            if err < tol:
                break
        
        return U, V, W, err
    
    # Try to find lower rank solutions
    # Known lower bound: rank >= n*m*p / max(n,m,p) = 84/7 = 12
    # But practical lower bounds are higher. Try from reasonable values.
    
    search_time_limit = 90.0
    start_time = time.time()
    
    # Try ranks in decreasing order from schoolbook - 1 down to a lower bound
    # Focus on ranks that are plausible: multiples of small numbers
    ranks_to_try = []
    for r in range(42, best_rank):  # Start from 42 (half of schoolbook)
        ranks_to_try.append(r)
    
    # Also try some specific ranks
    ranks_to_try = sorted(set(ranks_to_try + [36, 40, 44, 48, 52, 56, 60]))
    
    found_better = False
    for rank in ranks_to_try:
        if time.time() - start_time > search_time_limit:
            break
        if rank >= best_rank:
            continue
        
        # Multiple restarts for each rank
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(rank * 100 + restart)
            U, V, W, err = als_decompose(rank, max_iter=50, tol=1e-7)
            
            if err < 1e-5:
                # Try to round to half-integers
                U_r = to_rational_matrix(U, max_denom=2)
                V_r = to_rational_matrix(V, max_denom=2)
                W_r = to_rational_matrix(W, max_denom=2)
                
                if check_solution(U_r, V_r, W_r, tol=1e-4):
                    best_rank = rank
                    best_solution = (U_r, V_r, W_r)
                    found_better = True
                    break
                else:
                    # Try with smaller tolerance
                    if check_solution(U, V, W, tol=1e-4):
                        # Keep the non-rounded version if it's close enough
                        # But we need half-integers, so try harder rounding
                        U_r2 = to_rational_matrix(U, max_denom=4)
                        V_r2 = to_rational_matrix(V, max_denom=4)
                        W_r2 = to_rational_matrix(W, max_denom=4)
                        if check_solution(U_r2, V_r2, W_r2, tol=1e-4):
                            best_rank = rank
                            best_solution = (U_r2, V_r2, W_r2)
                            found_better = True
                            break
        
        if found_better:
            # Try one more rank down
            if best_rank > 30:
                continue
            else:
                break
    
    U, V, W = best_solution
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
