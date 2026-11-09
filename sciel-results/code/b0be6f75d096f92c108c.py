"""Rewritten program for matrix multiplication tensor rank minimization (3x4, 4x7)."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = time.time() + 140  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _als_decompose(n, m, p, rank, max_iter=500, seed=0, tol=1e-10):
    """
    ALS (Alternating Least Squares) to find exact tensor decomposition.
    Returns (U, V, W) as numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank)
    or None if not converged.
    """
    np.random.seed(seed)
    
    # Initialize randomly
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    # Target tensor T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            return None
        
        # Update U: for each (i,j), solve linear system
        # For fixed (i,j), the equation over k is:
        # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
        # This is a p x rank linear system: M[i,j] @ U[i*m+j, :] = ones(p)
        
        # Build all systems at once using vectorization
        # For each (i,j), M[i,j] is a p x rank matrix where M[i,j][k, r] = V[j*p+k, r] * W[k*n+i, r]
        
        # Let's do it iteratively for clarity
        for i in range(n):
            for j in range(m):
                # Build p x rank matrix
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                
                # Solve M @ u = ones(p)
                try:
                    u_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                except np.linalg.LinAlgError:
                    return None
                U[i*m+j, :] = u_sol
        
        # Update V: for each (j,k), solve
        # For fixed (j,k), equation over i:
        # sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                except np.linalg.LinAlgError:
                    return None
                V[j*p+k, :] = v_sol
        
        # Update W: for each (k,i), solve
        # For fixed (k,i), equation over j:
        # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    return None
                W[k*n+i, :] = w_sol
        
        # Check convergence
        # Compute max error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < tol:
            return U, V, W
    
    # Final check
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err < 1e-6:
        return U, V, W
    
    return None

def _round_to_half_int(x, tol=1e-6):
    """Try to round x to nearest half-integer if close enough."""
    # Check if x is close to an integer or half-integer
    nearest_half = np.round(x * 2) / 2
    if abs(x - nearest_half) < tol:
        return float(nearest_half)
    return float(x)

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank = n*m*p = 84)
    schoolbook_rank = n * m * p
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    min_possible = max(n*m, m*p, p*n)
    
    # Try ranks from low to high
    for target_rank in range(min_possible, schoolbook_rank + 1):
        if time.time() > DEADLINE - 15:
            break
        
        found = False
        # Try multiple seeds
        for seed in range(100):
            if time.time() > DEADLINE - 10:
                break
            
            result = _als_decompose(n, m, p, target_rank, max_iter=300, seed=seed)
            if result is not None:
                U_np, V_np, W_np = result
                
                # Convert to list-of-lists with exact values
                U_list = [[float(U_np[i, r]) for r in range(target_rank)] for i in range(n*m)]
                V_list = [[float(V_np[i, r]) for r in range(target_rank)] for i in range(m*p)]
                W_list = [[float(W_np[i, r]) for r in range(target_rank)] for i in range(p*n)]
                
                # Verify
                if _verify(U_list, V_list, W_list, n, m, p):
                    if target_rank < best_rank:
                        best_U = U_np
                        best_V = V_np
                        best_W = W_np
                        best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Found a valid decomposition at this rank; try to go lower
            continue
    
    # Convert best result to list-of-lists
    U_final = [[float(best_U[i, r]) for r in range(best_rank)] for i in range(n*m)]
    V_final = [[float(best_V[i, r]) for r in range(best_rank)] for i in range(m*p)]
    W_final = [[float(best_W[i, r]) for r in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
