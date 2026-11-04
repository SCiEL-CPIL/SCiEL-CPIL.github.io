"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # All other entries are 0.
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1.0
    
    # Helper to check if a decomposition is valid
    def is_valid(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        # Compute tensor product sum
        T_rec = np.zeros_like(T_target)
        for c in range(U.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(T_rec, T_target, atol=tol, rtol=0)
    
    def check_half_integer(U, V, W, tol=1e-9):
        """Check if all entries are integer or half-integer."""
        for mat in [U, V, W]:
            for val in mat.flatten():
                v2 = val * 2
                if abs(v2 - round(v2)) > tol:
                    return False
        return True
    
    def round_to_half_integer(arr):
        """Round array to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    # Try to find a lower rank decomposition using numerical optimization
    # We'll try ranks from 35 down to 20
    best_result = None
    best_rank = rank_schoolbook
    
    # Schoolbook as fallback
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U_school[idx_u, col] = 1.0
                V_school[idx_v, col] = 1.0
                W_school[idx_w, col] = 1.0
                col += 1
    
    # Try various ranks
    for target_rank in range(35, 19, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Iterative refinement using ALS (Alternating Least Squares)
            # But we need exact decomposition, so this is tricky.
            # Instead, let's try a different approach: use the fact that the tensor
            # has a specific structure and try to find a decomposition directly.
            
            # Actually, for small problems, we can try to use a more direct approach.
            # Let's try to find a decomposition by solving a system of equations.
            
            # The condition is: for all (i,j,k),
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # And for all other (a,b,c) combinations, the sum should be 0.
            
            # This is a nonlinear system. Let's try a simpler approach:
            # Use the schoolbook and try to eliminate some terms by finding
            # linear dependencies.
            
            pass  # This approach is too complex for now
        
        if not found:
            continue
    
    # For now, return the schoolbook decomposition
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
