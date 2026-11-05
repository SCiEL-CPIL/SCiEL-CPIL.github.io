"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from fractions import Fraction
import itertools

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Schoolbook fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify function: checks exact decomposition
    def verify(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], np.multiply(V[j*p+k], W[k*n+i]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Check if entries are integer or half-integer
    def is_half_integer(arr, tol=1e-9):
        arr = np.array(arr, dtype=float)
        doubled = arr * 2
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=tol)
    
    # Strategy: For a fixed rank, fix U and V, then for each (k,i) pair,
    # solve the linear system M_{k,i} @ w = ones(m) where M_{k,i}[j,c] = U[i*m+j,c]*V[j*p+k,c]
    # We need w to be half-integer.
    
    # We'll try ranks from best_rank-1 down to 50.
    # For each rank, we'll try several random U, V pairs.
    
    # To increase the chance of finding half-integer solutions, we can:
    # 1. Use structured U, V (e.g., block diagonal, or with many zeros)
    # 2. Use small integer values
    
    for target_rank in range(best_rank - 1, 49, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random attempts for this rank
        num_attempts = 50
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Generate random U and V with half-integer entries
            # Use values in {-1, -0.5, 0, 0.5, 1} to keep things simple
            # Or use integers {-2, -1, 0, 1, 2}
            
            # Let's use a mix: some zeros, some ±1
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            for c in range(target_rank):
                # For each column, assign random values
                # Make sure the column is not all zero
                U[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=n*m)
                V[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=m*p)
                
                # Ensure not all zero
                if np.all(U[:, c] == 0):
                    U[np.random.randint(n*m), c] = 1
                if np.all(V[:, c] == 0):
                    V[np.random.randint(m*p), c] = 1
            
            W = np.zeros((p * n, target_rank))
            valid_W = True
            
            for k in range(p):
                for i in range(n):
                    # Build M_i: shape (m, target_rank)
                    M_i = np.zeros((m, target_rank))
                    for j in range(m):
                        for c in range(target_rank):
                            M_i[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    
                    # Solve M_i @ w = ones(m)
                    # We want a half-integer solution.
                    # Find the minimum norm solution first
                    try:
                        w_min, residuals, rank_sol, sv = np.linalg.lstsq(M_i, np.ones(m), rcond=None)
                        
                        # Check if w_min is close to half-integer
                        w_rounded = np.round(w_min * 2) / 2
                        check = M_i @ w_rounded
                        if np.allclose(check, np.ones(m), atol=1e-9):
                            W[k*n + i, :] = w_rounded
                        else:
                            # Try to find a half-integer solution in the affine space
                            # w = w_min + z where z is in null space of M_i
                            # We need M_i @ z = 0 and w_min + z to be half-integer
                            # This is hard to solve directly.
                            # Alternative: try random half-integer vectors and check
                            # But that's expensive.
                            
                            # Try a few random perturbations in the null space
                            if rank_sol < target_rank:
                                # Compute null space basis
                                # Use SVD
                                U_svd, s, Vt = np.linalg.svd(M_i)
                                # Null space is the last (target_rank - rank_sol) columns of Vt
                                null_space = Vt[rank_sol:, :].T  # shape (target_rank, target_rank - rank_sol)
                                
                                # Try a few random combinations
                                found = False
                                for _ in range(20):
                                    # Random coefficients for null space
                                    if target_rank - rank_sol > 0:
                                        coeffs = np.random.choice([-1, -0.5, 0, 0.5, 1], size=target_rank - rank_sol)
                                    else:
                                        coeffs = np.array([])
                                    z = null_space @ coeffs
                                    w_candidate = w_min + z
                                    w_rounded2 = np.round(w_candidate * 2) / 2
                                    check2 = M_i @ w_rounded2
                                    if np.allclose(check2, np.ones(m), atol=1e-9):
                                        W[k*n + i, :] = w_rounded2
                                        found = True
                                        break
                                
                                if not found:
                                    valid_W = False
                                    break
                            else:
                                valid_W = False
                                break
                    except:
                        valid_W = False
                        break
                
                if not valid_W:
                    break
            
            if valid_W:
                # Final verification
                if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    # Check half-integer constraint
                    if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break  # Found valid for this rank, try lower
    
    # If we haven't found anything better, try a more structured approach
    # for a few specific ranks
    if best_rank == n * m * p:
        # Try rank 64, 60, 56 with more attempts
        for target_rank in [64, 62, 60, 58, 56]:
            if time.time() > deadline:
                break
            
            num_attempts = 100
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    break
                
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                
                for c in range(target_rank):
                    U[:, c] = np.random.choice([-1, 0, 1], size=n*m)
                    V[:, c] = np.random.choice([-1, 0, 1], size=m*p)
                    
                    if np.all(U[:, c] == 0):
                        U[np.random.randint(n*m), c] = 1
                    if np.all(V[:, c] == 0):
                        V[np.random.randint(m*p), c] = 1
                
                W = np.zeros((p * n, target_rank))
                valid_W = True
                
                for k in range(p):
                    for i in range(n):
                        M_i = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                M_i[j, c] = U[i*m + j, c] * V[j*p + k, c]
                        
                        try:
                            w_min, residuals, rank_sol, sv = np.linalg.lstsq(M_i, np.ones(m), rcond=None)
                            w_rounded = np.round(w_min * 2) / 2
                            check = M_i @ w_rounded
                            if np.allclose(check, np.ones(m), atol=1e-9):
                                W[k*n + i, :] = w_rounded
                            else:
                                valid_W = False
                                break
                        except:
                            valid_W = False
                            break
                    
                    if not valid_W:
                        break
                
                if valid_W:
                    if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                        if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                            best_rank = target_rank
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            break
    
    return best_result
# EVOLVE-BLOCK-END
