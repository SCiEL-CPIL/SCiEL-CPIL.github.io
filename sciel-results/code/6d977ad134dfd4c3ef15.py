"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, tol=1e-12):
    """Verify that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                expected = 1.0
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using integers (assuming U,V,W are integers or half-integers).
    We multiply W by 2 to clear half-integers if needed, but simpler: 
    Check if 2 * sum(U*V*W) is even integer 2? 
    Actually, just check if sum(U*V*W) is exactly 1.0 using fractions or high precision.
    Here we assume U,V,W are lists of floats that are exactly representable.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-10):
                    return False
    return True

def _is_half_integer(val, tol=1e-9):
    return abs(val * 2 - round(val * 2)) < tol

def _try_rank_r(n, m, p, r, max_attempts=1000, time_limit=None):
    """Try to find a decomposition of rank r."""
    start_time = time.time()
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    num_eq = n * m * p
    
    # Precompute indices for the linear system
    # Equation for (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is linear in W.
    # M[eq_idx, W_idx] = U[i*m+j, c] * V[j*p+k, c]
    # where W_idx = k*n+i + c * W_dim? No, W is (W_dim, r).
    # Let's flatten W to vector of length W_dim * r.
    # W[k*n+i, c] is at index (k*n+i)*r + c.
    
    # We will generate random U, V and solve for W.
    
    for attempt in range(max_attempts):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        # Generate random U, V with entries -1, 0, 1
        U = np.random.choice([-1, 0, 1], size=(U_dim, r))
        V = np.random.choice([-1, 0, 1], size=(V_dim, r))
        
        # Construct M (num_eq x W_dim*r)
        M = np.zeros((num_eq, W_dim * r))
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Coefficients for W[k*n+i, c]
                    # Coeff = U[i*m+j, c] * V[j*p+k, c]
                    coeff = U[i*m+j, :] * V[j*p+k, :]
                    # W indices for this (i, k) are (k*n+i)*r + c
                    w_base = (k*n+i) * r
                    M[eq_idx, w_base:w_base+r] = coeff
                    eq_idx += 1
        
        b = np.ones(num_eq)
        
        # Solve M W_flat = b
        try:
            W_flat, residuals, rank_m, s = np.linalg.lstsq(M, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Check residual
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
            
        # Reshape W_flat to (W_dim, r)
        W = W_flat.reshape(W_dim, r)
        
        # Check if W is integer or half-integer
        valid = True
        for val in W:
            if not _is_half_integer(val):
                valid = False
                break
        if not valid:
            continue
            
        # Convert to list of lists for return
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Exact verification
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
            return U_list, V_list, W_list
            
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Try to find a lower rank decomposition
    # Start from a reasonable lower bound. Rank 1 is impossible.
    # Let's try r from 20 to 44.
    # If we find one, we return it.
    # We prioritize smaller r.
    
    # To save time, we can skip very small r where it's obviously impossible?
    # But we don't know the exact lower bound.
    # Let's try r=30 to 44 first, as 3x3x3 is 23, so 3x3x5 should be >= 23.
    # Actually, let's just iterate from 25 to 44.
    
    best_result = None
    for r in range(25, 45):
        if time.time() > deadline:
            break
            
        # Allocate more time for larger r? No, solve is fast.
        # Just do a fixed number of attempts.
        # If r is close to 45, it's easier.
        # If r is small, it's harder.
        
        attempts = 500
        if r > 40:
            attempts = 200
            
        result = _try_rank_r(n, m, p, r, max_attempts=attempts, time_limit=deadline - time.time())
        if result:
            best_result = result
            break # Found the smallest r in this range
            
    if best_result:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

