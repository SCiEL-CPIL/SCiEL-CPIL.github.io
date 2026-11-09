"""Matrix multiplication 3,3,6 - rank-reduced decomposition."""
DIMENSIONS = [3, 3, 6]

import time
import random
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _get_3x3x3_targets():
    """Return list of (i, j, k) indices for the 3x3x3 multiplication tensor.
    The tensor T has T[i*3+j, j*3+k, k*3+i] = 1 for i,j,k in {0,1,2}.
    """
    targets = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                targets.append((i * 3 + j, j * 3 + k, k * 3 + i))
    return targets

def _verify_3x3x3(U, V, W, tol=1e-9):
    """Verify rank-r decomposition for 3x3x3 mat mult."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    targets = _get_3x3x3_targets()
    for a, b, c in targets:
        s = np.sum(Ua[a, :] * Va[b, :] * Wa[c, :])
        if abs(s - 1.0) > tol:
            return False
    # Check that other entries are 0? 
    # The problem statement says "exactly decompose the matrix multiplication tensor".
    # This implies T[a,b,c] must be 0 for all other (a,b,c).
    # However, for mat mult tensors, if the 27 non-zeros are satisfied and the rank is minimal, 
    # the zeros are often automatically satisfied if the structure is correct? 
    # No, we must ensure zeros.
    # But checking all 9*9*9 = 729 entries is fast.
    for a in range(9):
        for b in range(9):
            for c in range(9):
                if (a, b, c) in targets:
                    continue
                s = np.sum(Ua[a, :] * Va[b, :] * Wa[c, :])
                if abs(s) > tol:
                    return False
    return True

def _find_rank_3x3x3(target_rank, time_limit=10.0):
    """Attempt to find a decomposition of rank target_rank for 3x3x3 mat mult."""
    start = time.time()
    targets = _get_3x3x3_targets()
    target_set = set(targets)
    
    # Objective function for optimization
    def objective(params):
        if time.time() - start > time_limit:
            return 1e10
        U = params[0:9*target_rank].reshape(9, target_rank)
        V = params[9*target_rank:18*target_rank].reshape(9, target_rank)
        W = params[18*target_rank:27*target_rank].reshape(9, target_rank)
        
        err = 0.0
        # Check non-zeros
        for a, b, c in targets:
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            err += (s - 1.0)**2
        # Check zeros (sample or all? All is 729 ops, cheap)
        for a in range(9):
            for b in range(9):
                for c in range(9):
                    if (a, b, c) in target_set:
                        continue
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err += s**2
        return err

    # Try a few random initializations
    for _ in range(5):
        if time.time() - start > time_limit:
            break
        
        # Initialize with small random values
        params0 = np.random.randn(27 * target_rank) * 0.1
        
        res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 500, 'disp': False})
        
        if res.fun < 1e-8:
            # Found a solution!
            U = res.x[0:9*target_rank].reshape(9, target_rank)
            V = res.x[9*target_rank:18*target_rank].reshape(9, target_rank)
            W = res.x[18*target_rank:27*target_rank].reshape(9, target_rank)
            
            # Round to nearest integers or half-integers to get exact solution
            # The problem allows integer/half-integer entries.
            # Let's try rounding to nearest integer first.
            U_int = np.round(U).astype(int)
            V_int = np.round(V).astype(int)
            W_int = np.round(W).astype(int)
            
            if _verify_3x3x3(U_int, V_int, W_int):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
            # Try half-integers (multiply by 2, round, divide by 2)
            U_half = np.round(U * 2) / 2.0
            V_half = np.round(V * 2) / 2.0
            W_half = np.round(W * 2) / 2.0
            
            if _verify_3x3x3(U_half, V_half, W_half):
                # Convert to list of floats
                return U_half.tolist(), V_half.tolist(), W_half.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Strategy: Find rank-r decomposition for 3x3x3, use it twice for 3x3x6.
    # Total rank = 2r.
    # Schoolbook 3x3x3 is rank 27 -> Total 54.
    # Target: Find r < 27.
    
    best_rank = 27
    best_factors = None
    
    # Try ranks 20, 21, 22, 23, 24, 25, 26
    for r in range(20, 27):
        if time.time() > deadline - 20: # Leave time for construction and verification
            break
            
        factors = _find_rank_3x3x3(r, time_limit=10.0)
        if factors is not None:
            U3, V3, W3 = factors
            best_rank = r
            best_factors = (U3, V3, W3)
            break # Found the lowest rank in the search order
    
    if best_factors is not None:
        U3, V3, W3 = best_factors
        r3 = best_rank
        rank = 2 * r3
        
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # First half: B columns 0,1,2 (k=0,1,2)
        # U[i*m+j, c] = U3[i*3+j, c]
        # V[j*3+k, c] = V3[j*3+k, c] for k=0,1,2
        # W[k*3+i, c] = W3[k*3+i, c] for k=0,1,2
        
        for i in range(n):
            for j in range(m):
                for c in range(r3):
                    U[i*m+j][c] = U3[i*3+j][c]
        
        for j in range(m):
            for k in range(3): # k is 0,1,2
                for c in range(r3):
                    V[j*p+k][c] = V3[j*3+k][c]
        
        for k in range(3): # k is 0,1,2
            for i in range(n):
                for c in range(r3):
                    W[k*n+i][c] = W3[k*3+i][c]
        
        # Second half: B columns 3,4,5 (k=3,4,5)
        # Map k in {3,4,5} to k' in {0,1,2} for the 3x3x3 decomposition
        # U[i*m+j, c+r3] = U3[i*3+j, c]
        # V[j*3+k, c+r3] = V3[j*3+(k-3), c]
        # W[k*n+i, c+r3] = W3[(k-3)*3+i, c]
        
        for i in range(n):
            for j in range(m):
                for c in range(r3):
                    U[i*m+j][c+r3] = U3[i*3+j][c]
        
        for j in range(m):
            for k in range(3, 6): # k is 3,4,5
                k_prime = k - 3
                for c in range(r3):
                    V[j*p+k][c+r3] = V3[j*3+k_prime][c]
        
        for k in range(3, 6): # k is 3,4,5
            k_prime = k - 3
            for i in range(n):
                for c in range(r3):
                    W[k*n+i][c+r3] = W3[k_prime*3+i][c]
        
        # Verify the full 3x3x6 decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Verification failed, fall back to schoolbook
            pass

    # Fallback: Schoolbook for 3x3x6, rank 54
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
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 if (i,j,k) is a valid mat mult term, else 0."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    
    # Build target set
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_set.add((i*m+j, j*p+k, k*n+i))
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                s = np.sum(Ua[a, :] * Va[b, :] * Wa[c, :])
                if (a, b, c) in target_set:
                    if abs(s - 1.0) > tol:
                        return False
                else:
                    if abs(s) > tol:
                        return False
    return True

# EVOLVE-BLOCK-END
