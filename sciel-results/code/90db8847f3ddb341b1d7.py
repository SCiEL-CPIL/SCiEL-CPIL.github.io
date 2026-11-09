import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def _is_half_integer(x):
    """Check if a number is an integer or half-integer (multiple of 0.5)."""
    # Use Fraction for exactness if possible, but float check is faster for initial filter
    # We will do exact verification later.
    return True 

def _verify_exact(U, V, W, n, m, p):
    """
    Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    U, V, W are lists of lists of Fractions or floats.
    We assume entries are half-integers.
    """
    r = len(U[0])
    # Convert to Fractions for exact arithmetic
    U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U]
    V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V]
    W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U_frac[i*m + j][c] * V_frac[j*p + k][c] * W_frac[k*n + i][c]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, r) = (10, r)
    # V: (m*p, r) = (25, r)
    # W: (p*n, r) = (10, r)
    
    best_result = None
    
    # Strategy:
    # 1. Try structured U, V matrices.
    # 2. For a fixed U, V, the constraints on W are linear.
    #    Specifically, for each (i, k), we have a system:
    #    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j in 0..m-1
    #    Let A_{i,k} be the m x r matrix with A_{i,k}[j, c] = U[i*m+j, c] * V[j*p+k, c].
    #    We need A_{i,k} @ W_{i,k} = 1_m.
    #    If we can find a solution W_{i,k} with half-integer entries, we have a valid decomposition.
    
    # We will search for low rank r.
    # Lower bound: max(n*m, m*p, p*n) is not the bound, but rank >= max(n, m, p) is loose.
    # Actually, rank must be at least max(n, m, p) for non-trivial, but usually higher.
    # For 2x5x5, schoolbook is 50. Strassen-like might be lower.
    
    # Let's try ranks from a reasonable minimum up to 50.
    # Minimum rank for matrix mult is at least max(n, m, p) = 5.
    # But for 2x5x5, it's likely higher. Let's start from 10.
    
    # To make U and V structured, we can use a basis of "simple" vectors.
    # However, random search over U, V is hard.
    # Alternative: Fix U and V to be "block identity" or similar, and see if W can be solved.
    # But U and V must be optimized too.
    
    # Let's use a different approach:
    # We know that if we set U and V to be identity-like, the problem becomes sparse.
    # But we need to reduce rank.
    
    # Let's try a hybrid:
    # 1. Generate random U, V with small half-integer entries.
    # 2. Solve for W exactly using Fraction arithmetic.
    # 3. Check if W has half-integer entries.
    
    # To speed up, we can use floating point to filter, then exact.
    
    max_rank = 50
    min_rank = 10
    
    # We will iterate r from min_rank to max_rank
    # For each r, we try several random configurations of U, V
    
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Number of attempts decreases as r increases? Or stays same?
        # Let's do 200 attempts per rank.
        for attempt in range(200):
            if time.time() > deadline - 2:
                break
                
            # Random U, V with half-integer entries in [-2, 2]
            # Values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
            possible_vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            
            # Use np.random.choice for speed
            U = np.array([[np.random.choice(possible_vals) for _ in range(r)] for _ in range(n*m)])
            V = np.array([[np.random.choice(possible_vals) for _ in range(r)] for _ in range(m*p)])
            
            # Solve for W
            W = np.zeros((p*n, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build A_mat (m x r)
                    # A_mat[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # We can do this vectorized
                    U_block = U[i*m : (i+1)*m, :]  # m x r
                    V_block = V[k*p : (k+1)*p, :] # Wait, V index is j*p+k. 
                    # V is (m*p, r). Row index is j*p+k.
                    # For fixed k, j varies 0..m-1.
                    # So V_block[j, c] = V[j*p+k, c]
                    V_block = np.array([V[j*p+k, :] for j in range(m)]) # m x r
                    
                    A_mat = U_block * V_block # m x r
                    
                    # Solve A_mat @ w = ones(m)
                    # Use least squares
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        # Check if solution is close to half-integers
                        # And check residual
                        if np.linalg.norm(A_mat @ w - np.ones(m)) > 1e-9:
                            valid = False
                            break
                            
                        # Check if w entries are close to half-integers
                        w_half = np.round(w * 2) / 2
                        if np.linalg.norm(w - w_half) > 1e-9:
                            valid = False
                            break
                            
                        W[k*n + i, :] = w_half
                    except:
                        valid = False
                        break
                
            if not valid:
                continue
            
            # Verify exactness with Fractions
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify_exact(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: Schoolbook decomposition
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
