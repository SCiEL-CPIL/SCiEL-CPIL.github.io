"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from itertools import product

def _construct_rank24():
    """Trivial rank-24 decomposition for 2x3x4 matrix multiplication."""
    n, m, p = 2, 3, 4
    rank = n * m * p  # 24
    U = [[0] * rank for _ in range(n * m)]  # 6 x 24
    V = [[0] * rank for _ in range(m * p)]  # 12 x 24
    W = [[0] * rank for _ in range(p * n)]  # 8 x 24
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=3, p=4, tol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Precompute tensor indices
    # T[i*m+j, j*p+k, k*n+i] = 1
    # All other entries are 0.
    
    # We can check by reconstructing the tensor and comparing.
    # But full tensor is 6x12x8 = 576 entries.
    # We can iterate over all alpha, beta, gamma.
    
    max_err = 0.0
    for alpha in range(n * m):
        i = alpha // m
        j = alpha % m
        
        for beta in range(m * p):
            j_prime = beta // p
            k = beta % p
            
            for gamma in range(p * n):
                i_prime = gamma // n
                k_prime = gamma % n
                
                # Target value
                if i == i_prime and j == j_prime and k == k_prime:
                    target = 1.0
                else:
                    target = 0.0
                
                # Reconstructed value
                s = np.sum(U[alpha, :] * V[beta, :] * W[gamma, :])
                
                err = abs(s - target)
                if err > max_err:
                    max_err = err
                
                if max_err > tol:
                    return False
                    
    return True

def _als_decompose(n, m, p, rank, max_iter=500, tol=1e-12, seed=0):
    """
    Use Alternating Least Squares to find a rank-r decomposition.
    Fits the FULL tensor T.
    """
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Precompute index mappings
    # alpha = i*m + j
    # beta = j'*p + k
    # gamma = k'*n + i'
    
    # For ALS updates, we need to form the LS matrices.
    # For U[alpha, :]:
    #   Solve M_U x = t_U
    #   M_U has shape (m*p * p*n, rank) = (12*8, rank) = (96, rank)
    #   t_U has shape (96,)
    #   M_U[(beta, gamma), r] = V[beta, r] * W[gamma, r]
    #   t_U[(beta, gamma)] = T[alpha, beta, gamma]
    
    # We can precompute the structure of T.
    # T[alpha, beta, gamma] = 1 iff alpha=(i,j), beta=(j,k), gamma=(i,k)
    
    for iteration in range(max_iter):
        # Update U
        for alpha in range(n * m):
            i = alpha // m
            j = alpha % m
            
            # Build M_U and t_U
            # M_U is 96 x rank
            # We can build it efficiently
            M_U = np.zeros((m * p * p * n, rank))
            t_U = np.zeros(m * p * p * n)
            
            idx = 0
            for beta in range(m * p):
                j_prime = beta // p
                k = beta % p
                for gamma in range(p * n):
                    i_prime = gamma // n
                    k_prime = gamma % n
                    
                    M_U[idx, :] = V[beta, :] * W[gamma, :]
                    
                    if i == i_prime and j == j_prime and k == k_prime:
                        t_U[idx] = 1.0
                    else:
                        t_U[idx] = 0.0
                    idx += 1
            
            # Solve M_U x = t_U
            # x is U[alpha, :]^T
            try:
                U[alpha, :] = np.linalg.lstsq(M_U, t_U, rcond=None)[0]
            except:
                pass
        
        # Update V
        for beta in range(m * p):
            j = beta // p
            k = beta % p
            
            M_V = np.zeros((n * m * p * n, rank)) # 6*8 = 48
            t_V = np.zeros(n * m * p * n)
            
            idx = 0
            for alpha in range(n * m):
                i = alpha // m
                j_prime = alpha % m
                for gamma in range(p * n):
                    i_prime = gamma // n
                    k_prime = gamma % n
                    
                    M_V[idx, :] = U[alpha, :] * W[gamma, :]
                    
                    if i == i_prime and j_prime == j and k == k_prime:
                        t_V[idx] = 1.0
                    else:
                        t_V[idx] = 0.0
                    idx += 1
            
            try:
                V[beta, :] = np.linalg.lstsq(M_V, t_V, rcond=None)[0]
            except:
                pass
        
        # Update W
        for gamma in range(p * n):
            k = gamma // n
            i = gamma % n
            
            M_W = np.zeros((n * m * m * p, rank)) # 6*12 = 72
            t_W = np.zeros(n * m * m * p)
            
            idx = 0
            for alpha in range(n * m):
                i_prime = alpha // m
                j = alpha % m
                for beta in range(m * p):
                    j_prime = beta // p
                    k_prime = beta % p
                    
                    M_W[idx, :] = U[alpha, :] * V[beta, :]
                    
                    if i_prime == i and j == j_prime and k_prime == k:
                        t_W[idx] = 1.0
                    else:
                        t_W[idx] = 0.0
                    idx += 1
            
            try:
                W[gamma, :] = np.linalg.lstsq(M_W, t_W, rcond=None)[0]
            except:
                pass
        
        # Check convergence
        max_err = 0
        for alpha in range(n * m):
            i = alpha // m
            j = alpha % m
            for beta in range(m * p):
                j_prime = beta // p
                k = beta % p
                for gamma in range(p * n):
                    i_prime = gamma // n
                    k_prime = gamma % n
                    target = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                    s = np.sum(U[alpha, :] * V[beta, :] * W[gamma, :])
                    max_err = max(max_err, abs(s - target))
        
        if max_err < tol:
            break
    
    return U, V, W, max_err

def _to_rational_list(arr, max_denom=100):
    """Convert numpy array to list of Fraction, checking if entries are rational with small denom."""
    result = []
    for val in arr.flat:
        # Check if val is close to a rational with small denominator
        best_frac = None
        best_err = float('inf')
        for d in range(1, max_denom + 1):
            num = round(val * d)
            frac = Fraction(num, d)
            err = abs(float(frac) - val)
            if err < best_err:
                best_err = err
                best_frac = frac
            if err < 1e-10:
                break
        if best_err > 1e-8:
            return None  # Not rational enough
        result.append(best_frac)
    return result

def _construct():
    """Search for minimum rank decomposition."""
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_result = _construct_rank24()
    best_rank = n * m * p  # 24
    
    # Try ranks from low to high
    # Lower bound is likely around 18-19.
    # We try 18, 19, 20, 21, 22, 23.
    
    for rank in range(18, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        # Use fewer seeds for higher ranks if time is tight?
        # Just use 10 seeds per rank to be safe.
        for seed in range(10):
            if time.time() > deadline:
                break
            
            U, V, W, err = _als_decompose(n, m, p, rank, max_iter=100, tol=1e-10, seed=seed)
            
            if err < 1e-9:
                # Try to convert to rationals
                U_rat = _to_rational_list(U, max_denom=20)
                V_rat = _to_rational_list(V, max_denom=20)
                W_rat = _to_rational_list(W, max_denom=20)
                
                if U_rat is not None and V_rat is not None and W_rat is not None:
                    # Verify with exact arithmetic (using floats for speed, but strict tol)
                    U_list = [[float(U_rat[i*rank+j]) for j in range(rank)] for i in range(n*m)]
                    V_list = [[float(V_rat[i*rank+j]) for j in range(rank)] for i in range(m*p)]
                    W_list = [[float(W_rat[i*rank+j]) for j in range(rank)] for i in range(p*n)]
                    
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
        
        if found:
            break
    
    return best_result

def construct():
    return _construct()
# EVOLVE-BLOCK-END
