import numpy as np
import time
import itertools
from fractions import Fraction
import random

DIMENSIONS = [2, 3, 7]

def check_solution(U, V, W, n, m, p):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Convention: W[k*n+i, r]
    """
    R = U.shape[1]
    # Convert to Fraction for exactness if inputs are floats? 
    # The task says "integer/half-integer entries".
    # We will assume inputs are lists of numbers that can be exactly represented.
    # For verification, we use the property:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0.0
                for r in range(R):
                    s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Fallback: Schoolbook Solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_solution = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # 2. Strategy: Search for low-rank decompositions using Finite Field Support Patterns
    # We look for R < rank_school.
    # The tensor T_{ijk} = 1.
    # We want T = sum_r U_{ij}^r V_{jk}^r W_{ki}^r.
    
    # Since n=2, m=3, p=7, the dimensions of U, V, W are:
    # U: 6 x R
    # V: 21 x R
    # W: 14 x R
    
    # We will iterate R from a lower bound up to rank_school - 1.
    # Lower bound for matrix multiplication is max(nm, mp, np) = max(6, 21, 14) = 21.
    # Actually, a better lower bound is often higher, but let's start from 21.
    
    for R in range(21, rank_school):
        if time.time() - start_time > deadline:
            break
            
        # For each R, we try to find a solution.
        # We use a heuristic search:
        # 1. Randomly initialize U and V with entries from {-1, 0, 1} (sparse).
        # 2. Solve for W.
        # 3. Check if W has "nice" entries (integers or half-integers).
        
        # Optimization: We can fix one column of U to be non-zero to break scaling symmetry?
        # Or just random search.
        
        # To speed up, we can try to satisfy the equations for a subset of (i,j,k) first.
        # But let's stick to the full system with random restarts.
        
        # Number of restarts depends on time.
        max_restarts = 2000
        restarts_done = 0
        
        for _ in range(max_restarts):
            if time.time() - start_time > deadline:
                break
            restarts_done += 1
            
            # Random sparse U, V
            # Density around 0.2
            U = np.random.choice([-1, 0, 1], size=(n * m, R), p=[0.2, 0.6, 0.2])
            V = np.random.choice([-1, 0, 1], size=(m * p, R), p=[0.2, 0.6, 0.2])
            
            # Ensure U and V are not all zeros
            if np.all(U == 0) or np.all(V == 0):
                continue
                
            # Solve for W
            # For each (k, i), we have a linear system for W[k*n+i, :]
            # Equations: for all j in 0..m-1:
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
            
            W = np.zeros((p * n, R))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A x = 1
                    try:
                        # Use least squares
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if x is close to half-integers
                    # The problem allows integer/half-integer entries.
                    x_rounded = np.round(x * 2) / 2
                    if np.allclose(x, x_rounded, atol=1e-6):
                        W[k * n + i, :] = x_rounded
                    else:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Verify full solution
                if check_solution(U, V, W, n, m, p):
                    best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = R
                    # Since we are iterating R from low to high, the first valid one is minimal.
                    return best_solution

    return best_solution

if __name__ == "__main__":
    res = construct()
    print(res)
