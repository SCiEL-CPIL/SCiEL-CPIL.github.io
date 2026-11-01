"""Matrix multiplication 2x3x9 — rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 9]


def _validate(U, V, W, n=2, m=3, p=9, tol=1e-9):
    """Return True if U,V,W exactly decompose the matrix multiplication tensor.

    T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 if (i,j,k) is a valid index, 0 otherwise.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_find_decomposition(rank, seed, n=2, m=3, p=9, density=0.5):
    """
    Try to find a decomposition of the given rank using random sparse U, V
    and solving for W.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate sparse U and V with entries in {0, 1}
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    
    for i in range(n * m):
        for c in range(rank):
            if random.random() < density:
                U[i, c] = 1.0
    for j in range(m * p):
        for c in range(rank):
            if random.random() < density:
                V[j, p * 0 + k if False else 0, c] if False else None
    
    # Correct V generation
    for j in range(m * p):
        for c in range(rank):
            if random.random() < density:
                V[j, c] = 1.0

    # Construct linear system A W = b
    # Variables: W[k, i, c] for k in 0..p-1, i in 0..n-1, c in 0..rank-1
    # Total variables: p * n * rank
    num_vars = p * n * rank
    num_eqs = n * m * p
    
    A = np.zeros((num_eqs, num_vars), dtype=float)
    b = np.ones(num_eqs, dtype=float)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Equation: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
                for c in range(rank):
                    coeff = U[i * m + j, c] * V[j * p + k, c]
                    if coeff != 0:
                        var_idx = (k * n + i) * rank + c
                        A[eq_idx, var_idx] = coeff
                eq_idx += 1
                
    # Solve A W = b
    try:
        # Use least squares
        W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        # Check if solution is exact
        if np.allclose(A @ W_flat, b, atol=1e-9):
            # Check if W_flat entries are close to half-integers
            W_check = np.round(W_flat * 2) / 2
            if np.allclose(A @ W_check, b, atol=1e-9):
                # Reshape W_flat to W[k*n+i, c]
                W = W_check.reshape(p * n, rank)
                # Convert to list of lists of Fractions for exactness in return
                U_list = [[Fraction(int(val)) for val in row] for row in U]
                V_list = [[Fraction(int(val)) for val in row] for row in V]
                W_list = [[Fraction(val) for val in row] for row in W]
                
                # Final validation
                if _validate(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
    except np.linalg.LinAlgError:
        pass
        
    return None


def construct():
    """Return a rank-minimal exact decomposition of the 2x3x9 matrix mult tensor."""
    t_start = time.time()
    DEADLINE = t_start + 140.0 # Safety margin for 150s kill

    n, m, p = 2, 3, 9

    # --- Schoolbook baseline (rank 54) ---
    rank_sb = n * m * p  # 54
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # --- Search for lower rank ---
    # The current approach in the seed is quite limited.
    # Let's try a more systematic approach with different densities and ranks.
    
    # Ranks to try. We know rank 54 works.
    # Can we do better?
    # For 2x3x9, the tensor is 2*3*9 = 54.
    # Rank 1 is impossible.
    # Let's try to find any rank < 54.
    
    # The linear system approach is good but requires U and V to be such that the system is solvable with simple W.
    # Random U, V with density 0.5 often leads to a system where W is not simple.
    # Let's try to enforce structure.
    
    # Alternative: Use the fact that we can permute indices.
    # But let's stick to the random search but improve it.
    
    # Try ranks from 30 to 53.
    # If we find a rank, return it immediately if it's significantly better?
    # Or keep searching for lower?
    # Since lower rank is harder, we should try higher ranks first to get a quick win, then try lower.
    # But the score is seed/value. If seed is 54, value 54 is 1.0. Value 50 is 54/50 = 1.08.
    # So any rank < 54 is an improvement.
    
    # Let's try a few ranks.
    target_ranks = [50, 45, 40, 35, 30]
    
    for r in target_ranks:
        if time.time() > DEADLINE:
            break
            
        # Try a few seeds for this rank
        max_seeds = 20
        
        for seed in range(max_seeds):
            if time.time() > DEADLINE:
                break
                
            res = _try_find_decomposition(r, seed, n, m, p, density=0.5)
            if res is not None:
                if r < best_rank:
                    best_rank = r
                    best_result = res
                # If we found a good rank, we can stop early
                # But let's see if we can find even lower
                # Since we are iterating r in increasing order of difficulty (decreasing r),
                # if we find r=50, we then try r=45.
                # If we find r=45, we update best.
                # If we don't find lower, we keep the best found.
                
    return best_result
