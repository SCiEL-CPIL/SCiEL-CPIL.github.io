"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from itertools import product

def _verify_factors_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast vectorized verification using NumPy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Precompute all required dot products
    # For each (i,j,k), check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                actual = np.dot(U_arr[A], V_arr[B] * W_arr[C])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def _try_strassen_style_2x3x3(n, m, p):
    """
    Attempt a structured decomposition for M_{2,3,3} using Strassen-like ideas.
    
    For 2x2x2, Strassen reduces rank from 8 to 7.
    For 2x3x3, we can view it as two 1x3x3 blocks, or try to exploit structure.
    
    One approach: decompose the 2x3x3 multiplication into smaller blocks.
    A (2x3) * (3x3) = (2x3).
    
    Let's try to find a rank-14 or lower decomposition by hand-crafting
    or using a known algorithmic improvement.
    
    Actually, let's try a random search with more structure: fix some entries
    and solve for others, or use a smarter initialization.
    """
    # For now, this is a placeholder for structured search.
    # We'll rely on improved random search in the main loop.
    return None

def _try_random_search(n, m, p, target_rank, trials=50000, seed=42, time_limit=100):
    """Random search for a low-rank decomposition with small integer entries."""
    rng = random.Random(seed)
    start_time = time.time()
    
    for trial in range(trials):
        if time.time() - start_time > time_limit:
            break
        
        # Use smaller range for better chance of exact solutions
        # Try different value ranges
        val_range = rng.choice([1, 2, 3])
        U = [[rng.randint(-val_range, val_range) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[rng.randint(-val_range, val_range) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[rng.randint(-val_range, val_range) for _ in range(target_rank)] for _ in range(p * n)]
        
        if _verify_factors_fast(U, V, W, n, m, p):
            return U, V, W, target_rank
    return None

def _try_gaussian_elimination_rank(n, m, p, target_rank, time_limit=20):
    """
    Try to find a rank-target decomposition by solving a linear system.
    
    The tensor T has entries T[i,j,k] = 1.
    We want to find U, V, W such that for all (i,j,k):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is a system of n*m*p equations in rank*(n*m + m*p + p*n) unknowns.
    For n=2,m=3,p=3: 18 equations in 14*(6+9+6)=378 unknowns (for rank=14).
    This is underdetermined, so solutions likely exist.
    """
    start_time = time.time()
    
    # Number of equations
    num_eq = n * m * p  # 18
    
    # For each equation (i,j,k), we have:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a multilinear system. We can try to fix some variables and solve.
    
    # Strategy: fix U and V to random small integers, then solve for W.
    # But that's not quite right either.
    
    # Better: use a simple iterative approach. Start with random U, V, W and try to
    # adjust them to satisfy the equations.
    
    # For now, this is a placeholder. The random search in _try_random_search is the main approach.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # First, try structured approaches
    result = _try_strassen_style_2x3x3(n, m, p)
    if result is not None:
        U, V, W, rank = result
        if _verify_factors_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Try random search with decreasing rank, starting from lower ranks
    # Use small integer entries for better chance of exact solutions
    
    for target_rank in [11, 12, 13, 14, 15, 16, 17]:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining_time = deadline - elapsed
        if remaining_time < 10:
            break
        
        # Allocate time proportional to how likely this rank is to succeed
        # Lower ranks are harder, so give them less time
        time_budget = min(remaining_time - 5, 15)
        
        result = _try_random_search(n, m, p, target_rank, trials=50000, seed=target_rank * 1000, time_limit=time_budget)
        if result is not None:
            U, V, W, rank = result
            if _verify_factors_fast(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
