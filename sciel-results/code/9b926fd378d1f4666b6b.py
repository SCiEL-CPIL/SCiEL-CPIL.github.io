import time
import random
import math
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, tol=1e-9):
    """
    Verify the decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Optimization: Pre-compute V*W products?
    # Actually, for small dimensions, direct loop is fine.
    # U has 8 rows, V has 28 rows, W has 14 rows.
    # Total checks: 8 * 28 * 14 = 3136 checks.
    # Each check is a dot product of length 'rank'.
    
    for i in range(n):
        for j in range(m):
            u_row = U_arr[i * m + j]
            for k in range(p):
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                # Compute dot product
                s = float(np.dot(u_row, v_row * w_row))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_half_int_matrix(M, tol=1e-6):
    """
    Attempt to convert a float matrix to half-integers.
    Returns None if not possible within tolerance.
    """
    M = np.array(M, dtype=float)
    scaled = M * 2.0
    rounded = np.round(scaled)
    if np.max(np.abs(scaled - rounded)) > tol:
        return None
    # Return as list of lists of floats (0.5 increments are exactly representable)
    return (rounded / 2.0).tolist()

def _rank_one_tensor(u, v, w):
    """
    Create the tensor T such that T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
    Note: u is (n,m), v is (m,p), w is (p,n)
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = u[i, j] * v[j, k] * w[k, i]
    return T

def _sparse_search(rank, max_iter=2000, seed=42):
    """
    Randomly search for a sparse decomposition with rank 'rank'.
    Entries are chosen from {-1, 0, 1}.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rng = np.random.RandomState(seed)
    
    target = np.ones((n, m, p))
    
    for _ in range(max_iter):
        # Generate random sparse matrices
        # Density around 0.1 - 0.3 is typical for low rank
        density = 0.15 
        U = (rng.rand(nm, rank) < density).astype(int) * (1 + rng.randint(0, 2, size=(nm, rank)))
        U[rng.rand(nm, rank) < 0.5] = -U[rng.rand(nm, rank) < 0.5]
        
        V = (rng.rand(mp, rank) < density).astype(int) * (1 + rng.randint(0, 2, size=(mp, rank)))
        V[rng.rand(mp, rank) < 0.5] = -V[rng.rand(mp, rank) < 0.5]
        
        W = (rng.rand(pn, rank) < density).astype(int) * (1 + rng.randint(0, 2, size=(pn, rank)))
        W[rng.rand(pn, rank) < 0.5] = -W[rng.rand(pn, rank) < 0.5]
        
        # Check if it matches
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def _structured_search(rank, max_iter=500, seed=42):
    """
    Try to construct a decomposition by modifying the schoolbook solution
    or using symmetry.
    For 2x4x7, maybe we can combine terms?
    This is a heuristic attempt.
    """
    n, m, p = DIMENSIONS
    # Start with schoolbook and try to merge?
    # Schoolbook has rank 56.
    # If rank is close to 56, we might be able to perturb.
    # But for much lower ranks, this is unlikely to work directly.
    # Let's skip complex structured search for now and rely on sparse random.
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p # 56
    deadline = time.time() + 130  # Safety margin
    
    # Schoolbook fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try lower ranks
    # Theoretical lower bound is max(n*m, m*p, p*n) = 28
    # Known best for 2x4x7 is likely 56 or slightly less?
    # Let's try from 28 upwards.
    # Given time constraints, we can't try all.
    # We prioritize ranks that are "likely" to have solutions.
    # Often, improvements come in small steps or via specific constructions.
    # Let's try a few specific ranks.
    
    ranks_to_try = list(range(28, 56))
    
    # Shuffle to avoid bias, but keep some order?
    # Actually, lower rank is harder. Start from lower bound?
    # Or start from just below schoolbook?
    # Schoolbook is 56.
    # Let's try 55, 54... down to 28.
    # But 55 is very close to 56.
    
    # Strategy:
    # 1. Try 55, 54, 53 quickly with sparse search.
    # 2. If found, return.
    # 3. If not, try lower.
    
    for rank in range(schoolbook_rank - 1, 27, -1):
        if time.time() > deadline - 5:
            break
            
        # Increase iterations for lower ranks? No, lower ranks are harder, so maybe more iterations needed?
        # But we have limited time.
        # Let's give each rank a small budget.
        # If rank is close to 56, it might be easier to find a perturbation?
        # Actually, random sparse search is blind.
        
        # Use different seeds for different ranks
        seed = rank * 1000 + 42
        res = _sparse_search(rank, max_iter=100, seed=seed)
        if res is not None:
            best_result = res
            break
            
    return best_result

# Quick self-test if run directly
if __name__ == "__main__":
    res = construct()
    rank = len(res["U"][0])
    print(f"Found rank: {rank}")
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification passed.")
    else:
        print("Verification failed!")
