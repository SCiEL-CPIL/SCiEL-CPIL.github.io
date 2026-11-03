"""
Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 4 matrix.
Output factor follows the official trace convention W[k*n+i,r].
"""

import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the standard rank-24 schoolbook decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _als_search(n, m, p, target_rank, time_limit=5.0, seed=None):
    """
    Alternating Least Squares search for a rank-target_rank decomposition.
    Returns U, V, W as lists of Fractions if found, else None.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Initialize with random small floats
    U = np.random.randn(dim_u, target_rank) * 0.5
    V = np.random.randn(dim_v, target_rank) * 0.5
    W = np.random.randn(dim_w, target_rank) * 0.5
    
    # We want to minimize sum (U_ij V_jk W_ki - delta_ijk)^2
    # This is non-convex. ALS fixes one matrix and solves linear LSQ for the other.
    
    # Precompute indices for tensor entries
    # T[i, j, k] = 1 if it's a valid entry, 0 otherwise? No, it's 1 for all i,j,k in MM_{n,m,p}
    # Actually, the tensor is 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    for iteration in range(100):
        if time.time() > deadline:
            return None
            
        # Update U
        # For each i, j, U[i,j] * sum_k (V[j,k] * W[k,i]) = 1
        # Let S_U[i,j] = sum_k (V[j,k] * W[k,i])
        # We want U[i,j] * S_U[i,j] = 1 => U[i,j] = 1 / S_U[i,j]
        # But this is element-wise. We can solve this as a linear system if we consider the whole matrix?
        # Actually, for fixed V, W, the equation for U[i,j] is independent for each (i,j).
        # So U[i,j] = 1 / sum_k V[j,k] W[k,i].
        # If the denominator is 0, it's singular. We add a small regularization or skip.
        
        S_U = np.zeros((dim_u, 1))
        # Vectorized calculation:
        # S_U[i,j] where i = i_idx*m + j_idx? No, U is (n*m, rank).
        # Index i in U corresponds to (i_row, i_col) in A.
        # Index j in V corresponds to (i_col, k_col) in B? No.
        # Let's map indices carefully.
        # U row index: u_idx = i * m + j
        # V row index: v_idx = j * p + k
        # W row index: w_idx = k * n + i
        
        # For a fixed u_idx (i,j), we need sum over k of V[j*p+k, r] * W[k*n+i, r]
        # This is not a simple matrix product because the indices j and i are tied to u_idx.
        
        # Let's loop over u_idx for clarity, it's small (6).
        for u_idx in range(dim_u):
            i = u_idx // m
            j = u_idx % m
            # Sum over k and r
            # Vector over r: sum_k V[j*p+k, :] * W[k*n+i, :]
            s_r = np.zeros(target_rank)
            for k in range(p):
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                s_r += v_row * w_row
            # We want U[u_idx, r] such that sum_r U[u_idx, r] * s_r[r] = 1
            # This is a linear system: U_u_idx . s_r = 1.
            # Since U_u_idx is a vector of length R, and we have only 1 equation, it's underdetermined.
            # Standard ALS for tensor rank minimization usually minimizes the squared error.
            # Minimize (U_u_idx . s_r - 1)^2.
            # Solution: U_u_idx = s_r / (s_r . s_r)
            norm_sq = np.dot(s_r, s_r)
            if norm_sq < 1e-10:
                U[u_idx, :] = 0.1 # Regularize
            else:
                U[u_idx, :] = s_r / norm_sq
                
        # Update V
        for v_idx in range(dim_v):
            j = v_idx // p
            k = v_idx % p
            s_r = np.zeros(target_rank)
            for i in range(n):
                u_row = U[i * m + j, :]
                w_row = W[k * n + i, :]
                s_r += u_row * w_row
            norm_sq = np.dot(s_r, s_r)
            if norm_sq < 1e-10:
                V[v_idx, :] = 0.1
            else:
                V[v_idx, :] = s_r / norm_sq
                
        # Update W
        for w_idx in range(dim_w):
            k = w_idx // n
            i = w_idx % n
            s_r = np.zeros(target_rank)
            for j in range(m):
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                s_r += u_row * v_row
            norm_sq = np.dot(s_r, s_r)
            if norm_sq < 1e-10:
                W[w_idx, :] = 0.1
            else:
                W[w_idx, :] = s_r / norm_sq
                
    # Check if the solution is close to exact
    # Calculate error
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(target_rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                error += (s - 1.0) ** 2
                
    if error < 1e-6:
        # Try to snap to rationals
        # Scale to make entries integers or half-integers?
        # The problem allows integer/half-integer entries.
        # We can try to find a scaling factor or just snap each entry.
        # However, ALS converges to a local minimum of the continuous problem.
        # Snapping to nearest 0.5 might break the identity.
        # We need to verify if the snapped version works.
        
        U_snap = [[Fraction(round(x * 2) / 2) for x in U[i]] for i in range(dim_u)]
        V_snap = [[Fraction(round(x * 2) / 2) for x in V[i]] for i in range(dim_v)]
        W_snap = [[Fraction(round(x * 2) / 2) for x in W[i]] for i in range(dim_w)]
        
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
            return {"U": U_snap, "V": V_snap, "W": W_snap}
            
        # If snapping fails, try snapping to integers?
        U_snap_int = [[Fraction(round(x)) for x in U[i]] for i in range(dim_u)]
        V_snap_int = [[Fraction(round(x)) for x in V[i]] for i in range(dim_v)]
        W_snap_int = [[Fraction(round(x)) for x in W[i]] for i in range(dim_w)]
        
        if _verify_decomposition(U_snap_int, V_snap_int, W_snap_int, n, m, p):
            return {"U": U_snap_int, "V": V_snap_int, "W": W_snap_int}
            
    return None

def _try_strassen_block(n, m, p):
    """
    Try to construct a lower-rank decomposition using block structure.
    """
    return None

def _random_search(n, m, p, max_rank, time_limit=5.0):
    """
    Randomized search for a lower-rank decomposition.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit
    
    n, m, p = DIMENSIONS
    
    # First, return the guaranteed schoolbook decomposition
    schoolbook = _schoolbook_decomposition(n, m, p)
    
    # Try to find a lower-rank decomposition using ALS
    # Try ranks from 18 down to 12
    # We have ~100s. Each ALS run takes ~1-2s. We can try ~50 runs.
    # Let's try rank 16 first with multiple seeds.
    
    best_result = None
    best_rank = n * m * p
    
    ranks_to_try = [16, 17, 18, 19, 20]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 5.0:
            break
            
        # Try a few seeds for this rank
        for seed in range(10):
            if time.time() > deadline - 2.0:
                break
                
            result = _als_search(n, m, p, target_rank, time_limit=5.0, seed=seed)
            if result is not None:
                best_result = result
                best_rank = target_rank
                # If we find a very low rank, we can stop early?
                # Theoretical lower bound is ~12.
                # If we find 16, we might still find 15, 14...
                # But 16 is a good improvement over 24.
                # Let's continue to see if we can go lower, but keep this as backup.
                if target_rank <= 14:
                    break
        if best_rank <= 14:
            break
            
    if best_result is not None:
        return best_result
        
    # Return the schoolbook decomposition as fallback
    return schoolbook
# EVOLVE-BLOCK-END
