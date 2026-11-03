"""Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np
from numpy.linalg import norm, matrix_rank

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS

# Build the target tensor T of shape (n*m, m*p, p*n) with trace convention W[k*n+i, r]
def _build_target_tensor():
    T = np.zeros((n * m, m * p, p * n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_idx = i * m + j
                V_idx = j * p + k
                W_idx = k * n + i
                T[U_idx, V_idx, W_idx] = 1
    return T

# Schoolbook decomposition of rank n*m*p
def _schoolbook():
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

# Verify exact decomposition: sum over rank of U[r,a]*V[r,a]*W[r,a] == T[i,j,k]
def _verify(U, V, W, T):
    r = U.shape[1]
    # Compute tensor from decomposition
    # T_decomp[i,j,k] = sum_a U[i,a] * V[j,a] * W[k,a]
    # Do this efficiently: for each a, outer product
    for a in range(r):
        for i in range(n * m):
            if U[i, a] == 0:
                continue
            for j in range(m * p):
                if V[j, a] == 0:
                    continue
                for k in range(p * n):
                    if W[k, a] == 0:
                        continue
                    val = U[i, a] * V[j, a] * W[k, a]
                    if T[i, j, k] != val:
                        return False
                    # Check no double counting: T should be exactly val
    # More thorough: build the full tensor and compare
    T_decomp = np.zeros((n * m, m * p, p * n), dtype=object)
    for a in range(r):
        for i in range(n * m):
            if U[i, a] == 0:
                continue
            for j in range(m * p):
                if V[j, a] == 0:
                    continue
                for k in range(p * n):
                    if W[k, a] == 0:
                        continue
                    T_decomp[i, j, k] += U[i, a] * V[j, a] * W[k, a]
    # Compare
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                if T_decomp[i, j, k] != T[i, j, k]:
                    return False
    return True

# Try to reduce rank by removing zero columns or merging
def _reduce_rank(U, V, W, T):
    rank = U.shape[1]
    # Remove zero columns
    keep = []
    for a in range(rank):
        col_zero = all(U[i, a] == 0 for i in range(n * m)) and \
                   all(V[j, a] == 0 for j in range(m * p)) and \
                   all(W[k, a] == 0 for k in range(p * n))
        if not col_zero:
            keep.append(a)
    if len(keep) < rank:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    return U, V, W

# Random perturbation + local search to find lower rank
def _try_find_lower_rank(T, max_rank, time_limit):
    start_time = time.time()
    best_rank = max_rank
    best_U, best_V, best_W = None, None, None
    
    # Initialize with schoolbook
    U, V, W = _schoolbook()
    rank = U.shape[1]
    
    # Try random linear combinations of columns to reduce effective rank
    # Idea: if we can find a linear transformation that makes some columns zero,
    # we've effectively reduced rank.
    
    # Strategy: try to find rank-r decompositions for r < n*m*p
    # by random initialization and gradient-free search
    
    target_ranks = list(range(max_rank // 2, max_rank))
    
    for target_rank in target_ranks:
        if time.time() - start_time > time_limit:
            break
        for trial in range(50):
            if time.time() - start_time > time_limit:
                break
            # Random initialization
            U_r = np.random.randint(-2, 3, (n * m, target_rank))
            V_r = np.random.randint(-2, 3, (m * p, target_rank))
            W_r = np.random.randint(-2, 3, (p * n, target_rank))
            
            # Check if this decomposes T
            if _verify(U_r, V_r, W_r, T):
                best_rank = target_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                break
        
        if best_rank <= target_rank:
            break
    
    return best_rank, best_U, best_V, best_W

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    T = _build_target_tensor()
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank
    if time.time() < deadline:
        r, U_try, V_try, W_try = _try_find_lower_rank(T, best_rank, deadline - time.time() - 10)
        if U_try is not None and r < best_rank:
            best_rank = r
            best_U, best_V, best_W = U_try, V_try, W_try
    
    # Convert to lists for return
    U_list = [[int(U_list_val) if isinstance(U_list_val, Fraction) else U_list_val 
               for U_list_val in row] for row in best_U.tolist()]
    V_list = [[int(V_list_val) if isinstance(V_list_val, Fraction) else V_list_val 
               for V_list_val in row] for row in best_V.tolist()]
    W_list = [[int(W_list_val) if isinstance(W_list_val, Fraction) else W_list_val 
               for W_list_val in row] for row in best_W.tolist()]
    
    # Handle numpy object arrays properly
    def to_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                if isinstance(val, np.integer):
                    r.append(int(val))
                elif isinstance(val, np.floating):
                    r.append(float(val))
                elif isinstance(val, Fraction):
                    r.append(val)
                elif hasattr(val, '__int__'):
                    try:
                        r.append(int(val))
                    except:
                        r.append(float(val))
                else:
                    try:
                        r.append(int(val))
                    except:
                        try:
                            r.append(float(val))
                        except:
                            r.append(val)
            result.append(r)
        return result
    
    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)
    
    return {"U": U_final, "V": V_final, "W": W_final}
