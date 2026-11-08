import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 9]

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    r = U.shape[1]
    T_recon = np.zeros((n, m, p))
    for c in range(r):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        T_recon += np.einsum('il,lk,ki->ilk', u_c, v_c, w_c)
    T_target = np.ones((n, m, p))
    return np.allclose(T_target, T_recon, atol=1e-9)

def _search_small_rank(n, m, p, target_rank, max_time=30.0, max_val=2.0):
    """
    Search for a rank-`target_rank` decomposition with half-integer entries
    in the range [-max_val, max_val] for M_{n,m,p}.
    """
    start_time = time.time()
    T_target = np.ones((n, m, p))
    
    # Generate candidate values (half-integers)
    candidates = np.arange(-max_val, max_val + 0.5, 0.5)
    n_cand = len(candidates)
    
    # Total variables: n*m*r + m*p*r + p*n*r
    total_vars = (n*m + m*p + p*n) * target_rank
    
    # This is too large for exhaustive search if total_vars is big
    # Instead, use a greedy/iterative approach
    
    # Try random initialization with half-integer values
    np.random.seed(42)
    for restart in range(10000):
        if time.time() - start_time > max_time:
            break
        
        U = np.random.choice(candidates, size=(n*m, target_rank))
        V = np.random.choice(candidates, size=(m*p, target_rank))
        W = np.random.choice(candidates, size=(p*n, target_rank))
        
        if _check_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_block_composition(r_233, n=2, m=3, p=9):
    """
    If we have a rank-r_233 decomposition for M_{2,3,3},
    compose it to get M_{2,3,9} with rank 3*r_233.
    """
    # First find the M_{2,3,3} decomposition
    factors = _search_small_rank(2, 3, 3, r_233, max_time=10.0, max_val=2.0)
    if factors is None:
        return None
    
    U_3, V_3, W_3 = factors
    new_rank = 3 * r_233
    
    U = np.zeros((n * m, new_rank))
    V = np.zeros((m * p, new_rank))
    W = np.zeros((p * n, new_rank))
    
    # U is repeated 3 times
    for j in range(3):
        U[:, j*r_233 : (j+1)*r_233] = U_3
    
    # V and W are block-diagonal
    for l in range(m):
        for k_prime in range(3):
            for j in range(3):
                k = 3 * j + k_prime
                v_idx = l * p + k
                v3_idx = l * 3 + k_prime
                V[v_idx, j*r_233 : (j+1)*r_233] = V_3[v3_idx, :]
    
    for i in range(n):
        for k_prime in range(3):
            for j in range(3):
                k = 3 * j + k_prime
                w_idx = k * n + i
                w3_idx = k_prime * 2 + i
                W[w_idx, j*r_233 : (j+1)*r_233] = W_3[w3_idx, :]
    
    if _check_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                U_school[i * m + l, col] = 1
                V_school[l * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find better solutions via block composition
    # M_{2,3,9} can be decomposed into 3 blocks of M_{2,3,3}
    for r in range(15, 19):  # Try ranks 15, 16, 17, 18 for M_{2,3,3}
        if time.time() > deadline - 10:
            break
        
        result = _try_block_composition(r, n, m, p)
        if result is not None:
            U, V, W = result
            new_rank = 3 * r
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                print(f"Found rank {new_rank} for M_{2,3,9}")
    
    return best_result
