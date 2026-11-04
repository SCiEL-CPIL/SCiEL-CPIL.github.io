import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

def _construct_strassen_28():
    """
    Returns the rank 28 Strassen-based decomposition.
    """
    n, m, p = 2, 4, 4
    R = 28
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen patterns for 2x2
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    
    blocks = [
        (0, 0, 0, 0),  # A[:,0:2] x B[0:2,0:2] -> C[:,0:2]
        (2, 2, 0, 0),  # A[:,2:4] x B[2:4,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # A[:,0:2] x B[0:2,2:4] -> C[:,2:4]
        (2, 2, 2, 2),  # A[:,2:4] x B[2:4,2:4] -> C[:,2:4]
    ]
    
    col_idx = 0
    for (a_cs, b_rs, b_cs, c_cs) in blocks:
        for pattern in patterns:
            for local_idx, coeff in pattern['a'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                U[local_r * m + (a_cs + local_c), col_idx] = coeff
            for local_idx, coeff in pattern['b'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                V[(b_rs + local_r) * p + (b_cs + local_c), col_idx] = coeff
            for local_idx, coeff in pattern['c'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                W[(c_cs + local_c) * n + local_r, col_idx] = coeff
            col_idx += 1
            
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=4):
    """
    Verifies if U, V, W correctly decompose the n x m x p matrix multiplication tensor.
    """
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_true = np.ones((n, m, p))
    return np.allclose(T_recon, T_true, atol=1e-9)

def _try_random_rank(R, n=2, m=4, p=4, max_attempts=1000, deadline=100.0):
    """
    Try to find a rank R decomposition with integer entries in {-1, 0, 1}
    or {-2, -1, 0, 1, 2} via random search.
    """
    t0 = time.time()
    U_shape = (n * m, R)
    V_shape = (m * p, R)
    W_shape = (p * n, R)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Random integer entries in {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=U_shape)
        V = np.random.choice([-1, 0, 1], size=V_shape)
        W = np.random.choice([-1, 0, 1], size=W_shape)
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    t0 = time.time()
    deadline = 140.0  # Safety margin
    
    # Get the baseline rank 28 solution
    U28, V28, W28 = _construct_strassen_28()
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try to find rank 27 solution
    if time.time() - t0 < deadline - 30:
        sol = _try_random_rank(27, max_attempts=5000, deadline=time.time() + 30)
        if sol is not None:
            best_U, best_V, best_W = sol
            best_rank = 27
    
    # Try to find rank 26 solution
    if best_rank > 26 and time.time() - t0 < deadline - 20:
        sol = _try_random_rank(26, max_attempts=5000, deadline=time.time() + 20)
        if sol is not None:
            best_U, best_V, best_W = sol
            best_rank = 26
    
    # Try to find rank 25 solution
    if best_rank > 25 and time.time() - t0 < deadline - 10:
        sol = _try_random_rank(25, max_attempts=5000, deadline=time.time() + 10)
        if sol is not None:
            best_U, best_V, best_W = sol
            best_rank = 25
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
