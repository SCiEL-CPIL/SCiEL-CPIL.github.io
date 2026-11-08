import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that U, V, W form a valid decomposition."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    T_decomp = np.zeros_like(T)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', W[:, r], U[:, r], V[:, r])
    
    return np.allclose(T_decomp, T, atol=1e-6)

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _gradient_descent_3x3(rank, max_iters=5000, seed=42, lr=0.01):
    """
    Gradient descent to find a rank-r decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W or None.
    """
    rng = np.random.RandomState(seed)
    n = m = p = 3
    
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
        R = T_decomp - T
        err = np.linalg.norm(R)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-8:
            return best_U, best_V, best_W
        
        grad_U = np.einsum('abc,cr,ar->br', R, V, W)
        grad_V = np.einsum('abc,br,ar->cr', R, U, W)
        grad_W = np.einsum('abc,br,cr->ar', R, U, V)
        
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm > 0:
            step_lr = lr * err / grad_norm
        else:
            step_lr = lr
            
        U -= step_lr * grad_U
        V -= step_lr * grad_V
        W -= step_lr * grad_W
        
        if it % 200 == 0 and it > 0:
            noise_scale = max(0.01, 0.05 * err)
            U += noise_scale * rng.randn(n * m, rank)
            V += noise_scale * rng.randn(m * p, rank)
            W += noise_scale * rng.randn(p * n, rank)
            
    if best_err < 1e-8:
        return best_U, best_V, best_W
    return None

def _construct_3x6_from_3x3(U3, V3, W3):
    """
    Construct a decomposition for 3x3x6 from a decomposition for 3x3x3.
    U3, V3, W3 have shape (9, R).
    Returns U, V, W for 3x3x6 with shape (9, 2R), (18, 2R), (18, 2R).
    """
    R = U3.shape[1]
    U = np.hstack([U3, U3])
    
    V = np.zeros((18, 2 * R))
    W = np.zeros((18, 2 * R))
    
    # First block: k in 0..2
    # V mapping: c_std = 3j+k' -> c_total = 6j+k'
    for j in range(3):
        for k_prime in range(3):
            c_std = j * 3 + k_prime
            c_total = j * 6 + k_prime
            V[c_total, :R] = V3[c_std, :]
            
    # W mapping: a_std = 3k'+i -> a_total = 3k'+i
    for k_prime in range(3):
        for i in range(3):
            a_std = k_prime * 3 + i
            a_total = k_prime * 3 + i
            W[a_total, :R] = W3[a_std, :]
            
    # Second block: k in 3..5
    # Let k = 3 + k'
    # V mapping: c_std = 3j+k' -> c_total = 6j + 3 + k'
    for j in range(3):
        for k_prime in range(3):
            c_std = j * 3 + k_prime
            c_total = j * 6 + 3 + k_prime
            V[c_total, R:] = V3[c_std, :]
            
    # W mapping: a_std = 3k'+i -> a_total = 9 + 3k'+i
    for k_prime in range(3):
        for i in range(3):
            a_std = k_prime * 3 + i
            a_total = 9 + k_prime * 3 + i
            W[a_total, R:] = W3[a_std, :]
            
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_result = (U_fb, V_fb, W_fb, schoolbook_rank)
    
    # Try to find a rank 19 or 20 decomposition for 3x3x3
    # and use it to build a rank 38 or 40 decomposition for 3x3x6
    
    found_3x3 = None
    best_3x3_rank = 21 # Start searching from 19 up to 20
    
    for target_rank in [19, 20]:
        if time.time() > deadline:
            break
            
        for seed in range(50):
            if time.time() > deadline:
                break
                
            result = _gradient_descent_3x3(target_rank, max_iters=2000, seed=seed, lr=0.05)
            if result is not None:
                U3, V3, W3 = result
                # Verify 3x3 decomposition
                T3 = np.zeros((9, 9, 9))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            T3[k * 3 + i, i * 3 + j, j * 3 + k] = 1.0
                T3_decomp = np.einsum('ar,br,cr->abc', W3, U3, V3)
                if np.allclose(T3_decomp, T3, atol=1e-6):
                    found_3x3 = (U3, V3, W3)
                    break
        if found_3x3 is not None:
            break
            
    if found_3x3 is not None:
        U3, V3, W3 = found_3x3
        U, V, W = _construct_3x6_from_3x3(U3, V3, W3)
        if _verify(U, V, W, n, m, p):
            rank = U.shape[1]
            best_result = (U, V, W, rank)
            # We found a better result, return it immediately?
            # Or continue to see if we can find rank 18? Unlikely.
            # Since 19 is the known lower bound for 3x3, 38 is likely optimal for this structure.
            pass
            
    U, V, W, rank = best_result
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
