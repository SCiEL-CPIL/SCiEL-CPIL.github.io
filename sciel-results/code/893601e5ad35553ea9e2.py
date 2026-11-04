"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(result, T, atol=tol)

def _make_rank17():
    """Return the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = [
        ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _try_sat_rank16(deadline):
    """
    Try to find a rank-16 decomposition with entries in {-1, 0, 1}
    using a local search / SAT-like approach.
    Returns (U, V, W) or None.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        U, V, W = _make_rank17()
        drop_col = random.randint(0, 16)
        U = np.delete(U, drop_col, axis=1)
        V = np.delete(V, drop_col, axis=1)
        W = np.delete(W, drop_col, axis=1)
        
        max_iters = 500
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.sum((result - T) ** 2)
            
            if err < 1e-6:
                return U, V, W
            
            which = random.randint(0, 2)
            if which == 0:
                a, r = random.randint(0, n_a-1), random.randint(0, rank-1)
                old_val = U[a, r]
            elif which == 1:
                b, r = random.randint(0, n_b-1), random.randint(0, rank-1)
                old_val = V[b, r]
            else:
                c, r = random.randint(0, n_c-1), random.randint(0, rank-1)
                old_val = W[c, r]
            
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = (new_val + 1) % 3 - 1
            
            if which == 0:
                U[a, r] = new_val
            elif which == 1:
                V[b, r] = new_val
            else:
                W[c, r] = new_val
            
            result_new = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err_new = np.sum((result_new - T) ** 2)
            
            if err_new < err:
                pass
            else:
                if which == 0:
                    U[a, r] = old_val
                elif which == 1:
                    V[b, r] = old_val
                else:
                    W[c, r] = old_val
        
        result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if np.allclose(result, T, atol=1e-9):
            return U, V, W
    
    return None

def _try_structural_rank16(deadline):
    """
    Placeholder for structural search.
    Returns None immediately to prioritize other paths and avoid timeout.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    
    # First, try to find rank 16
    T = _build_tensor(n, m, p)
    
    # Try SAT-style search for rank 16
    result_16 = _try_sat_rank16(deadline)
    if result_16 is not None:
        U, V, W = result_16
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try structural approach
    result_16b = _try_structural_rank16(deadline)
    if result_16b is not None:
        U, V, W = result_16b
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 17
    U, V, W = _make_rank17()
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n_a, rank_fb))
    V_fb = np.zeros((n_b, rank_fb))
    W_fb = np.zeros((n_c, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
