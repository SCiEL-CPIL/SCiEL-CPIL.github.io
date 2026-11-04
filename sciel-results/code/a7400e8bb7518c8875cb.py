"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import combinations, product

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
        ([0, 4], [1, 1], [1, 1], [1, 1], [0, 3], [1, 1]),
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

def _tensor_from_matrices(U, V, W):
    """Compute the tensor from factor matrices."""
    return np.einsum('ar,br,cr->abc', U, V, W, optimize=True)

def _try_rank16_search(deadline):
    """
    Search for a rank-16 decomposition using local optimization.
    Start from rank-17, remove one term, then try to fix the resulting error
    by adjusting other entries with local search.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    
    U17, V17, W17 = _make_rank17()
    
    # Try each possible term to remove
    for drop in range(17):
        if time.time() > deadline - 10:
            break
        
        U = np.delete(U17, drop, axis=1)
        V = np.delete(V17, drop, axis=1)
        W = np.delete(W17, drop, axis=1)
        
        # Compute error
        result = _tensor_from_matrices(U, V, W)
        err = result - T
        err_norm = np.linalg.norm(err)
        
        # Local search to fix the error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = err_norm
        
        for it in range(300):
            if time.time() > deadline - 5:
                break
            if best_err < 1e-10:
                break
            
            # Pick a random entry to modify
            which = random.randint(0, 2)
            if which == 0:
                a, r = random.randint(0, n_a-1), random.randint(0, 15)
                old = U[a, r]
            elif which == 1:
                b, r = random.randint(0, n_b-1), random.randint(0, 15)
                old = V[b, r]
            else:
                c, r = random.randint(0, n_c-1), random.randint(0, 15)
                old = W[c, r]
            
            # Try different values
            candidates = [-1, 0, 1, 0.5, -0.5, 2, -2]
            if old not in candidates:
                candidates.append(old)
            
            best_val = old
            best_local_err = best_err
            
            for new_val in candidates:
                if which == 0:
                    U[a, r] = new_val
                elif which == 1:
                    V[b, r] = new_val
                else:
                    W[c, r] = new_val
                
                result_new = _tensor_from_matrices(U, V, W)
                err_new = np.linalg.norm(result_new - T)
                
                if err_new < best_local_err:
                    best_local_err = err_new
                    best_val = new_val
                
                # Restore
                if which == 0:
                    U[a, r] = old
                elif which == 1:
                    V[b, r] = old
                else:
                    W[c, r] = old
            
            # Apply best
            if which == 0:
                U[a, r] = best_val
            elif which == 1:
                V[b, r] = best_val
            else:
                W[c, r] = best_val
            
            best_err = best_local_err
        
        if best_err < 1e-8:
            return U, V, W
    
    return None

def _try_structural_rank16(deadline):
    """
    Try a structural approach: exploit the block structure of 2x3 * 3x3.
    The tensor decomposes into two 3x3x3 sub-tensors (one for each row of A).
    If we can find rank-8 decompositions for each that share terms, we get rank < 16.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    
    # For now, this is a placeholder that returns None to avoid timeout
    # A more sophisticated approach would search for shared rank-1 terms
    # between the two row sub-tensors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    
    T = _build_tensor(n, m, p)
    
    # Try to find rank 16
    result_16 = _try_rank16_search(deadline)
    if result_16 is not None:
        U, V, W = result_16
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
