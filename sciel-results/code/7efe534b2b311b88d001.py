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
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
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

def _try_rank16_search(deadline):
    """
    Try to find a rank-16 decomposition with entries in {-1, 0, 1}
    using an efficient local search with incremental error updates.
    Returns (U, V, W) or None.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    
    # Precompute the nonzero entries of T
    T_nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_nonzero.append((a, b, c))
    
    T_dict = {}
    for (a, b, c) in T_nonzero:
        T_dict[(a, b, c)] = 1.0
    
    # Start from rank-17 and drop a column
    U, V, W = _make_rank17()
    
    # We'll try multiple restarts
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        random.seed(42 + restart)
        
        # Drop a random column
        drop_col = random.randint(0, 16)
        U = np.delete(np.array(_make_rank17()[0]), drop_col, axis=1)
        V = np.delete(np.array(_make_rank17()[1]), drop_col, axis=1)
        W = np.delete(np.array(_make_rank17()[2]), drop_col, axis=1)
        
        # Compute initial error
        def compute_error(U, V, W):
            result = np.zeros((n_a, n_b, n_c))
            for r in range(rank):
                result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            err = 0.0
            for (a, b, c) in T_nonzero:
                diff = result[a, b, c] - 1.0
                err += diff * diff
            # Also count errors at zero positions
            for a in range(n_a):
                for b in range(n_b):
                    for c in range(n_c):
                        if (a, b, c) not in T_dict:
                            diff = result[a, b, c]
                            err += diff * diff
            return err
        
        current_err = compute_error(U, V, W)
        best_err = current_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        max_iters = 3000
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            if best_err < 1e-9:
                return best_U, best_V, best_W
            
            # Pick a random entry to modify
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
            
            # Apply change
            if which == 0:
                U[a, r] = new_val
            elif which == 1:
                V[b, r] = new_val
            else:
                W[c, r] = new_val
            
            # Compute new error
            new_err = compute_error(U, V, W)
            
            if new_err <= current_err:
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err < 1e-9:
                        return best_U, best_V, best_W
            else:
                # Revert
                if which == 0:
                    U[a, r] = old_val
                elif which == 1:
                    V[b, r] = old_val
                else:
                    W[c, r] = old_val
        
        if best_err < 1e-9:
            return best_U, best_V, best_W
    
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
