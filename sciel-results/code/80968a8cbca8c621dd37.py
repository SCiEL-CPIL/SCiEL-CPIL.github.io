"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

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

def _tensor_from_matrices(U, V, W):
    """Compute the tensor from factor matrices."""
    n_a, rank = U.shape
    n_b = V.shape[0]
    n_c = W.shape[0]
    T = np.zeros((n_a, n_b, n_c))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return T

def _make_rank17():
    """Build the known rank-17 decomposition."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = []
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    terms.append(([0], [1], [2], [1], [4], [1]))
    terms.append(([1], [1], [5], [1], [4], [1]))
    terms.append(([3], [1], [2], [1], [5], [1]))
    terms.append(([4], [1], [5], [1], [5], [1]))
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    terms.append(([2], [1], [6], [1], [0], [1]))
    terms.append(([2], [1], [7], [1], [2], [1]))
    terms.append(([2], [1], [8], [1], [4], [1]))
    terms.append(([5], [1], [6], [1], [1], [1]))
    terms.append(([5], [1], [7], [1], [3], [1]))
    terms.append(([5], [1], [8], [1], [5], [1]))
    
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
    Attempt to find a rank-16 decomposition using numerical optimization.
    """
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 16
    T = _build_tensor(n, m, p)
    
    # Flatten T for optimization
    T_flat = T.flatten()
    
    def objective(x):
        # x contains flattened U, V, W
        U = x[:n_a*rank].reshape(n_a, rank)
        V = x[n_a*rank:n_a*rank+n_b*rank].reshape(n_b, rank)
        W = x[n_a*rank+n_b*rank:].reshape(n_c, rank)
        
        T_calc = _tensor_from_matrices(U, V, W)
        err = np.sum((T_calc - T)**2)
        return err
    
    # Initialize with random sparse values
    rng = np.random.default_rng(42)
    
    best_err = np.inf
    best_x = None
    
    for restart in range(20):
        if time.time() > deadline - 30:
            break
            
        # Random initialization
        U_init = rng.normal(0, 1, (n_a, rank))
        V_init = rng.normal(0, 1, (n_b, rank))
        W_init = rng.normal(0, 1, (n_c, rank))
        
        # Make it sparse
        for _ in range(3):
            mask_u = rng.random((n_a, rank)) < 0.5
            mask_v = rng.random((n_b, rank)) < 0.5
            mask_w = rng.random((n_c, rank)) < 0.5
            U_init *= mask_u
            V_init *= mask_v
            W_init *= mask_w
            
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 5000})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception:
            continue
            
    if best_x is not None and best_err < 1e-6:
        U = best_x[:n_a*rank].reshape(n_a, rank)
        V = best_x[n_a*rank:n_a*rank+n_b*rank].reshape(n_b, rank)
        W = best_x[n_a*rank+n_b*rank:].reshape(n_c, rank)
        
        # Try to round to nearest integers or half-integers
        # Check if it's close to integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
        # If not close to integers, check if it's close to integers directly
        U_rounded = np.round(U)
        V_rounded = np.round(V)
        W_rounded = np.round(W)
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Try rank 16 search
    result_16 = _try_rank16_search(deadline)
    if result_16 is not None:
        return result_16
    
    # Fall back to rank-17 solution
    U17, V17, W17 = _make_rank17()
    
    # Verify
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U17, V17, W17, T):
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
    
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
