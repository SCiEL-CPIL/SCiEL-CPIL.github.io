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

def _get_rank17_solution():
    """Returns the known rank-17 solution using Strassen block."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
            
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    return U, V, W

def _try_rank_search(rank, T, n_a, n_b, n_c, start_time, deadline):
    """
    Attempt to find a solution of given rank using scipy.optimize.
    """
    if time.time() > deadline - 10:
        return None
        
    # Flatten U, V, W into a single vector x
    # x = [U_flat, V_flat, W_flat]
    # U shape (n_a, rank), V (n_b, rank), W (n_c, rank)
    
    def error_func(x):
        U = x[:n_a*rank].reshape(n_a, rank)
        V = x[n_a*rank:n_a*rank+n_b*rank].reshape(n_b, rank)
        W = x[n_a*rank+n_b*rank:].reshape(n_c, rank)
        
        # Compute tensor
        T_comp = np.zeros(T.shape)
        for r in range(rank):
            T_comp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        err = np.linalg.norm(T_comp - T)
        return err**2 # Minimize squared error

    # Initialize with random sparse matrices
    rng = np.random.default_rng(42)
    
    best_err = np.inf
    best_x = None
    
    # Try a few random restarts
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        # Initialize sparsely
        x0 = np.zeros(n_a*rank + n_b*rank + n_c*rank)
        for r in range(rank):
            # Pick random indices
            for _ in range(2): # 2 non-zeros per vector
                idx_u = rng.integers(0, n_a)
                idx_v = rng.integers(0, n_b)
                idx_w = rng.integers(0, n_c)
                val = rng.choice([-1, 1])
                x0[idx_u*rank + r] = val
                x0[n_a*rank + idx_v*rank + r] = val
                x0[n_a*rank + n_b*rank + idx_w*rank + r] = val
        
        # Optimize
        res = minimize(error_func, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        
        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x
            
        if best_err < 1e-6:
            break
            
    if best_x is None or best_err > 1e-4:
        return None
        
    # Extract U, V, W
    U = best_x[:n_a*rank].reshape(n_a, rank)
    V = best_x[n_a*rank:n_a*rank+n_b*rank].reshape(n_b, rank)
    W = best_x[n_a*rank+n_b*rank:].reshape(n_c, rank)
    
    # Round to nearest half-integer
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify
    if _verify_decomposition(U_rounded, V_rounded, W_rounded, T):
        return U_rounded, V_rounded, W_rounded
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    T = _build_tensor(n, m, p)
    
    # 1. Attempt Rank 16
    result_16 = _try_rank_search(16, T, n_a, n_b, n_c, start_time, deadline)
    if result_16 is not None:
        U, V, W = result_16
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # 2. Fallback to verified Rank 17
    U, V, W = _get_rank17_solution()
    
    if not _verify_decomposition(U, V, W, T):
        # Final fallback: Schoolbook (Rank 18)
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
