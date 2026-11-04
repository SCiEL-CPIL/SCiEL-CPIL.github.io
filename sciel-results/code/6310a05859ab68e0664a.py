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

def _rank17_solution():
    """Construct the known rank-17 solution."""
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

def _try_rank16_search(T, deadline):
    """Attempt to find a rank-16 decomposition using optimization."""
    n_a, n_b, n_c = T.shape
    rank = 16
    
    # Objective function to minimize: || T - sum_r U_r * V_r * W_r ||_F^2
    def objective(params):
        # Unpack parameters
        # U: 6x16, V: 9x16, W: 6x16
        # Total params: 6*16 + 9*16 + 6*16 = 96 + 144 + 96 = 336
        U_flat, V_flat, W_flat = params[0:96], params[96:240], params[240:336]
        U = U_flat.reshape(6, 16)
        V = V_flat.reshape(9, 16)
        W = W_flat.reshape(6, 16)
        
        # Compute tensor
        result = np.zeros(T.shape)
        for r in range(rank):
            result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        # Return squared error
        err = result - T
        return np.sum(err**2)
    
    best_error = float('inf')
    best_params = None
    rng = np.random.default_rng(42)
    
    # Try multiple random initializations
    for i in range(10):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        U_init = rng.normal(size=(6, 16))
        V_init = rng.normal(size=(9, 16))
        W_init = rng.normal(size=(6, 16))
        
        # Scale initialization to be closer to expected magnitude
        U_init *= 0.5
        V_init *= 0.5
        W_init *= 0.5
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use L-BFGS-B for optimization
        res = minimize(objective, params_init, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_error:
            best_error = res.fun
            best_params = res.x
            
            # Check if we found a valid decomposition
            if best_error < 1e-8:
                U_flat, V_flat, W_flat = best_params[0:96], best_params[96:240], best_params[240:336]
                U = U_flat.reshape(6, 16)
                V = V_flat.reshape(9, 16)
                W = W_flat.reshape(6, 16)
                
                # Verify exactly
                if _verify_decomposition(U, V, W, T):
                    return U, V, W
                    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # First, try to find a rank-16 decomposition
    rank16_result = _try_rank16_search(T, deadline)
    
    if rank16_result is not None:
        U, V, W = rank16_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank-17 solution
    U, V, W = _rank17_solution()
    
    # Verify rank-17 solution
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook rank 18
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
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
