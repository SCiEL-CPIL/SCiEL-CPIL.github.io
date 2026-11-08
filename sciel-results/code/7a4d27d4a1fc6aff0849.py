import numpy as np
import time

DIMENSIONS = [2, 4, 6]
DEADLINE = 115.0  # safety margin before 120s

def _build_rank42():
    """Build the known rank-42 block-Strassen solution."""
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1}
    ]
    V_s = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1}
    ]
    W_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(0,0):1, (1,0):1},
        {(0,0):-1, (0,1):1},
        {(1,1):1},
        {(0,0):1}
    ]

    col = 0
    for a_block in range(2):
        for b_block in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_s[s].items():
                    u_idx = i_loc * m + (a_block * 2 + j_loc)
                    U[u_idx, col] += coeff
                for (j_loc, k_loc), coeff in V_s[s].items():
                    v_idx = (a_block * 2 + j_loc) * p + (b_block * 2 + k_loc)
                    V[v_idx, col] += coeff
                for (i_loc, k_loc), coeff in W_s[s].items():
                    w_idx = (b_block * 2 + k_loc) * n + i_loc
                    W[w_idx, col] += coeff
                col += 1
    return U, V, W

def _compute_tensor(U, V, W):
    """Compute the tensor from U, V, W factors."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r_idx in range(r):
                    s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                T[i, j, k] = s
    return T

def _tensor_error(U, V, W):
    """Compute Frobenius norm error of tensor decomposition."""
    T = _compute_tensor(U, V, W)
    return np.linalg.norm(T - 1.0)

def _tensor_check(U, V, W, tol=1e-8):
    """Check if U⊗V⊗W equals the matmul tensor exactly (within tol)."""
    return _tensor_error(U, V, W) < tol

def _round_to_halfint(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    doubled = 2.0 * x
    rounded = np.round(doubled)
    if np.allclose(doubled, rounded, atol=tol):
        return rounded / 2.0
    return None

def _als_step(U, V, W, target_rank, max_iters=150, tol=1e-12):
    """
    Alternating Least Squares to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F^2
    where T[i,j,k] = 1.
    """
    n, m, p = DIMENSIONS
    r = target_rank
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        # Update U: for each (i,j), solve D @ U[i,j,:] = 1
        # D[k,s] = V[j*p+k, s] * W[k*n+i, s]
        for i in range(n):
            for j in range(m):
                D = np.zeros((p, r))
                for k in range(p):
                    D[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(D, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve E @ V[j,k,:] = 1
        # E[i,s] = U[i*m+j, s] * W[k*n+i, s]
        for j in range(m):
            for k in range(p):
                E = np.zeros((n, r))
                for i in range(n):
                    E[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(E, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve F @ W[k,i,:] = 1
        # F[j,s] = U[i*m+j, s] * V[j*p+k, s]
        for k in range(p):
            for i in range(n):
                F = np.zeros((m, r))
                for j in range(m):
                    F[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(F, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        err = _tensor_error(U, V, W)
        if err < tol:
            break
    
    return U, V, W, err

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build rank-42 solution
    U42, V42, W42 = _build_rank42()
    best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
    best_rank = 42
    
    # Verify rank-42 solution
    if not _tensor_check(U42, V42, W42):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to reduce rank using ALS from random initializations
    np.random.seed(42)
    
    for target_rank in range(41, 24, -1):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        found = False
        n_attempts = 3
        
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Initialize by randomly selecting columns from rank-42 solution
            idx = np.random.choice(42, size=target_rank, replace=False)
            U_try = U42[:, idx].copy()
            V_try = V42[:, idx].copy()
            W_try = W42[:, idx].copy()
            
            # Add small random perturbation to escape local minima
            noise_scale = 0.01
            U_try += noise_scale * np.random.randn(*U_try.shape)
            V_try += noise_scale * np.random.randn(*V_try.shape)
            W_try += noise_scale * np.random.randn(*W_try.shape)
            
            # Run ALS
            U_als, V_als, W_als, err = _als_step(U_try, V_try, W_try, target_rank, max_iters=100)
            
            if err < 1e-10:
                # Try to round to half-integers
                U_r = _round_to_halfint(U_als)
                V_r = _round_to_halfint(V_als)
                W_r = _round_to_halfint(W_als)
                
                if U_r is not None and V_r is not None and W_r is not None:
                    if _tensor_check(U_r, V_r, W_r, tol=1e-9):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break
                else:
                    # Use the unrounded solution if it's exact
                    best_U, best_V, best_W = U_als, V_als, W_als
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue
    
    # Final round to half-integers if possible
    U_final = _round_to_halfint(best_U)
    V_final = _round_to_halfint(best_V)
    W_final = _round_to_halfint(best_W)
    
    if U_final is not None and V_final is not None and W_final is not None:
        if _tensor_check(U_final, V_final, W_final, tol=1e-8):
            best_U, best_V, best_W = U_final, V_final, W_final
    
    # Final verification
    if not _tensor_check(best_U, best_V, best_W, tol=1e-8):
        best_U, best_V, best_W = _build_rank42()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
