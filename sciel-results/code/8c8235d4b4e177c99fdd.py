"""Tensor decomposition for 2x2 by 2x5 matrix multiplication."""
import numpy as np
import time
import itertools
import sys

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START

def _build_target():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _check_decomposition(U, V, W, T, atol=1e-10):
    R = np.einsum('ad,bd,cd->abc', U, V, W)
    return np.allclose(R, T, atol=atol, rtol=0)

def _build_rank18():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank = 18
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    term_idx = 0
    def add_term(u_vals, v_vals, w_vals):
        nonlocal term_idx
        for idx, val in u_vals.items():
            U[idx, term_idx] += val
        for idx, val in v_vals.items():
            V[idx, term_idx] += val
        for idx, val in w_vals.items():
            W[idx, term_idx] += val
        term_idx += 1
    add_term({0:1, 3:1}, {0:1, 6:1}, {0:1, 1:1, 2:1, 3:1})
    add_term({2:1, 3:1}, {0:1}, {0:-1, 1:1})
    add_term({0:1}, {6:1, 5:-1}, {0:1, 1:1})
    add_term({3:1}, {0:1, 5:-1}, {0:-1, 1:1})
    add_term({0:1, 1:1}, {5:1}, {0:1, 1:-1})
    add_term({2:1, 3:1}, {1:1, 0:-1}, {0:1, 1:1})
    add_term({1:1, 2:-1}, {0:1, 5:1}, {0:1, 1:-1})
    add_term({0:1, 3:1}, {2:1, 7:1}, {4:1, 5:1, 6:1, 7:1})
    add_term({2:1, 3:1}, {2:1}, {4:-1, 5:1})
    add_term({0:1}, {7:1, 6:-1}, {4:1, 5:1})
    add_term({3:1}, {2:1, 6:-1}, {4:-1, 5:1})
    add_term({0:1, 1:1}, {6:1}, {4:1, 5:-1})
    add_term({2:1, 3:1}, {3:1, 2:-1}, {4:1, 5:1})
    add_term({1:1, 2:-1}, {2:1, 6:1}, {4:1, 5:-1})
    add_term({0:1}, {4:1}, {8:1})
    add_term({1:1}, {9:1}, {8:1})
    add_term({2:1}, {4:1}, {9:1})
    add_term({3:1}, {9:1}, {9:1})
    return U, V, W

def _als_solve_U(V, W, T, rank):
    """Solve for U given V, W, T."""
    nm = T.shape[0]
    mp = T.shape[1]
    pn = T.shape[2]
    # T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
    # For fixed a: T[a,:,:] = sum_d U[a,d] * (V[:,d] outer W[:,d])
    # Flatten: T[a,:,:].flatten() = sum_d U[a,d] * (V[:,d]*W[:,d]).flatten()
    A_mat = np.zeros((mp*pn, rank))
    for d in range(rank):
        A_mat[:, d] = (V[:, d] * W[:, d]).flatten()
    T_flat = T.reshape(nm, -1)
    U, _, _, _ = np.linalg.lstsq(A_mat, T_flat.T, rcond=None)
    return U.T

def _als_solve_V(U, W, T, rank):
    """Solve for V given U, W, T."""
    nm = T.shape[0]
    mp = T.shape[1]
    pn = T.shape[2]
    A_mat = np.zeros((nm*pn, rank))
    for d in range(rank):
        A_mat[:, d] = (U[:, d] * W[:, d]).flatten()
    T_flat = T.reshape(mp, -1)
    V, _, _, _ = np.linalg.lstsq(A_mat, T_flat.T, rcond=None)
    return V.T

def _als_solve_W(U, V, T, rank):
    """Solve for W given U, V, T."""
    nm = T.shape[0]
    mp = T.shape[1]
    pn = T.shape[2]
    A_mat = np.zeros((nm*mp, rank))
    for d in range(rank):
        A_mat[:, d] = (U[:, d] * V[:, d]).flatten()
    T_flat = T.reshape(pn, -1)
    W, _, _, _ = np.linalg.lstsq(A_mat, T_flat.T, rcond=None)
    return W.T

def _try_rank_als(U_init, V_init, W_init, T, target_rank, max_iters=200, tol=1e-8):
    """Try ALS to find a valid decomposition at target_rank."""
    rank = target_rank
    nm = T.shape[0]
    mp = T.shape[1]
    pn = T.shape[2]
    
    # Initialize with first `rank` columns
    U = U_init[:, :rank].copy()
    V = V_init[:, :rank].copy()
    W = W_init[:, :rank].copy()
    
    prev_err = np.inf
    for it in range(max_iters):
        U_new = _als_solve_U(V, W, T, rank)
        V_new = _als_solve_V(U_new, W, T, rank)
        W_new = _als_solve_W(U_new, V_new, T, rank)
        
        R = np.einsum('ad,bd,cd->abc', W_new, V_new, U_new)
        err = np.max(np.abs(R - T))
        
        if err < prev_err * (1 - 1e-12):
            prev_err = err
        else:
            break
        
        U, V, W = U_new, V_new, W_new
        
        if err < tol:
            break
    
    if _check_decomposition(U, V, W, T, atol=1e-8):
        return U, V, W
    return None

def _try_rank17_search(T, deadline):
    """Try to find rank-17 decomposition via ALS from perturbed rank-18."""
    U18, V18, W18 = _build_rank18()
    nm, mp, pn = T.shape
    
    # Strategy: try dropping each pair of terms and re-solving
    # The rank-18 has 18 terms. For rank 17, we need to find a valid 17-term decomposition.
    # Try: remove one term, add random perturbation to remaining, then ALS
    
    np.random.seed(42)
    best_result = None
    
    # Try removing each single term and doing ALS
    for drop_idx in range(18):
        if time.time() > deadline - 5:
            break
        
        U_mod = U18.copy()
        V_mod = V18.copy()
        W_mod = W18.copy()
        # Remove column drop_idx
        U_mod = np.delete(U_mod, drop_idx, axis=1)
        V_mod = np.delete(V_mod, drop_idx, axis=1)
        W_mod = np.delete(W_mod, drop_idx, axis=1)
        
        # Add small random perturbation
        U_mod = U_mod + np.random.randn(nm, 17) * 0.01
        V_mod = V_mod + np.random.randn(mp, 17) * 0.01
        W_mod = W_mod + np.random.randn(pn, 17) * 0.01
        
        result = _try_rank_als(U_mod, V_mod, W_mod, T, 17, max_iters=100)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Try: keep all 18 but merge two terms via ALS at rank 17
    # Initialize with random rank-17 factors
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        U_rand = np.random.randn(nm, 17) * 0.5
        V_rand = np.random.randn(mp, 17) * 0.5
        W_rand = np.random.randn(pn, 17) * 0.5
        
        result = _try_rank_als(U_rand, V_rand, W_rand, T, 17, max_iters=150)
        if result is not None:
            best_result = result
            break
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    T = _build_target()
    
    # First: get the rank-18 solution (guaranteed correct)
    U18, V18, W18 = _build_rank18()
    
    # Verify rank-18
    if not _check_decomposition(U18, V18, W18, T, atol=1e-10):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((nm, rank_fb))
        V_fb = np.zeros((mp, rank_fb))
        W_fb = np.zeros((pn, rank_fb))
        for c in range(rank_fb):
            i = c // (m*p)
            rem = c % (m*p)
            j = rem // p
            k = rem % p
            U_fb[i*m+j, c] = 1.0
            V_fb[j*p+k, c] = 1.0
            W_fb[k*n+i, c] = 1.0
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to find rank-17 via ALS
    if time.time() < deadline - 10:
        result17 = _try_rank17_search(T, deadline)
        if result17 is not None:
            U17, V17, W17 = result17
            # Final verification
            if _check_decomposition(U17, V17, W17, T, atol=1e-8):
                # Try to round to integers/half-integers
                U17_r = np.round(U17 * 2) / 2
                V17_r = np.round(V17 * 2) / 2
                W17_r = np.round(W17 * 2) / 2
                if _check_decomposition(U17_r, V17_r, W17_r, T, atol=1e-8):
                    U17, V17, W17 = U17_r, V17_r, W17_r
                return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Return rank-18
    U_list = U18.astype(int).tolist()
    V_list = V18.astype(int).tolist()
    W_list = W18.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
