import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 10]

def _build_strassen_decomp():
    """Build the rank-55 Strassen-based decomposition as numpy arrays."""
    n, m, p = DIMENSIONS
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    W_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            r = col
            for local_idx, val in L[s].items():
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j, r] = val
            for local_idx, val in R[s].items():
                j = local_idx // 2
                k = local_idx % 2
                V[j * p + (k_offset + k), r] = val
            for local_idx, val in W_coeffs[s].items():
                i = local_idx // 2
                k = local_idx % 2
                W[(k_offset + k) * n + i, r] = val
            col += 1
    
    for i in range(2):
        for k in range(10):
            r = col
            U[i * m + 2, r] = 1
            V[2 * p + k, r] = 1
            W[k * n + i, r] = 1
            col += 1
    
    return U, V, W

def _check_decomposition_fast(U, V, W):
    """Fast check using einsum."""
    n, m, p = DIMENSIONS
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    max_err = np.max(np.abs(T - 1.0))
    return max_err

def _check_decomposition(U, V, W):
    """Check if the decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _round_to_halfint(arr):
    """Round to nearest half-integer."""
    rounded = np.round(arr * 2) / 2
    return rounded

def _greedy_column_removal(U, V, W, deadline, max_rounds=50):
    """
    Greedy removal of redundant columns.
    A column r is redundant if removing it and solving for the remaining columns
    still gives an exact decomposition.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        removed = False
        for r in range(U_cur.shape[1]):
            if time.time() > deadline:
                break
            
            # Try removing column r
            keep = [i for i in range(U_cur.shape[1]) if i != r]
            U_try = U_cur[:, keep]
            V_try = V_cur[:, keep]
            W_try = W_cur[:, keep]
            
            # Check if the remaining columns still span the tensor
            # We need to solve: for all (i,j,k), sum_r' U[i*m+j,r'] * V[j*p+k,r'] * W[k*n+i,r'] = 1
            # This is a system of linear equations in the remaining columns.
            # Actually, the tensor is fixed at 1 for all (i,j,k).
            # The question is whether the current set of columns is linearly dependent
            # in a way that allows removal.
            
            # Simpler check: is column r a linear combination of the others?
            # For the tensor to still be decomposable, we need the column to be redundant.
            # Let's check if removing it still gives error 0.
            err = _check_decomposition_fast(U_try, V_try, W_try)
            if err < 1e-10:
                U_cur = U_try
                V_cur = V_try
                W_cur = W_try
                removed = True
                break
        
        if not removed:
            break
    
    return U_cur, V_cur, W_cur

def _als_decomposition(U, V, W, target_rank, max_iters=200, tol=1e-10, deadline=None):
    """ALS to find a lower-rank decomposition."""
    n, m, p = DIMENSIONS
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for init in range(3):
        rng = np.random.RandomState(42 + init)
        Ur = rng.randn(n * m, target_rank) * 0.1
        Vr = rng.randn(m * p, target_rank) * 0.1
        Wr = rng.randn(p * n, target_rank) * 0.1
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            M = np.einsum('jkr,kir->ijk', Vr.reshape(m, p, target_rank), Wr.reshape(p, n, target_rank))
            U_new = np.zeros((n * m, target_rank))
            for i in range(n):
                for j in range(m):
                    Mij = M[i, j, :]
                    Tij = T[i, j, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(Mij, Tij, rcond=None)
                    except:
                        u = np.zeros(target_rank)
                    U_new[i * m + j, :] = u
            
            N = np.einsum('ijr,kir->jki', Ur.reshape(n, m, target_rank), Wr.reshape(p, n, target_rank))
            V_new = np.zeros((m * p, target_rank))
            for j in range(m):
                for k in range(p):
                    Njk = N[j, k, :]
                    Tjk = T[:, j, k]
                    try:
                        v, _, _, _ = np.linalg.lstsq(Njk, Tjk, rcond=None)
                    except:
                        v = np.zeros(target_rank)
                    V_new[j * p + k, :] = v
            
            P = np.einsum('ijr,jkr->kij', Ur.reshape(n, m, target_rank), Vr.reshape(m, p, target_rank))
            W_new = np.zeros((p * n, target_rank))
            for k in range(p):
                for i in range(n):
                    Pki = P[k, i, :]
                    Tki = T[i, :, k]
                    try:
                        w, _, _, _ = np.linalg.lstsq(Pki, Tki, rcond=None)
                    except:
                        w = np.zeros(target_rank)
                    W_new[k * n + i, :] = w
            
            Ur, Vr, Wr = U_new, V_new, W_new
            
            U3 = Ur.reshape(n, m, target_rank)
            V3 = Vr.reshape(m, p, target_rank)
            W3 = Wr.reshape(p, n, target_rank)
            T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            err = np.max(np.abs(T_recon - T))
            
            if err < best_err:
                best_err = err
                best_U = Ur.copy()
                best_V = Vr.copy()
                best_W = Wr.copy()
            
            if err < tol:
                break
        
        if deadline is not None and time.time() > deadline:
            break
    
    return best_U, best_V, best_W, best_err

def _sparse_support_search(U, V, W, target_rank, deadline, max_trials=50):
    """
    Try to find a subset of columns of size target_rank that still decomposes the tensor.
    Uses random sampling of column subsets.
    """
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W, 0.0
    
    rng = np.random.RandomState(123)
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        # Randomly select target_rank columns
        cols = rng.choice(current_rank, size=target_rank, replace=False)
        U_try = U[:, cols]
        V_try = V[:, cols]
        W_try = W[:, cols]
        
        err = _check_decomposition_fast(U_try, V_try, W_try)
        if err < best_err:
            best_err = err
            best_U = U_try.copy()
            best_V = V_try.copy()
            best_W = W_try.copy()
        
        if err < 1e-10:
            break
    
    if best_err < 1e-10:
        return best_U, best_V, best_W, best_err
    else:
        return None, None, None, best_err

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build the initial rank-55 decomposition
    U, V, W = _build_strassen_decomp()
    
    err = _check_decomposition_fast(U, V, W)
    if err > 1e-6:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 55
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Phase 1: Greedy column removal
    U_g, V_g, W_g = _greedy_column_removal(U, V, W, deadline)
    rank_g = U_g.shape[1]
    if rank_g < best_rank:
        best_rank = rank_g
        best_result = (U_g.copy(), V_g.copy(), W_g.copy())
    
    # Phase 2: Sparse support search from the greedy-reduced rank downward
    current_rank = rank_g
    for target_rank in range(current_rank - 1, 29, -1):
        if time.time() > deadline:
            break
        
        Ur, Vr, Wr, err = _sparse_support_search(U_g, V_g, W_g, target_rank, deadline, max_trials=30)
        
        if Ur is not None and err < 1e-10:
            Ur_r = _round_to_halfint(Ur)
            Vr_r = _round_to_halfint(Vr)
            Wr_r = _round_to_halfint(Wr)
            err_rounded = _check_decomposition_fast(Ur_r, Vr_r, Wr_r)
            if err_rounded < 1e-6:
                best_rank = target_rank
                best_result = (Ur_r, Vr_r, Wr_r)
            else:
                best_rank = target_rank
                best_result = (Ur, Vr, Wr)
            break
    
    # Phase 3: ALS refinement if time permits
    if time.time() < deadline - 20:
        current_rank = best_rank
        for target_rank in range(current_rank - 1, 29, -1):
            if time.time() > deadline:
                break
            
            Ur, Vr, Wr, err = _als_decomposition(U_g, V_g, W_g, target_rank, max_iters=50, tol=1e-10, deadline=deadline)
            
            if Ur is not None and err < 1e-8:
                Ur_r = _round_to_halfint(Ur)
                Vr_r = _round_to_halfint(Vr)
                Wr_r = _round_to_halfint(Wr)
                err_rounded = _check_decomposition_fast(Ur_r, Vr_r, Wr_r)
                if err_rounded < 1e-6:
                    best_rank = target_rank
                    best_result = (Ur_r, Vr_r, Wr_r)
                else:
                    best_rank = target_rank
                    best_result = (Ur, Vr, Wr)
                break
    
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
