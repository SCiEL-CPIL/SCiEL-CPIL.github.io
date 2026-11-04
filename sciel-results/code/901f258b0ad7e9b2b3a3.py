import numpy as np
import time
from itertools import combinations, product

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

def _round_to_halfint(arr):
    """Round to nearest half-integer."""
    rounded = np.round(arr * 2) / 2
    return rounded

def _try_greedy_column_removal(U, V, W, deadline):
    """
    Try to reduce rank by removing columns that can be expressed as linear
    combinations of other columns, using a greedy approach.
    """
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    if current_rank <= 1:
        return U, V, W, current_rank
    
    # We'll work with the current decomposition and try to find linear dependencies
    # among the columns that preserve the tensor structure.
    
    # For each column r, check if removing it and adjusting others can still
    # produce the correct tensor. This is complex, so let's try a different approach:
    # Try to express the tensor with fewer rank-1 terms.
    
    # Build the full tensor
    U3 = U.reshape(n, m, current_rank)
    V3 = V.reshape(m, p, current_rank)
    W3 = W.reshape(p, n, current_rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = current_rank
    
    # Try removing columns from the end
    for target_rank in range(current_rank - 1, 10, -1):
        if time.time() > deadline - 5:
            break
        
        # Try to find a valid decomposition with target_rank
        success = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random factors
            rng = np.random.RandomState(hash((target_rank, attempt)) % (2**32))
            Ur = rng.randn(n * m, target_rank) * 0.1
            Vr = rng.randn(m * p, target_rank) * 0.1
            Wr = rng.randn(p * n, target_rank) * 0.1
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                M = np.einsum('jkr,kir->ijk', Vr.reshape(m, p, target_rank), Wr.reshape(p, n, target_rank))
                U_new = np.zeros((n * m, target_rank))
                for i in range(n):
                    for j in range(m):
                        Mij = M[i, j, :]
                        Tij = T[i, j, :]
                        if np.linalg.matrix_rank(Mij) >= target_rank:
                            try:
                                u, _, _, _ = np.linalg.lstsq(Mij, Tij, rcond=None)
                            except:
                                u = np.zeros(target_rank)
                        else:
                            u = np.zeros(target_rank)
                        U_new[i * m + j, :] = u
                Ur = U_new
                
                # Update V
                N = np.einsum('ijr,kir->jki', Ur.reshape(n, m, target_rank), Wr.reshape(p, n, target_rank))
                V_new = np.zeros((m * p, target_rank))
                for j in range(m):
                    for k in range(p):
                        Njk = N[j, k, :]
                        Tjk = T[:, j, k]
                        if np.linalg.matrix_rank(Njk) >= target_rank:
                            try:
                                v, _, _, _ = np.linalg.lstsq(Njk, Tjk, rcond=None)
                            except:
                                v = np.zeros(target_rank)
                        else:
                            v = np.zeros(target_rank)
                        V_new[j * p + k, :] = v
                Vr = V_new
                
                # Update W
                P = np.einsum('ijr,jkr->kij', Ur.reshape(n, m, target_rank), Vr.reshape(m, p, target_rank))
                W_new = np.zeros((p * n, target_rank))
                for k in range(p):
                    for i in range(n):
                        Pki = P[k, i, :]
                        Tki = T[i, :, k]
                        if np.linalg.matrix_rank(Pki) >= target_rank:
                            try:
                                w, _, _, _ = np.linalg.lstsq(Pki, Tki, rcond=None)
                            except:
                                w = np.zeros(target_rank)
                        else:
                            w = np.zeros(target_rank)
                        W_new[k * n + i, :] = w
                Wr = W_new
                
                # Check error
                U3 = Ur.reshape(n, m, target_rank)
                V3 = Vr.reshape(m, p, target_rank)
                W3 = Wr.reshape(p, n, target_rank)
                T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                err = np.max(np.abs(T_recon - T))
                
                if err < 1e-10:
                    success = True
                    break
            
            if success:
                Ur_r = _round_to_halfint(Ur)
                Vr_r = _round_to_halfint(Vr)
                Wr_r = _round_to_halfint(Wr)
                err_rounded = _check_decomposition_fast(Ur_r, Vr_r, Wr_r)
                if err_rounded < 1e-6:
                    best_U = Ur_r
                    best_V = Vr_r
                    best_W = Wr_r
                    best_rank = target_rank
                else:
                    best_U = Ur
                    best_V = Vr
                    best_W = Wr
                    best_rank = target_rank
                break
        
        if best_rank < current_rank:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Build the initial rank-55 decomposition
    U, V, W = _build_strassen_decomp()
    
    err = _check_decomposition_fast(U, V, W)
    if err > 1e-6:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank using greedy column removal / ALS
    if time.time() < deadline - 10:
        U2, V2, W2, new_rank = _try_greedy_column_removal(U, V, W, deadline)
        if new_rank < U.shape[1]:
            best_U, best_V, best_W = U2, V2, W2
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
