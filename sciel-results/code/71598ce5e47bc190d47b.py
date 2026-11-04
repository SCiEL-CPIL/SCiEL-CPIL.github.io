"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _get_splits(d):
    """Generate all compositions of d into parts of size 1 or 2."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        if remaining >= 1:
            current.append(1)
            _gen(remaining - 1, current)
            current.pop()
        if remaining >= 2:
            current.append(2)
            _gen(remaining - 2, current)
            current.pop()
    _gen(d, [])
    return results

def _strassen_patterns():
    """Return the 7 Strassen patterns for 2x2x2 matrix multiplication."""
    return [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
    ]

def _block_rank(nb, mb, pb):
    if nb == 2 and mb == 2 and pb == 2:
        return 7
    return nb * mb * pb

def _compute_rank(n_splits, m_splits, p_splits):
    total = 0
    for nb in n_splits:
        for mb in m_splits:
            for pb in p_splits:
                total += _block_rank(nb, mb, pb)
    return total

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p):
    n_off = [0]
    for s in n_splits[:-1]:
        n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in m_splits[:-1]:
        m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in p_splits[:-1]:
        p_off.append(p_off[-1] + s)
    
    rank = _compute_rank(n_splits, m_splits, p_splits)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    strassen_patterns = _strassen_patterns()
    
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                i_off, j_off, k_off = n_off[i_b], m_off[j_b], p_off[k_b]
                
                if nb == 2 and mb == 2 and pb == 2:
                    for pattern in strassen_patterns:
                        for li, lj, val in pattern['U']:
                            U[(i_off + li) * m + (j_off + lj), col] = val
                        for lj, lk, val in pattern['V']:
                            V[(j_off + lj) * p + (k_off + lk), col] = val
                        for lk, li, val in pattern['W']:
                            W[(k_off + lk) * n + (i_off + li), col] = val
                        col += 1
                else:
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                U[(i_off + li) * m + (j_off + lj), col] = 1.0
                                V[(j_off + lj) * p + (k_off + lk), col] = 1.0
                                W[(k_off + lk) * n + (i_off + li), col] = 1.0
                                col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r).transpose(1, 0, 2)
    T = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _try_reduce(U, V, W, n, m, p):
    """Attempt to reduce rank by finding linearly dependent joint columns."""
    r = U.shape[1]
    if r <= 1:
        return U, V, W
    joint = np.vstack([U, V, W])
    try:
        _, S, Vt = np.linalg.svd(joint, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W
    tol = 1e-10
    num_sig = int(np.sum(S > tol))
    if num_sig >= r:
        return U, V, W
    # Try to find a valid subset of columns of size num_sig
    # First try: keep first num_sig
    for attempt in range(min(20, r)):
        if attempt == 0:
            idx = list(range(num_sig))
        else:
            # Random subset
            idx = list(np.random.choice(r, num_sig, replace=False))
        U_r = U[:, idx]
        V_r = V[:, idx]
        W_r = W[:, idx]
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin under 120s
    
    n, m, p = DIMENSIONS
    
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    # Sort all combinations by estimated rank
    candidates = []
    for ns in n_splits_list:
        for ms in m_splits_list:
            for ps in p_splits_list:
                r = _compute_rank(ns, ms, ps)
                candidates.append((r, ns, ms, ps))
    candidates.sort()
    
    best_rank = n * m * p
    best_result = None
    
    for rank_est, ns, ms, ps in candidates:
        if time.time() > deadline:
            break
        if rank_est >= best_rank and best_result is not None:
            break
        
        U, V, W = _build_decomposition(ns, ms, ps, n, m, p)
        
        if not _verify(U, V, W, n, m, p):
            continue
        
        actual_rank = U.shape[1]
        if actual_rank < best_rank:
            best_rank = actual_rank
            # Try to reduce further
            U_red, V_red, W_red = _try_reduce(U, V, W, n, m, p)
            final_rank = U_red.shape[1]
            if final_rank < best_rank:
                best_rank = final_rank
            best_result = (U_red, V_red, W_red)
    
    if best_result is None:
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        best_result = (U, V, W)
    
    U, V, W = best_result
    # Convert to lists for the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
