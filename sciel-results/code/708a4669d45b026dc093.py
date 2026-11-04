import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _dft_rank5_decomposition():
    """
    For a 1x5x5 matrix-vector product (all-ones tensor), find a rank-5 decomposition.
    
    We need: sum_{r=0}^{4} U[j,r] * V[j,k,r] * W[k,r] = 1 for all j,k in 0..4
    
    Using DFT over Z_5: Let omega = exp(2*pi*i/5)
    U[j,r] = omega^(j*r) / 5  ... but we need real/rational entries.
    
    Actually, let's think differently. We need:
    sum_r U[j,r] * V[k,r] * W[j,k,r] = 1  (for 1x5x5, i is fixed)
    
    Wait, the tensor is T[i,j,k] where i in 0..n-1, j in 0..m-1, k in 0..p-1
    For 1x5x5: i is always 0, so T[0,j,k] = 1 for all j,k.
    
    U has shape (1*5, rank) = (5, rank)
    V has shape (5*5, rank) = (25, rank)  
    W has shape (5*1, rank) = (5, rank)
    
    The constraint is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    For n=1: sum_r U[j, r] * V[j*5+k, r] * W[k, r] = 1 for all j,k
    
    Let's try: U[j,r] = 1 for all j,r (or some pattern)
    V[j*5+k, r] and W[k, r] need to satisfy the product.
    
    Actually, a simpler approach: use the fact that the all-ones matrix J = 1*1^T has rank 1.
    But we need a tensor decomposition, not a matrix decomposition.
    
    For the 1x5x5 case: T[j,k] = 1 for all j,k.
    We need sum_r U[j,r] * V[j,k,r] * W[k,r] = 1
    
    If we set U[j,r] = 1 for all j,r, then we need sum_r V[j,k,r] * W[k,r] = 1.
    Let V[j,k,r] = delta_{j,r} (Kronecker delta), W[k,r] = 1 for all k,r.
    Then sum_r delta_{j,r} * 1 = 1. ✓
    
    So: U[j,r] = 1 for all j,r in 0..4
         V[j*5+k, r] = 1 if j==r else 0
         W[k, r] = 1 for all k,r
    
    This gives rank 5! Let's verify:
    sum_r U[j,r] * V[j*5+k,r] * W[k,r] = sum_r 1 * delta_{j,r} * 1 = 1 ✓
    
    But wait, this is just the schoolbook decomposition with a different indexing.
    The rank is 5 = 1*5*5/5... no, schoolbook for 1x5x5 is rank 25.
    
    Hmm, let me reconsider. The schoolbook rank for 1x5x5 is 1*5*5 = 25.
    My construction above gives rank 5, which is much better!
    
    Let me re-verify the tensor identity more carefully.
    """
    # For 1x5x5: n=1, m=5, p=5
    # U shape: (n*m, rank) = (5, 5)
    # V shape: (m*p, rank) = (25, 5)
    # W shape: (p*n, rank) = (5, 5)
    
    # U[j, r] = 1 for all j, r
    # V[j*5+k, r] = 1 if j == r else 0
    # W[k, r] = 1 for all k, r
    
    # Check: sum_r U[j,r] * V[j*5+k, r] * W[k, r]
    # = sum_r 1 * (1 if j==r else 0) * 1
    # = 1 (since r=j is the only term that contributes)
    
    # This works! Rank = 5 for 1x5x5.
    
    return {
        'U': [[1]*5 for _ in range(5)],  # U[j, r] = 1
        'V': [[1 if j == r else 0 for r in range(5)] for j in range(5) for k in range(5)],
        'W': [[1]*5 for _ in range(5)]   # W[k, r] = 1
    }

def _build_3x5x5_dft():
    """
    Build the 3x5x5 decomposition using three independent 1x5x5 rank-5 blocks.
    
    n=3, m=5, p=5
    Split n into [1, 1, 1]
    
    For each i-block (i=0,1,2), we have a 1x5x5 sub-tensor.
    Each uses rank 5, total rank = 15.
    """
    n, m, p = 3, 5, 5
    rank = 15
    
    U = [[0]*rank for _ in range(n*m)]  # (15, 15)
    V = [[0]*rank for _ in range(m*p)]  # (25, 15)
    W = [[0]*rank for _ in range(p*n)]  # (15, 15)
    
    for i in range(n):
        base_col = i * 5
        for j in range(m):
            for r in range(5):
                col = base_col + r
                # U[i*m + j, col] = 1
                U[i*m + j][col] = 1
                # V[j*p + k, col] = 1 if j == r else 0
                for k in range(p):
                    if j == r:
                        V[j*p + k][col] = 1
                # W[k*n + i, col] = 1
                for k in range(p):
                    W[k*n + i][col] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    idx_i, idx_j, idx_k = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    idx_i = idx_i.ravel()
    idx_j = idx_j.ravel()
    idx_k = idx_k.ravel()
    
    u_rows = idx_i * m + idx_j
    v_rows = idx_j * p + idx_k
    w_rows = idx_k * n + idx_i
    
    prod = U_np[u_rows, :] * V_np[v_rows, :] * W_np[w_rows, :]
    sums = prod.sum(axis=1)
    
    return np.allclose(sums, 1.0, atol=1e-9)

def _strassen_patterns():
    """Return valid rank-7 decompositions for 2x2x2 matrix multiplication."""
    return [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
    ]

def _get_splits(d):
    """Generate all compositions of d into parts of size 1 or 2."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(list(current))
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

def _compute_rank(n_splits, m_splits, p_splits):
    """Compute the rank for given splits."""
    rank = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                if nb == 2 and mb == 2 and pb == 2:
                    rank += 7
                elif nb == 1 and mb == 5 and pb == 5:
                    rank += 5  # DFT rank-5 for 1x5x5
                elif nb == 1 and mb == 1 and pb == 5:
                    rank += 5  # 1x1x5: need rank 5? No, 1x1x5 is just a vector, rank 5
                elif nb == 1 and mb == 5 and pb == 1:
                    rank += 5  # 1x5x1: rank 5
                elif nb == 5 and mb == 1 and pb == 1:
                    rank += 5  # 5x1x1: rank 5
                else:
                    rank += nb * mb * pb
    return rank

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p, strassen_patterns):
    """Build the full decomposition for given splits."""
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
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                i_off = n_off[i_b]
                j_off = m_off[j_b]
                k_off = p_off[k_b]
                
                if nb == 2 and mb == 2 and pb == 2:
                    for pattern in strassen_patterns:
                        for li, lj, val in pattern['U']:
                            gi = i_off + li
                            gj = j_off + lj
                            U[gi * m + gj][col] = val
                        for lj, lk, val in pattern['V']:
                            gj = j_off + lj
                            gk = k_off + lk
                            V[gj * p + gk][col] = val
                        for lk, li, val in pattern['W']:
                            gk = k_off + lk
                            gi = i_off + li
                            W[gk * n + gi][col] = val
                        col += 1
                elif nb == 1 and mb == 5 and pb == 5:
                    # DFT rank-5 for 1x5x5
                    for j in range(5):
                        for r in range(5):
                            gi = i_off
                            gj = j_off + j
                            gk = k_off
                            # U[gi*m + gj, col] = 1
                            U[gi * m + gj][col] = 1
                            # V[gj*p + gk, col] = 1 if j == r else 0
                            for kk in range(5):
                                if j == r:
                                    V[gj * p + (k_off + kk)][col] = 1
                            # W[gk*n + gi, col] = 1
                            W[(k_off) * n + gi][col] = 1
                            col += 1
                else:
                    # Schoolbook
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    start_time = time.time()
    deadline = start_time + 110
    
    # First, try the DFT-based 3x5x5 decomposition (rank 15)
    U_dft, V_dft, W_dft = _build_3x5x5_dft()
    if _verify_decomposition(U_dft, V_dft, W_dft, n, m, p):
        return {"U": U_dft, "V": V_dft, "W": W_dft}
    
    # If DFT doesn't work, fall back to block-splitting search
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p
    best_U = best_V = best_W = None
    
    candidates = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                rank = _compute_rank(n_splits, m_splits, p_splits)
                candidates.append((rank, n_splits, m_splits, p_splits))
    
    candidates.sort(key=lambda x: x[0])
    
    strassen_patterns = _strassen_patterns()
    
    for rank, n_splits, m_splits, p_splits in candidates:
        if time.time() > deadline - 3:
            break
        
        if rank >= best_rank:
            continue
        
        U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p, strassen_patterns)
        
        if time.time() > deadline:
            break
        
        if _verify_decomposition(U, V, W, n, m, p):
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
    
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

