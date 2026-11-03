"""Candidate for matrix multiplication tensor decomposition (3x5x5)."""
import time
import numpy as np
from itertools import product

_DIM = (3, 5, 5)

def _strassen_patterns():
    """Return the 7 Strassen patterns for 2x2x2 matrix multiplication."""
    return [
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
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
                else:
                    rank += nb * mb * pb
    return rank

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p):
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
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    strassen_patterns = _strassen_patterns()
    
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
                            U[gi * m + gj, col] = val
                        for lj, lk, val in pattern['V']:
                            gj = j_off + lj
                            gk = k_off + lk
                            V[gj * p + gk, col] = val
                        for lk, li, val in pattern['W']:
                            gk = k_off + lk
                            gi = i_off + li
                            W[gk * n + gi, col] = val
                        col += 1
                else:
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj, col] = 1.0
                                V[gj * p + gk, col] = 1.0
                                W[gk * n + gi, col] = 1.0
                                col += 1
    
    return U, V, W

def _verify_vectorized(U, V, W, n, m, p):
    """Fast vectorized verification using einsum."""
    r = U.shape[1]
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    # Need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # Compute T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Using einsum: 'ijr,jkr,kr->ijk' won't work directly because W3 is (p,n,r)
    # W3[k,i,r] means we need W3 transposed: W3.T gives (n,p,r), then W3.T[i,k,r]
    W3t = W3.transpose(1, 0, 2)  # (n, p, r)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3t[i,k,r]
    T = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W3t)
    
    # Check if T is all ones
    return np.allclose(T, 1.0, atol=1e-9)

def _try_reduce_rank(U, V, W, n, m, p, max_iter=50):
    """Try to reduce rank by checking for linearly dependent columns."""
    r = U.shape[1]
    if r <= 1:
        return U, V, W
    
    # Check for linearly dependent columns in U, V, W simultaneously
    # A column r is redundant if it can be expressed as a linear combination of others
    # in all three matrices simultaneously
    
    # Build the joint matrix: for each column r, we have U[:,r], V[:,r], W[:,r]
    # Flatten each into a vector and check rank
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    
    # Stack column-wise: each "super-column" r has entries from U[:,r], V[:,r], W[:,r]
    # Shape: (n*m + m*p + p*n, r)
    joint = np.vstack([U, V, W])  # (total_rows, r)
    
    # Find the rank of the joint matrix
    try:
        svd_U, svd_S, svd_Vt = np.linalg.svd(joint, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W
    
    # Count significant singular values
    tol = 1e-10
    num_significant = np.sum(svd_S > tol)
    
    if num_significant >= r:
        return U, V, W
    
    # Keep only the first num_significant columns
    # But we need to be careful: the decomposition must still be valid
    # The SVD tells us the joint matrix has rank num_significant
    # So we can project onto the principal components
    Q = svd_U[:, :num_significant]  # (total_rows, num_significant)
    
    # New matrices: project U, V, W onto the reduced basis
    # U_new = U @ inv(svd_Vt[:num_significant, :num_significant] @ svd_S[:num_significant])
    # Actually, let's use a simpler approach: just keep the first num_significant columns
    # and verify
    
    # Try keeping different subsets of columns
    # For now, just try the first num_significant
    U_reduced = U[:, :num_significant]
    V_reduced = V[:, :num_significant]
    W_reduced = W[:, :num_significant]
    
    if _verify_vectorized(U_reduced, V_reduced, W_reduced, n, m, p):
        return U_reduced, V_reduced, W_reduced
    
    return U, V, W

def construct():
    n, m, p = _DIM
    t0 = time.time()
    deadline = t0 + 100  # Safety margin
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p  # schoolbook rank
    best_result = None
    
    # Pre-compute all (n_splits, m_splits, p_splits) combinations sorted by rank
    all_combos = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                rank = _compute_rank(n_splits, m_splits, p_splits)
                all_combos.append((rank, n_splits, m_splits, p_splits))
    
    # Sort by rank (ascending) to find the best first
    all_combos.sort(key=lambda x: x[0])
    
    for rank, n_splits, m_splits, p_splits in all_combos:
        if time.time() > deadline:
            break
        
        if rank >= best_rank:
            # No need to check higher ranks
            break
        
        U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
        
        if _verify_vectorized(U, V, W, n, m, p):
            best_rank = rank
            best_result = (U, V, W)
            
            # Try to reduce rank further
            U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p)
            if U_red.shape[1] < best_rank:
                best_rank = U_red.shape[1]
                best_result = (U_red, V_red, W_red)
    
    # Convert best_result to list-of-lists format
    if best_result is not None:
        U, V, W = best_result
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
