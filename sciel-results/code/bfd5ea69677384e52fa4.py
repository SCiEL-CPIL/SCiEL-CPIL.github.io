"""Combined program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

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
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
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
                                U[gi * m + gj, col] = 1
                                V[gj * p + gk, col] = 1
                                W[gk * n + gi, col] = 1
                                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity for the decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(W[k*n + i], U[i*m + j] * V[j*p + k])
                if not np.isclose(s, 1.0, atol=1e-10):
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank):
    """
    Try to reduce the rank by checking if any column is linearly dependent
    on others in a way that preserves the tensor identity.
    Returns (U, V, W, success) with reduced rank if possible.
    """
    r = U.shape[1]
    if r <= target_rank:
        return U, V, W, True
    
    # Build the constraint matrix: for each (i,j,k) triple, we need
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
    # This is a system of n*m*p equations in the r variables (columns).
    # To check if we can drop one column, we check if the remaining
    # columns can still satisfy all constraints.
    
    # More efficient: check if any column is in the span of others
    # in the "tensor" sense. This is complex, so we use a heuristic:
    # try dropping each column and see if the remaining still works.
    
    # For speed, only try a subset of columns
    n_constraints = n * m * p
    # Build constraint matrix A where A[c, t] = contribution of column c to constraint t
    # But this is O(r * n*m*p) which is fine for r~71, n*m*p=75
    
    # Actually, let's just verify the current decomposition works and
    # attempt a simple linear algebra approach to find a lower rank representation.
    
    # Construct the full tensor constraint system
    # For each constraint (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a multilinear system, not linear in the columns.
    
    # Alternative: use the fact that if rank 70 exists, we need to find
    # 70 vectors that span the same "space". This is NP-hard in general.
    
    # Given time constraints, we'll just return the current decomposition
    # and let the caller try other split configurations.
    return U, V, W, False

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as numpy arrays for schoolbook multiplication."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p
    best_U = best_V = best_W = None
    
    # Sort split combinations by expected rank to find good ones first
    candidates = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                rank = _compute_rank(n_splits, m_splits, p_splits)
                candidates.append((rank, n_splits, m_splits, p_splits))
    
    candidates.sort(key=lambda x: x[0])
    
    for rank, n_splits, m_splits, p_splits in candidates:
        if time.time() > deadline - 5:
            break
        
        if rank >= best_rank:
            continue
        
        U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
        
        if _verify_decomposition(U, V, W, n, m, p):
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            # If we hit rank 70 or lower, we can stop early
            if best_rank <= 70:
                break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
