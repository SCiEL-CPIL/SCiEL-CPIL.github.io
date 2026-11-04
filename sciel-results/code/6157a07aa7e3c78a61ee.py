import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_patterns():
    """Return the 7 Strassen patterns for 2x2x2 matrix multiplication.
    
    Each pattern is a rank-1 contribution to the 7 rank-1 tensors that
    decompose the 2x2x2 matrix multiplication tensor.
    
    For the tensor identity: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta(i,j,k)
    where the 2x2x2 multiplication tensor has 1s at all (i,j,k) positions
    (since C[i,k] = sum_j A[i,j]*B[j,k], the tensor T[i,j,k]=1 for all i,j,k in {0,1}).
    """
    # Standard Strassen decomposition for 2x2x2
    # The 8 entries of the multiplication tensor are all 1.
    # We need 7 rank-1 tensors whose sum gives all-ones 2x2x2 tensor.
    # 
    # Using the standard Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # Each Mr corresponds to a product of a linear form in A, a linear form in B,
    # and a linear form in C (the output coefficients).
    #
    # For the tensor decomposition, each rank-1 term is:
    # u_r(i,j) * v_r(j,k) * w_r(k,i)
    #
    # Let me derive each:
    # M1: u1(i,j) = A00+A11 → (0,0,1)+(1,1,1)
    #     v1(j,k) = B00+B11 → (0,0,1)+(1,1,1)
    #     w1(k,i) = C00+C11 → (0,0,1)+(1,1,1)  [C[k,i]: C00 is k=0,i=0; C11 is k=1,i=1]
    #
    # M2: u2(i,j) = A10+A11 → (1,0,1)+(1,1,1)
    #     v2(j,k) = B00 → (0,0,1)
    #     w2(k,i) = C10-C11 → (0,1,1)-(1,1,1)  [C10: k=1,i=0; C11: k=1,i=1]
    #
    # M3: u3(i,j) = A00 → (0,0,1)
    #     v3(j,k) = B10-B11 → (1,0,1)-(1,1,1)
    #     w3(k,i) = C01+C11 → (1,0,1)+(1,1,1)  [C01: k=0,i=1; C11: k=1,i=1]
    #
    # M4: u4(i,j) = A11 → (1,1,1)
    #     v4(j,k) = B01-B10 → (0,1,1)-(1,0,1)
    #     w4(k,i) = C00+C10 → (0,0,1)+(0,1,1)  [C00: k=0,i=0; C10: k=1,i=0]
    #
    # M5: u5(i,j) = A00+A01 → (0,0,1)+(0,1,1)
    #     v5(j,k) = B11 → (1,1,1)
    #     w5(k,i) = -C00+C10 → (0,0,-1)+(0,1,1)
    #
    # M6: u6(i,j) = A10-A00 → (1,0,1)-(0,0,1)
    #     v6(j,k) = B00+B10 → (0,0,1)+(1,0,1)
    #     w6(k,i) = C11 → (1,1,1)
    #
    # M7: u7(i,j) = A01-A11 → (0,1,1)-(1,1,1)
    #     v7(j,k) = B01+B11 → (0,1,1)+(1,1,1)
    #     w7(k,i) = C00 → (0,0,1)
    
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
            'V': [(1,0,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        {
            'U': [(1,1,1)],
            'V': [(0,1,1), (1,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (0,1,1)]
        },
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (1,0,1)],
            'W': [(1,1,1)]
        },
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(0,1,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]

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
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
                else:
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity for the decomposition using fully vectorized numpy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    
    # Compute tensor[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using einsum: sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    # = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    tensor = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # Check all entries are 1
    return np.allclose(tensor, 1.0, atol=1e-10)

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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
    return U, V, W

def _try_split(n_splits, m_splits, p_splits, n, m, p, deadline):
    """Try a specific combination of splits and return (rank, U, V, W) or None."""
    if time.time() > deadline:
        return None
    rank = _compute_rank(n_splits, m_splits, p_splits)
    U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
    if _verify_decomposition(U, V, W, n, m, p):
        return (rank, U, V, W)
    return None

def _count_twos(splits):
    """Count number of 2s in a split tuple."""
    return sum(1 for s in splits if s == 2)

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p  # schoolbook rank
    best_U = best_V = best_W = None
    
    # Pre-compute rank for all combinations and sort by rank ascending
    all_combos = []
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                rank = _compute_rank(n_splits, m_splits, p_splits)
                all_combos.append((rank, n_splits, m_splits, p_splits))
    
    # Sort by rank ascending to try best candidates first
    all_combos.sort(key=lambda x: x[0])
    
    # Try combinations in order of predicted rank
    for rank, n_splits, m_splits, p_splits in all_combos:
        if time.time() > deadline:
            break
        
        if rank >= best_rank:
            continue
        
        result = _try_split(n_splits, m_splits, p_splits, n, m, p, deadline)
        if result is not None:
            r, U, V, W = result
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U, V, W
    
    # If we found a better decomposition, return it
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
