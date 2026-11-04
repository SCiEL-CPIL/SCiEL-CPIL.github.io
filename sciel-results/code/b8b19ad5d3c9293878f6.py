"""Combined program: exhaustive split search + vectorized verification + rank reduction."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    
    Convention: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k in {0,1}
    """
    # Standard Strassen formulas:
    # C00 = M1 + M4 - M2 + M6
    # C01 = M3 + M5
    # C10 = M7 + M4
    # C11 = M1 - M3 + M5 + M7
    #
    # M1 = (A00-A11)(B10-B11)
    # M2 = (A00+A01)B11
    # M3 = A00(B00-B11)
    # M4 = A11(B00+B11)
    # M5 = (A01-A11)(B00+B01)
    # M6 = (A00+A11)B00
    # M7 = (A10-A00)(B01+B11)
    
    terms = []
    
    # M1: U=[1,0,0,-1], V=[0,0,1,-1], W contributes +1 to C00 and C11
    # W[k,i]: W[0,0]=1, W[1,1]=1
    terms.append(([1,0,0,-1], [0,0,1,-1], [1,0,0,1]))
    
    # M2: U=[1,1,0,0], V=[0,0,0,1], W contributes -1 to C00, +1 to C11
    # W[0,0]=-1, W[1,1]=1
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,0,1]))
    
    # M3: U=[1,0,0,0], V=[1,0,0,-1], W contributes +1 to C01, -1 to C11
    # W[0,1]=1, W[1,1]=-1
    terms.append(([1,0,0,0], [1,0,0,-1], [0,1,-1,0]))
    
    # M4: U=[0,0,0,1], V=[1,0,0,1], W contributes +1 to C00, +1 to C10
    # W[0,0]=1, W[0,1]=1
    terms.append(([0,0,0,1], [1,0,0,1], [1,1,0,0]))
    
    # M5: U=[0,1,0,-1], V=[1,1,0,0], W contributes +1 to C01, +1 to C11
    # W[0,1]=1, W[1,1]=1
    terms.append(([0,1,0,-1], [1,1,0,0], [0,1,0,1]))
    
    # M6: U=[1,0,0,1], V=[1,0,0,0], W contributes +1 to C00
    # W[0,0]=1
    terms.append(([1,0,0,1], [1,0,0,0], [1,0,0,0]))
    
    # M7: U=[-1,0,1,0], V=[0,1,0,1], W contributes +1 to C10, +1 to C11
    # W[0,1]=1, W[1,1]=1
    terms.append(([-1,0,1,0], [0,1,0,1], [0,1,0,1]))
    
    return terms

def _verify_strassen():
    """Verify the Strassen decomposition is correct."""
    n = m = p = 2
    terms = _strassen_2x2x2()
    reconstructed = np.zeros((n, m, p))
    for u_flat, v_flat, w_flat in terms:
        u_mat = np.array(u_flat).reshape(n, m)
        v_mat = np.array(v_flat).reshape(m, p)
        w_mat = np.array(w_flat).reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    reconstructed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
    expected = np.ones((n, m, p))
    return np.allclose(reconstructed, expected)

def _generate_splits(total, max_part=2):
    """Generate all compositions of 'total' into parts of size 1 or max_part."""
    results = []
    def helper(remaining, parts_so_far):
        if remaining == 0:
            results.append(tuple(parts_so_far))
            return
        for part in range(1, min(max_part, remaining) + 1):
            parts_so_far.append(part)
            helper(remaining - part, parts_so_far)
            parts_so_far.pop()
    helper(total, [])
    return results

def _block_cost(n_b, m_b, p_b):
    """Cost of a block: Strassen if 2x2x2, else schoolbook."""
    if n_b == 2 and m_b == 2 and p_b == 2:
        return 7
    return n_b * m_b * p_b

def _total_rank(n_splits, m_splits, p_splits):
    total = 0
    for n_b in n_splits:
        for m_b in m_splits:
            for p_b in p_splits:
                total += _block_cost(n_b, m_b, p_b)
    return total

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p):
    """Build the full decomposition for given splits using numpy arrays."""
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
        
    rank = _total_rank(n_splits, m_splits, p_splits)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    strassen_patterns = _strassen_2x2x2()
    
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                n_b = n_splits[i_b]
                m_b = m_splits[j_b]
                p_b = p_splits[k_b]
                
                i_off = n_offsets[i_b]
                j_off = m_offsets[j_b]
                k_off = p_offsets[k_b]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    for u_flat, v_flat, w_flat in strassen_patterns:
                        u_mat = np.array(u_flat).reshape(n_b, m_b)
                        v_mat = np.array(v_flat).reshape(m_b, p_b)
                        w_mat = np.array(w_flat).reshape(p_b, n_b)
                        for li in range(n_b):
                            for lj in range(m_b):
                                gi = i_off + li
                                gj = j_off + lj
                                U[gi * m + gj, col] = u_mat[li, lj]
                        for lj in range(m_b):
                            for lk in range(p_b):
                                gj = j_off + lj
                                gk = k_off + lk
                                V[gj * p + gk, col] = v_mat[lj, lk]
                        for lk in range(p_b):
                            for li in range(n_b):
                                gk = k_off + lk
                                gi = i_off + li
                                W[gk * n + gi, col] = w_mat[lk, li]
                        col += 1
                else:
                    for li in range(n_b):
                        for lj in range(m_b):
                            for lk in range(p_b):
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
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    W3t = W3.transpose(1, 0, 2)  # (n, p, r)
    T = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W3t)
    return np.allclose(T, 1.0, atol=1e-9)

def _try_reduce_rank(U, V, W, n, m, p):
    """Try to reduce rank by checking for linearly dependent columns."""
    r = U.shape[1]
    if r <= 1:
        return U, V, W
    
    # Stack all matrices vertically to check joint column dependence
    joint = np.vstack([U, V, W])  # (n*m + m*p + p*n, r)
    
    try:
        svd_U, svd_S, svd_Vt = np.linalg.svd(joint, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W
    
    tol = 1e-10
    num_significant = np.sum(svd_S > tol)
    
    if num_significant >= r:
        return U, V, W
    
    # Try keeping the first num_significant columns
    U_reduced = U[:, :num_significant]
    V_reduced = V[:, :num_significant]
    W_reduced = W[:, :num_significant]
    
    if _verify_vectorized(U_reduced, V_reduced, W_reduced, n, m, p):
        return U_reduced, V_reduced, W_reduced
    
    # Try a more aggressive approach: use the SVD basis
    # Project onto the principal components
    Q = svd_U[:, :num_significant]
    # We need to find new U', V', W' such that the decomposition still holds
    # This is complex; for now, just return the original if simple reduction fails
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Verify Strassen
    if not _verify_strassen():
        # Fallback to schoolbook if Strassen is wrong
        U = np.zeros((n*m, n*m*p), dtype=np.float64)
        V = np.zeros((m*p, n*m*p), dtype=np.float64)
        W = np.zeros((p*n, n*m*p), dtype=np.float64)
        rank = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, rank] = 1
                    V[j*p+k, rank] = 1
                    W[k*n+i, rank] = 1
                    rank += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Generate all possible splits (parts of size 1 or 2)
    n_splits_list = _generate_splits(n, max_part=2)
    m_splits_list = _generate_splits(m, max_part=2)
    p_splits_list = _generate_splits(p, max_part=2)
    
    best_rank = n * m * p
    best_result = None
    
    # Search for optimal splits
    for ns in n_splits_list:
        if time.time() > deadline:
            break
        for ms in m_splits_list:
            if time.time() > deadline:
                break
            for ps in p_splits_list:
                r = _total_rank(ns, ms, ps)
                if r < best_rank:
                    best_rank = r
                    U, V, W = _build_decomposition(ns, ms, ps, n, m, p)
                    
                    # Verify
                    if _verify_vectorized(U, V, W, n, m, p):
                        # Try to reduce rank
                        U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p)
                        if U_red.shape[1] < U.shape[1]:
                            U, V, W = U_red, V_red, W_red
                            best_rank = U.shape[1]
                    best_result = (U, V, W)
    
    if best_result is None:
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p), dtype=np.float64)
        V = np.zeros((m*p, n*m*p), dtype=np.float64)
        W = np.zeros((p*n, n*m*p), dtype=np.float64)
        rank = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, rank] = 1
                    V[j*p+k, rank] = 1
                    W[k*n+i, rank] = 1
                    rank += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_result
    
    # Convert to lists of lists as required
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
