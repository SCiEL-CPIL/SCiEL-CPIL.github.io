import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130  # safety margin before 150s kill

def _time_ok():
    return time.time() < _DEADLINE

def _strassen_2x2():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication using Strassen (rank 7).
    Convention: U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
    U[0,0] += 1; U[3,0] += 1
    V[0,0] += 1; V[1,0] += 1; V[2,0] += 1; V[3,0] += 1
    W[0,0] += 1; W[3,0] += 1
    
    # M2 = (A10+A11)*B00 -> C10 += 1
    U[2,1] += 1; U[3,1] += 1
    V[0,1] += 1
    W[1,1] += 1  # k=0,i=1 -> index 0*2+1=1
    
    # M3 = A00*(B01-B11) -> C01 += 1
    U[0,2] += 1
    V[1,2] += 1; V[3,2] -= 1
    W[2,2] += 1  # k=1,i=0 -> index 1*2+0=2
    
    # M4 = A11*(B10-B00) -> C00 += 1, C10 += 1
    U[3,3] += 1
    V[2,3] += 1; V[0,3] -= 1
    W[0,3] += 1; W[1,3] += 1
    
    # M5 = (A00+A01)*B11 -> C00 -= 1, C01 += 1
    U[0,4] += 1; U[1,4] += 1
    V[3,4] += 1
    W[0,4] -= 1; W[2,4] += 1
    
    # M6 = (A10-A00)*(B00+B01) -> C11 += 1
    U[2,5] += 1; U[0,5] -= 1
    V[0,5] += 1; V[1,5] += 1
    W[3,5] += 1
    
    # M7 = (A01-A11)*(B10+B11) -> C11 += 1
    U[1,6] += 1; U[3,6] -= 1
    V[2,6] += 1; V[3,6] += 1
    W[3,6] += 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """
    Returns (U, V, W) for n x m @ m x p using schoolbook (rank n*m*p).
    Convention: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1
    return U, V, W

def _build_block_decomp(n, m, p, n_blocks, m_blocks, p_blocks):
    """
    Build a block decomposition for n x m @ m x p.
    n_blocks: list of block sizes for rows (sum = n)
    m_blocks: list of block sizes for middle (sum = m)
    p_blocks: list of block sizes for cols (sum = p)
    
    For each (I, J, K) block triple, use Strassen if all sizes are 2,
    otherwise use schoolbook for that block.
    """
    U_list = []
    V_list = []
    W_list = []
    
    for I, (nI, nI_start) in enumerate(zip(n_blocks, _cumsum(n_blocks))):
        for J, (mJ, mJ_start) in enumerate(zip(m_blocks, _cumsum(m_blocks))):
            for K, (pK, pK_start) in enumerate(zip(p_blocks, _cumsum(p_blocks))):
                if nI == 2 and mJ == 2 and pK == 2:
                    U_blk, V_blk, W_blk = _strassen_2x2()
                    rank_blk = 7
                else:
                    U_blk, V_blk, W_blk = _schoolbook_block(nI, mJ, pK)
                    rank_blk = nI * mJ * pK
                
                # Expand block matrices to full size
                U_full = np.zeros((n*m, rank_blk), dtype=int)
                V_full = np.zeros((m*p, rank_blk), dtype=int)
                W_full = np.zeros((p*n, rank_blk), dtype=int)
                
                for c in range(rank_blk):
                    for i in range(nI):
                        for j in range(mJ):
                            U_full[(nI_start + i)*m + (mJ_start + j), c] = U_blk[i*mJ + j, c]
                    for j in range(mJ):
                        for k in range(pK):
                            V_full[(mJ_start + j)*p + (pK_start + k), c] = V_blk[j*pK + k, c]
                    for k in range(pK):
                        for i in range(nI):
                            W_full[(pK_start + k)*n + (nI_start + i), c] = W_blk[k*nI + i, c]
                
                U_list.append(U_full)
                V_list.append(V_full)
                W_list.append(W_full)
    
    if not U_list:
        return None
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return U, V, W, U.shape[1]

def _cumsum(lst):
    """Return cumulative sums (starting positions)."""
    result = []
    s = 0
    for x in lst:
        result.append(s)
        s += x
    return result

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact:
    For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    (since the matrix multiplication tensor T[i,j,k] = 1 for all valid i,j,k)
    """
    rank = U.shape[1]
    # Check all (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """
    Faster verification using tensor contraction.
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    """
    rank = U.shape[1]
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    # For each c, the contribution to T is U[:, :, c] ⊗ V[:, :, c] ⊗ W[:, :, c]
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
    
    # T should be all ones
    if not np.all(T == 1):
        return False
    return True

def _all_partitions(total, min_size=1, max_size=None):
    """
    Generate all compositions of 'total' into parts >= min_size.
    """
    if max_size is None:
        max_size = total
    
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(list(current))
            return
        if not _time_ok():
            return
        for x in range(min_size, min(max_size, remaining) + 1):
            current.append(x)
            _gen(remaining - x, current)
            current.pop()
    
    _gen(total, [])
    return results

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of matrix multiplication M_{4,5,6}.
    """
    n, m, p = 4, 5, 6
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook = 120
    
    # Always have schoolbook as fallback
    U_school, V_school, W_school, _ = _build_block_decomp(n, m, p, [n], [m], [p])
    if U_school is not None and _verify_fast(U_school, V_school, W_school, n, m, p):
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = U_school.shape[1]
    
    # Try various block decompositions
    # Strategy: split dimensions into blocks of size 1 and 2 (and possibly 3)
    # For 4: [4], [2,2], [1,3], [3,1], [2,1,1], [1,2,1], [1,1,2], [1,1,1,1], [2,1,1] etc.
    # For 5: [5], [2,3], [3,2], [1,4], [4,1], [2,2,1], [2,1,2], [1,2,2], etc.
    # For 6: [6], [2,4], [4,2], [3,3], [2,2,2], [2,1,3], etc.
    
    # Focus on partitions that include 2s (for Strassen benefit)
    # n=4: [2,2] is best (two 2x2 blocks)
    # m=5: [2,3], [3,2], [2,2,1], [2,1,2], [1,2,2]
    # p=6: [2,2,2], [3,3], [2,4], [4,2], [2,1,3], [3,1,2], [1,2,3], etc.
    
    n_partitions = [
        [2, 2],
        [1, 3],
        [3, 1],
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [1, 1, 1, 1],
        [4],
    ]
    
    m_partitions = [
        [2, 3],
        [3, 2],
        [2, 2, 1],
        [2, 1, 2],
        [1, 2, 2],
        [1, 4],
        [4, 1],
        [1, 1, 3],
        [3, 1, 1],
        [1, 3, 1],
        [5],
    ]
    
    p_partitions = [
        [2, 2, 2],
        [3, 3],
        [2, 4],
        [4, 2],
        [2, 1, 3],
        [3, 1, 2],
        [1, 2, 3],
        [2, 3, 1],
        [3, 2, 1],
        [1, 3, 2],
        [1, 1, 4],
        [4, 1, 1],
        [1, 4, 1],
        [2, 1, 1, 2],
        [1, 2, 1, 2],
        [1, 1, 2, 2],
        [6],
    ]
    
    # Try all combinations (but be selective to stay within time)
    # Prioritize combinations with more 2x2x2 blocks
    candidates = []
    for n_p in n_partitions:
        for m_p in m_partitions:
            for p_p in p_partitions:
                # Count potential 2x2x2 blocks
                n_starts = _cumsum(n_p)
                m_starts = _cumsum(m_p)
                p_starts = _cumsum(p_p)
                
                count_222 = 0
                for i, (ni, ni_s) in enumerate(zip(n_p, n_starts)):
                    for j, (mj, mj_s) in enumerate(zip(m_p, m_starts)):
                        for k, (pk, pk_s) in enumerate(zip(p_p, p_starts)):
                            if ni == 2 and mj == 2 and pk == 2:
                                count_222 += 1
                
                # Estimate rank: 222 blocks cost 7, others cost ni*mj*pk
                est_rank = 0
                for i, (ni, ni_s) in enumerate(zip(n_p, n_starts)):
                    for j, (mj, mj_s) in enumerate(zip(m_p, m_starts)):
                        for k, (pk, pk_s) in enumerate(zip(p_p, p_starts)):
                            if ni == 2 and mj == 2 and pk == 2:
                                est_rank += 7
                            else:
                                est_rank += ni * mj * pk
                
                if est_rank < best_rank:
                    candidates.append((est_rank, count_222, n_p, m_p, p_p))
    
    # Sort by estimated rank, then by number of 222 blocks (more is better)
    candidates.sort(key=lambda x: (x[0], -x[1]))
    
    for est_rank, count_222, n_p, m_p, p_p in candidates:
        if not _time_ok():
            break
        if est_rank >= best_rank:
            break
        
        result = _build_block_decomp(n, m, p, n_p, m_p, p_p)
        if result is None:
            continue
        
        U, V, W, rank = result
        if rank >= best_rank:
            continue
        
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = rank
    
    # Also try: split only middle dimension
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:5] @ B[2:5, :]
    # First part: 4x2 @ 2x6, second: 4x3 @ 3x6
    # For 4x2 @ 2x6: split rows into [2,2], middle is [2], cols into [2,2,2]
    # Each 2x2x2 block uses Strassen (rank 7)
    # 2 row blocks * 1 mid block * 3 col blocks = 6 blocks, all 2x2x2 -> rank 42
    # For 4x3 @ 3x6: schoolbook = 72
    # Total: 42 + 72 = 114
    
    # Let's try this specific case
    # Actually this is covered by n_p=[2,2], m_p=[2,3], p_p=[2,2,2]
    # Block (I,J,K): I in {0,1}, J in {0,1}, K in {0,1,2}
    # (0,0,0): 2x2x2 -> Strassen (7)
    # (0,0,1): 2x2x2 -> Strassen (7)
    # (0,0,2): 2x2x2 -> Strassen (7)
    # (0,1,0): 2x3x2 -> schoolbook (12)
    # (0,1,1): 2x3x2 -> schoolbook (12)
    # (0,1,2): 2x3x2 -> schoolbook (12)
    # (1,0,0): 2x2x2 -> Strassen (7)
    # (1,0,1): 2x2x2 -> Strassen (7)
    # (1,0,2): 2x2x2 -> Strassen (7)
    # (1,1,0): 2x3x2 -> schoolbook (12)
    # (1,1,1): 2x3x2 -> schoolbook (12)
    # (1,1,2): 2x3x2 -> schoolbook (12)
    # Total: 6*7 + 6*12 = 42 + 72 = 114
    
    # This should be in our candidates. Let's make sure it's tried.
    # It is: n_p=[2,2], m_p=[2,3], p_p=[2,2,2]
    
    # Another option: n_p=[2,2], m_p=[3,2], p_p=[2,2,2]
    # (0,0,K): 2x3x2 -> 12 each, 3 blocks = 36
    # (0,1,K): 2x2x2 -> 7 each, 3 blocks = 21
    # (1,0,K): 2x3x2 -> 12 each, 3 blocks = 36
    # (1,1,K): 2x2x2 -> 7 each, 3 blocks = 21
    # Total: 36+21+36+21 = 114 (same)
    
    # What about m_p=[2,2,1], p_p=[2,2,2]?
    # (I,0,K): 2x2x2 -> 7, for I in {0,1}, K in {0,1,2} -> 6*7 = 42
    # (I,1,K): 2x2x2 -> 7, for I in {0,1}, K in {0,1,2} -> 6*7 = 42
    # (I,2,K): 2x1x2 -> 4, for I in {0,1}, K in {0,1,2} -> 6*4 = 24
    # Total: 42+42+24 = 108!
    
    # This is the one that achieved 108 in the cell findings!
    # n_p=[2,2], m_p=[2,2,1], p_p=[2,2,2]
    
    # Let's verify this is in our candidates list.
    # est_rank = 42+42+24 = 108 < 120, so yes.
    
    # Can we do even better?
    # Try n_p=[2,2], m_p=[1,2,2], p_p=[2,2,2]:
    # (I,0,K): 2x1x2 -> 4, 6 blocks = 24
    # (I,1,K): 2x2x2 -> 7, 6 blocks = 42
    # (I,2,K): 2x2x2 -> 7, 6 blocks = 42
    # Total: 24+42+42 = 108 (same)
    
    # Try n_p=[2,2], m_p=[2,2,1], p_p=[3,3]:
    # (I,0,0): 2x2x3 -> 12
    # (I,0,1): 2x2x3 -> 12
    # (I,1,0): 2x2x3 -> 12
    # (I,1,1): 2x2x3 -> 12
    # (I,2,0): 2x1x3 -> 6
    # (I,2,1): 2x1x3 -> 6
    # For I in {0,1}: total = 2*(12+12+12+12+6+6) = 2*60 = 120. Worse.
    
    # Try n_p=[1,3], m_p=[2,2,1], p_p=[2,2,2]:
    # (0,0,K): 1x2x2 -> 4, 3 blocks = 12
    # (0,1,K): 1x2x2 -> 4, 3 blocks = 12
    # (0,2,K): 1x1x2 -> 2, 3 blocks = 6
    # (1,0,K): 3x2x2 -> 12, 3 blocks = 36
    # (1,1,K): 3x2x2 -> 12, 3 blocks = 36
    # (1,2,K): 3x1x2 -> 6, 3 blocks = 18
    # Total: 12+12+6+36+36+18 = 120. Worse.
    
    # So 108 seems to be the best with this block approach.
    
    # Let's also try some less obvious partitions
    # n_p=[2,2], m_p=[2,2,1], p_p=[2,1,3]:
    # (I,0,0): 2x2x2 -> 7
    # (I,0,1): 2x2x1 -> 4
    # (I,0,2): 2x2x3 -> 12
    # (I,1,0): 2x2x2 -> 7
    # (I,1,1): 2x2x1 -> 4
    # (I,1,2): 2x2x3 -> 12
    # (I,2,0): 2x1x2 -> 4
    # (I,2,1): 2x1x1 -> 2
    # (I,2,2): 2x1x3 -> 6
    # For I in {0,1}: total = 2*(7+4+12+7+4+12+4+2+6) = 2*58 = 116. Worse.
    
    # The best is 108. Let's make sure we find it.
    
    if best_U is None:
        # Fallback: schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
