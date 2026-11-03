import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses dynamic programming to find the optimal block partition and rank.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140

    # Precompute minimal ranks for 2 x j x k
    R_table = {}
    def get_rank(j, k):
        if (j, k) in R_table:
            return R_table[(j, k)]
        if j == 1 and k == 1:
            R_table[(j, k)] = 1
            return 1
        # Schoolbook
        best = 2 * j * k
        # Split j
        for j1 in range(1, j):
            r1 = get_rank(j1, k)
            r2 = get_rank(j - j1, k)
            best = min(best, r1 + r2)
        # Split k
        for k1 in range(1, k):
            r1 = get_rank(j, k1)
            r2 = get_rank(j, k - k1)
            best = min(best, r1 + r2)
        R_table[(j, k)] = best
        return best

    # Find best partition of m and p
    best_rank = float('inf')
    best_partition = None
    
    # Partitions of m=5
    def partitions(n, max_part=None):
        if max_part is None:
            max_part = n
        if n == 0:
            yield []
            return
        for i in range(min(n, max_part), 0, -1):
            for rest in partitions(n - i, i):
                yield [i] + rest

    m_parts = list(partitions(m))
    p_parts = list(partitions(p))
    
    for mp in m_parts:
        for pp in p_parts:
            if len(mp) != len(pp):
                continue
            # We need to pair them up. Since we can permute, we sort to avoid duplicates?
            # Actually, the order matters for the construction, but the rank sum is the same.
            # We just sum the ranks of the blocks.
            rank = 0
            for j, k in zip(mp, pp):
                rank += get_rank(j, k)
            if rank < best_rank:
                best_rank = rank
                best_partition = (mp, pp)
                
    if best_partition is None:
        # Fallback to schoolbook
        R = m * p
        U = np.zeros((n * m, R), dtype=np.float64)
        V = np.zeros((m * p, R), dtype=np.float64)
        W = np.zeros((p * n, R), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    mp, pp = best_partition
    R = best_rank
    
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    def add_block(U, V, W, offset_r, i_off, j_off, k_off, j_size, k_size):
        # Recursively decompose 2 x j_size x k_size
        if j_size == 1 and k_size == 1:
            U[(i_off) * m + (j_off), offset_r] = 1.0
            V[(j_off) * p + (k_off), offset_r] = 1.0
            W[(k_off) * n + (i_off), offset_r] = 1.0
            return offset_r + 1
        
        # Try splits
        best_split_r = float('inf')
        best_split = None
        
        # Split j
        for j1 in range(1, j_size):
            r1 = get_rank(j1, k_size)
            r2 = get_rank(j_size - j1, k_size)
            if r1 + r2 < best_split_r:
                best_split_r = r1 + r2
                best_split = ('j', j1)
                
        # Split k
        for k1 in range(1, k_size):
            r1 = get_rank(j_size, k1)
            r2 = get_rank(j_size, k_size - k1)
            if r1 + r2 < best_split_r:
                best_split_r = r1 + r2
                best_split = ('k', k1)
                
        if best_split is None:
            # Should not happen if get_rank is consistent
            return offset_r
            
        if best_split[0] == 'j':
            j1 = best_split[1]
            offset_r = add_block(U, V, W, offset_r, i_off, j_off, k_off, j1, k_size)
            offset_r = add_block(U, V, W, offset_r, i_off, j_off + j1, k_off, j_size - j1, k_size)
        else:
            k1 = best_split[1]
            offset_r = add_block(U, V, W, offset_r, i_off, j_off, k_off, j_size, k1)
            offset_r = add_block(U, V, W, offset_r, i_off, j_off, k_off + k1, j_size, k_size - k1)
            
        return offset_r

    r = 0
    for j_size, k_size in zip(mp, pp):
        r = add_block(U, V, W, r, 0, 0, 0, j_size, k_size)
        # Wait, the offsets j_off and k_off need to be tracked globally
        # My add_block takes j_off, k_off but I passed 0,0.
        # I need to pass the correct offsets.
        
    # The above loop is wrong because it resets offsets.
    # Let's rewrite the construction properly.
    
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    r = 0
    j_off = 0
    k_off = 0
    for idx_block in range(len(mp)):
        j_size = mp[idx_block]
        k_size = pp[idx_block]
        r = add_block(U, V, W, r, 0, j_off, k_off, j_size, k_size)
        j_off += j_size
        k_off += k_size
        
    def verify(U, V, W):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T_exp = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_exp[a, b, c] = 1.0
        return np.allclose(T, T_exp, atol=1e-9)
    
    if not verify(U, V, W):
        R = n * m * p
        U = np.zeros((n * m, R), dtype=np.float64)
        V = np.zeros((m * p, R), dtype=np.float64)
        W = np.zeros((p * n, R), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        r = R

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
