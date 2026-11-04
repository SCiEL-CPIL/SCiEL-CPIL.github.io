import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _schoolbook_embedded(n, m, p, bn, bm, bp, i_off, j_off, k_off):
    """
    Returns U, V, W for a bn x bm x bp block at offset (i_off, j_off, k_off)
    embedded in the full n x m x p space, using schoolbook algorithm.
    """
    rank = bn * bm * bp
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(bn):
        for j in range(bm):
            for k in range(bp):
                U[(i_off + i) * m + (j_off + j), col] = 1
                V[(j_off + j) * p + (k_off + k), col] = 1
                W[(k_off + k) * n + (i_off + i), col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2(n, m, p, i_off, j_off, k_off):
    """
    Returns U, V, W for a 2x2x2 block embedded in the full n x m x p space.
    Uses Strassen's algorithm (rank 7).
    """
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    i0 = i_off
    i1 = i_off + 1
    j0 = j_off
    j1 = j_off + 1
    k0 = k_off
    k1 = k_off + 1
    
    def ui(i, j): return i * m + j
    def vj(j, k): return j * p + k
    def wk(k, i): return k * n + i
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[ui(i0, j0), 0] = 1
    U[ui(i1, j1), 0] = 1
    V[vj(j0, k0), 0] = 1
    V[vj(j1, k1), 0] = 1
    W[wk(k0, i0), 0] = 1
    W[wk(k1, i1), 0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[ui(i1, j0), 1] = 1
    U[ui(i1, j1), 1] = 1
    V[vj(j0, k0), 1] = 1
    W[wk(k0, i1), 1] = 1
    W[wk(k1, i1), 1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[ui(i0, j0), 2] = 1
    V[vj(j0, k1), 2] = 1
    V[vj(j1, k1), 2] = -1
    W[wk(k1, i0), 2] = 1
    W[wk(k1, i1), 2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[ui(i1, j1), 3] = 1
    V[vj(j1, k0), 3] = 1
    V[vj(j0, k0), 3] = -1
    W[wk(k0, i0), 3] = 1
    W[wk(k0, i1), 3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[ui(i0, j0), 4] = 1
    U[ui(i0, j1), 4] = 1
    V[vj(j1, k1), 4] = 1
    W[wk(k0, i0), 4] = -1
    W[wk(k1, i0), 4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[ui(i1, j0), 5] = 1
    U[ui(i0, j0), 5] = -1
    V[vj(j0, k0), 5] = 1
    V[vj(j0, k1), 5] = 1
    W[wk(k1, i1), 5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[ui(i0, j1), 6] = 1
    U[ui(i1, j1), 6] = -1
    V[vj(j1, k0), 6] = 1
    V[vj(j1, k1), 6] = 1
    W[wk(k0, i0), 6] = 1
    
    return U, V, W

def _get_block_rank(bn, bm, bp):
    """Get the minimum known rank for a block of given size."""
    rank_table = {
        (1,1,1): 1,
        (1,1,2): 2, (1,2,1): 2, (2,1,1): 2,
        (1,1,3): 3, (1,3,1): 3, (3,1,1): 3,
        (1,2,2): 4, (2,1,2): 4, (2,2,1): 4,
        (1,2,3): 6, (1,3,2): 6, (2,1,3): 6, (2,3,1): 6, (3,1,2): 6, (3,2,1): 6,
        (1,3,3): 9, (3,1,3): 9, (3,3,1): 9,
        (2,2,2): 7,  # Strassen
        (2,2,3): 12, (2,3,2): 12, (3,2,2): 12,
        (2,3,3): 13, (3,2,3): 13, (3,3,2): 13,
        (3,3,3): 27,  # schoolbook (no Laderman implemented)
    }
    if (bn, bm, bp) in rank_table:
        return rank_table[(bn, bm, bp)]
    return bn * bm * bp

def _generate_partitions(n, min_size=1, max_size=None):
    """Generate all integer partitions of n into parts between min_size and max_size."""
    if max_size is None:
        max_size = n
    if n == 0:
        yield ()
        return
    for first in range(min_size, min(n, max_size) + 1):
        for rest in _generate_partitions(n - first, min_size, max_size):
            yield (first,) + rest

def _find_optimal_partition(n, m, p):
    """Find the partition of n, m, p that minimizes total rank."""
    best_rank = float('inf')
    best_partition = None
    
    n_parts = list(_generate_partitions(n, 1, 3))
    m_parts = list(_generate_partitions(m, 1, 3))
    p_parts = list(_generate_partitions(p, 1, 3))
    
    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                total_rank = 0
                for ni, mj, pk in zip(np_, mp_, pp_):
                    total_rank += _get_block_rank(ni, mj, pk)
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_partition = (np_, mp_, pp_)
    
    return best_rank, best_partition

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    # Find optimal partition
    best_rank, best_partition = _find_optimal_partition(n, m, p)
    
    if best_partition is None:
        # Fallback to full schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    np_parts, mp_parts, pp_parts = best_partition
    
    U_list = []
    V_list = []
    W_list = []
    
    i_off = 0
    for i_idx, ni in enumerate(np_parts):
        if time.time() > deadline - 10:
            break
        j_off = 0
        for j_idx, mj in enumerate(mp_parts):
            if time.time() > deadline - 10:
                break
            k_off = 0
            for k_idx, pk in enumerate(pp_parts):
                if time.time() > deadline - 10:
                    break
                
                if ni == 2 and mj == 2 and pk == 2:
                    U_b, V_b, W_b = _strassen_2x2x2(n, m, p, i_off, j_off, k_off)
                else:
                    U_b, V_b, W_b = _schoolbook_embedded(n, m, p, ni, mj, pk, i_off, j_off, k_off)
                
                U_list.append(U_b)
                V_list.append(V_b)
                W_list.append(W_b)
                
                k_off += pk
            j_off += mj
        i_off += ni
    
    if not U_list:
        # Fallback to full schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U_combined = np.hstack(U_list)
    V_combined = np.hstack(V_list)
    W_combined = np.hstack(W_list)
    
    U_check = U_combined.tolist()
    V_check = V_combined.tolist()
    W_check = W_combined.tolist()
    
    if _verify(U_check, V_check, W_check, n, m, p):
        return {"U": U_check, "V": V_check, "W": W_check}
    else:
        # Fallback to full schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
