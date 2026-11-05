"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
import time
from itertools import product

def _get_block_rank(nb, mb, pb):
    """Return the rank for a block of size nb x mb x pb (matches implementation)."""
    if nb == 1 and mb == 1 and pb == 1:
        return 1
    if nb == 1 and mb == 1 and pb == 2:
        return 2
    if nb == 1 and mb == 2 and pb == 1:
        return 2
    if nb == 1 and mb == 2 and pb == 2:
        return 4  # schoolbook
    if nb == 2 and mb == 1 and pb == 1:
        return 2
    if nb == 2 and mb == 1 and pb == 2:
        return 4
    if nb == 2 and mb == 2 and pb == 1:
        return 4  # schoolbook
    if nb == 2 and mb == 2 and pb == 2:
        return 7  # Strassen
    # Fallback: schoolbook
    return nb * mb * pb

def _find_best_split(dim):
    """
    Find the split of `dim` into parts of size 1 or 2 that minimizes
    the sum of block ranks (given a fixed other-dimension context).
    We return all valid splits as lists of (offset, size).
    """
    results = []
    def _recurse(remaining, current):
        if remaining == 0:
            results.append(list(current))
            return
        if remaining >= 1:
            current.append(1)
            _recurse(remaining - 1, current)
            current.pop()
        if remaining >= 2:
            current.append(2)
            _recurse(remaining - 2, current)
            current.pop()
    _recurse(dim, [])
    # Convert to (offset, size) lists
    split_list = []
    for sizes in results:
        offsets = []
        off = 0
        for s in sizes:
            offsets.append(off)
            off += s
        split_list.append(list(zip(offsets, sizes)))
    return split_list

def _total_rank(i_split, j_split, k_split):
    """Compute total rank for given splits."""
    total = 0
    for (i0, ib) in i_split:
        for (j0, jb) in j_split:
            for (k0, kb) in k_split:
                total += _get_block_rank(ib, jb, kb)
    return total

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Searches over all valid block splits (parts of size 1 or 2) for each dimension
    and selects the split with minimum total rank.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 5, 6
    
    # Find all valid splits
    i_splits = _find_best_split(n)
    j_splits = _find_best_split(m)
    k_splits = _find_best_split(p)
    
    # Find the split combination with minimum total rank
    best_rank = float('inf')
    best_i_split = i_splits[0]
    best_j_split = j_splits[0]
    best_k_split = k_splits[0]
    
    for i_s in i_splits:
        for j_s in j_splits:
            for k_s in k_splits:
                r = _total_rank(i_s, j_s, k_s)
                if r < best_rank:
                    best_rank = r
                    best_i_split = i_s
                    best_j_split = j_s
                    best_k_split = k_s
    
    i_split = best_i_split
    j_split = best_j_split
    k_split = best_k_split
    
    # Pre-allocate with safety margin
    max_rank = best_rank + 10
    
    U = np.zeros((n * m, max_rank), dtype=int)
    V = np.zeros((m * p, max_rank), dtype=int)
    W = np.zeros((p * n, max_rank), dtype=int)
    
    col = 0
    
    for i_0, n_b in i_split:
        if time.time() > deadline:
            break
        for j_0, m_b in j_split:
            if time.time() > deadline:
                break
            for k_0, p_b in k_split:
                if n_b == 1 and m_b == 1 and p_b == 1:
                    # Rank 1
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1

                elif n_b == 1 and m_b == 1 and p_b == 2:
                    # Rank 2
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    col += 1

                elif n_b == 1 and m_b == 2 and p_b == 1:
                    # Rank 2
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1
                    U[(i_0+0)*m + (j_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1

                elif n_b == 1 and m_b == 2 and p_b == 2:
                    # Rank 3: C_00 = A0*(B00+B01), C_01 = A0*B01, C_10 = A0*B10, C_11 = A0*(B10+B11)
                    # Actually for 1x2x2: A is 1x2, B is 2x2, C is 1x2
                    # C_0 = A_0*B_00 + A_1*B_10
                    # C_1 = A_0*B_01 + A_1*B_11
                    # Rank 3: M1 = (A0+A1)(B00+B10) -> C0
                    #         M2 = (A0+A1)(B01+B11) -> C1
                    #         M3 = A1*B10 -> C0(-), C1(-) ... let me think
                    # C_0 = A0*B00 + A1*B10
                    # C_1 = A0*B01 + A1*B11
                    # M1 = (A0+A1)(B00+B10) = A0*B00 + A0*B10 + A1*B00 + A1*B10
                    # M2 = (A0+A1)(B01+B11) = A0*B01 + A0*B11 + A1*B01 + A1*B11
                    # M3 = A1*B10
                    # M4 = A1*B11
                    # C0 = M1 - A0*B10 - A1*B00 + M3 ... this is getting complex
                    # Simpler: just use schoolbook rank 4, but we claimed rank 3
                    # Let me use a known rank-3 decomposition for 1x2x2
                    # C_0 = A_0*B_00 + A_1*B_10
                    # C_1 = A_0*B_01 + A_1*B_11
                    # M1 = A_0*(B_00 + B_01)
                    # M2 = A_1*(B_10 + B_11)
                    # M3 = A_1*B_10
                    # C_0 = M1 - A_0*B_01 + M3 ... no
                    # Let me just use schoolbook (rank 4) for safety, or verify rank 3
                    # Actually, the tensor for 1x2x2 has rank 3 (it's a known result)
                    # C_0 = A_0 B_00 + A_1 B_10
                    # C_1 = A_0 B_01 + A_1 B_11
                    # M1 = (A_0 + A_1)(B_00 + B_10)
                    # M2 = (A_0 + A_1)(B_01 + B_11)  
                    # M3 = A_1 B_10
                    # C_0 = M1 - A_0 B_00 - A_0 B_10 - A_1 B_00 + M3 ... not clean
                    # I'll use schoolbook rank 4 to be safe
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i_0+u)*m + (j_0+v), col] = 1
                                V[(j_0+v)*p + (k_0+w), col] = 1
                                W[(k_0+w)*n + (i_0+u), col] = 1
                                col += 1

                elif n_b == 2 and m_b == 1 and p_b == 1:
                    # Rank 2
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    col += 1

                elif n_b == 2 and m_b == 1 and p_b == 2:
                    # Rank 4: Schoolbook for 2x1x2
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    col += 1
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    col += 1
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1

                elif n_b == 2 and m_b == 2 and p_b == 1:
                    # Schoolbook for 2x2x1 (rank 4)
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i_0+u)*m + (j_0+v), col] = 1
                                V[(j_0+v)*p + (k_0+w), col] = 1
                                W[(k_0+w)*n + (i_0+u), col] = 1
                                col += 1

                elif n_b == 2 and m_b == 2 and p_b == 2:
                    # Rank 7: Strassen for 2x2x2
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = -1
                    col += 1
                    
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = -1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = -1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    col += 1
                    
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    U[(i_0+0)*m + (j_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = -1
                    col += 1
                    
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    U[(i_0+0)*m + (j_0+0), col] = -1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    U[(i_0+0)*m + (j_0+1), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = -1
                    V[(j_0+1)*p + (k_0+0), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1

                else:
                    # Fallback: Schoolbook
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i_0+u)*m + (j_0+v), col] = 1
                                V[(j_0+v)*p + (k_0+w), col] = 1
                                W[(k_0+w)*n + (i_0+u), col] = 1
                                col += 1

    # Trim arrays to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
