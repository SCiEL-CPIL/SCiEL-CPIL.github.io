"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
from fractions import Fraction

def _all_partitions(size, max_block=2):
    """Generate all partitions of 'size' into blocks of size 1 or 2."""
    if size == 0:
        return [[]]
    results = []
    for rest in _all_partitions(size - 1, max_block):
        results.append([1] + rest)
    if size >= 2:
        for rest in _all_partitions(size - 2, max_block):
            results.append([2] + rest)
    return results

def _compute_rank(n_blocks, m_blocks, p_blocks):
    """Compute the total rank for a given block partition."""
    rank = 0
    for ns in n_blocks:
        for ms in m_blocks:
            for ps in p_blocks:
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                else:
                    rank += ns * ms * ps
    return rank

def _strassen_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (A_factors, B_factors, C_factors) where each is a list of (row, col, coeff).
    A is indexed by (i, j), B by (j, k), C by (i, k) where W is indexed by (k, i).
    """
    return [
        # M1 = (A00+A11)(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00(B10-B11)
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11(B01-B00)
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,1,-1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

def _build_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """
    Build the decomposition using the given block partitions.
    """
    strassen_data = _strassen_terms()
    
    # Compute offsets
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    # Compute total rank
    total_rank = _compute_rank(n_blocks, m_blocks, p_blocks)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                n0 = n_offsets[ni]
                m0 = m_offsets[mi]
                p0 = p_offsets[pi]
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    # Use Strassen
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            # cr is row of C (i), cc is col of C (k)
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                else:
                    # Use schoolbook
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                U[(n0 + i) * m + (m0 + j), col] = 1
                                V[(m0 + j) * p + (p0 + k), col] = 1
                                W[(p0 + k) * n + (n0 + i), col] = 1
                                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank

def _validate_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Check that the factor matrices define the matrix multiplication tensor.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U_arr.shape[1]):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err < tol

def _schoolbook_decomposition(n, m, p):
    """
    Constructs the schoolbook decomposition for matrix multiplication.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Find the minimum-rank block decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Generate all possible partitions
    n_partitions = _all_partitions(n)
    m_partitions = _all_partitions(m)
    p_partitions = _all_partitions(p)
    
    # Find the partition combination that gives minimum rank
    best_rank = n * m * p  # schoolbook
    best_partition = (n_partitions[0], m_partitions[0], p_partitions[0])
    
    for np_ in n_partitions:
        for mp_ in m_partitions:
            for pp_ in p_partitions:
                r = _compute_rank(np_, mp_, pp_)
                if r < best_rank:
                    best_rank = r
                    best_partition = (np_, mp_, pp_)
    
    # Build the decomposition with the best partition
    np_, mp_, pp_ = best_partition
    result, actual_rank = _build_decomposition(n, m, p, np_, mp_, pp_)
    
    # Verify the decomposition
    if time.time() < deadline:
        if not _validate_decomposition(result["U"], result["V"], result["W"], n, m, p):
            # Fallback to schoolbook
            result = _schoolbook_decomposition(n, m, p)
    
    return result
# EVOLVE-BLOCK-END
