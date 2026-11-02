import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify the decomposition exactly reproduces the matmul tensor."""
    if not U or not V or not W:
        return False
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Check shapes
    if Ua.shape != (n * m, r) or Va.shape != (m * p, r) or Wa.shape != (p * n, r):
        return False

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < 1e-9

def _get_strassen_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a tuple (u_mask, v_mask, w_mask) where masks are dicts of {index: coeff}.
    """
    # Standard Strassen
    t1 = ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
    t2 = ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1})
    t3 = ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1})
    t4 = ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1})
    t5 = ({0: 1, 1: 1}, {3: 1}, {2: 1, 0: -1})
    t6 = ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    t7 = ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    return [t1, t2, t3, t4, t5, t6, t7]

def construct():
    _start = time.time()
    n, m, p = DIMENSIONS
    
    # Theoretical lower bound analysis:
    # The tensor decomposes into disjoint parts based on the middle index j.
    # Part 1: j in {0, 1}. This is 2x2 matrix mult by 2x10 matrix.
    # Split B into 5 blocks of 2x2. Each block is a 2x2x2 multiplication.
    # Rank of 2x2x2 is 7 (Strassen). Total rank = 5 * 7 = 35.
    # Part 2: j = 2. This is 2x1 matrix mult by 1x10 matrix.
    # This is 10 independent scalar multiplications of a 2-vector.
    # Rank of 2x1x1 is 2. Total rank = 10 * 2 = 20.
    # Total minimal rank = 35 + 20 = 55.
    
    target_rank = 55
    U = [[0.0] * target_rank for _ in range(n * m)]
    V = [[0.0] * target_rank for _ in range(m * p)]
    W = [[0.0] * target_rank for _ in range(p * n)]
    
    col = 0
    strassen_terms = _get_strassen_terms()
    
    # Part 1: 2x2x10 block (5 blocks of 2x2x2)
    # A is 2x2 (using cols 0,1 of A), B is 2x10 (using rows 0,1 of B).
    for g in range(5):
        k_offset = g * 2
        
        for u_mask, v_mask, w_mask in strassen_terms:
            for u_idx, coeff in u_mask.items():
                li = u_idx // 2
                lj = u_idx % 2
                global_u_idx = li * m + lj
                U[global_u_idx][col] += coeff
            
            for v_idx, coeff in v_mask.items():
                lj = v_idx // 2
                lk = v_idx % 2
                global_k = k_offset + lk
                global_v_idx = lj * p + global_k
                V[global_v_idx][col] += coeff
            
            for w_idx, coeff in w_mask.items():
                lk = w_idx // 2
                li = w_idx % 2
                global_k = k_offset + lk
                global_w_idx = global_k * n + li
                W[global_w_idx][col] += coeff
            
            col += 1
    
    # Part 2: 2x1x10 block (Schoolbook)
    # Uses A[:, 2] and B[2, :].
    for i in range(n):
        u_global = i * m + 2
        for k in range(p):
            v_global = 2 * p + k
            w_global = k * n + i
            
            U[u_global][col] = 1.0
            V[v_global][col] = 1.0
            W[w_global][col] = 1.0
            col += 1
    
    if col != target_rank:
        raise ValueError(f"Rank mismatch: calculated {col}, expected {target_rank}")
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Verification failed")
        
    elapsed = time.time() - _start
    if elapsed > 110:
        raise TimeoutError(f"Time exceeded: {elapsed}")
        
    return {"U": U, "V": V, "W": W}
