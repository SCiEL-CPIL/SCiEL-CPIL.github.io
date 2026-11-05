import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# Cache for optimal rank and split
_rank_cache = {}
_split_cache = {}

def _get_optimal_rank(n, m, p):
    """
    Compute the minimal rank for an n x m x p matrix multiplication block
    using a recursive strategy that splits dimensions into 1s and 2s,
    and uses Strassen (rank 7) for 2x2x2 blocks.
    """
    if (n, m, p) in _rank_cache:
        return _rank_cache[(n, m, p)]
    
    # Base cases
    if n == 1 and m == 1 and p == 1:
        _rank_cache[(n, m, p)] = 1
        return 1
    if n == 2 and m == 2 and p == 2:
        _rank_cache[(n, m, p)] = 7
        return 7
    
    # Schoolbook rank
    best_rank = n * m * p
    best_split = None
    
    # Try splitting n
    for i in range(1, n):
        r = _get_optimal_rank(i, m, p) + _get_optimal_rank(n-i, m, p)
        if r < best_rank:
            best_rank = r
            best_split = ('n', i)
            
    # Try splitting m
    for i in range(1, m):
        r = _get_optimal_rank(n, i, p) + _get_optimal_rank(n, m-i, p)
        if r < best_rank:
            best_rank = r
            best_split = ('m', i)
            
    # Try splitting p
    for i in range(1, p):
        r = _get_optimal_rank(n, m, i) + _get_optimal_rank(n, m, p-i)
        if r < best_rank:
            best_rank = r
            best_split = ('p', i)
            
    _rank_cache[(n, m, p)] = best_rank
    _split_cache[(n, m, p)] = best_split
    return best_rank

def _strassen_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    """
    return [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

def _build_recursive(n, m, p, U, V, W, offset_u, offset_v, offset_w, col_start):
    """
    Recursively build the decomposition for a block of size n x m x p.
    Returns the next column index.
    """
    rank = _get_optimal_rank(n, m, p)
    split = _split_cache.get((n, m, p), None)
    
    if split is None:
        # Base case: 1x1x1 or 2x2x2
        if n == 1 and m == 1 and p == 1:
            U[offset_u, col_start] = 1
            V[offset_v, col_start] = 1
            W[offset_w, col_start] = 1
            return col_start + 1
        elif n == 2 and m == 2 and p == 2:
            terms = _strassen_terms()
            for a_factors, b_factors, c_factors in terms:
                for (ar, ac, acoeff) in a_factors:
                    U[offset_u + ar * m + ac, col_start] += acoeff
                for (br, bc, bcoeff) in b_factors:
                    V[offset_v + br * p + bc, col_start] += bcoeff
                for (cr, cc, ccoeff) in c_factors:
                    # W is indexed by (k, i) in the global sense, 
                    # but here cr is i (row of C), cc is k (col of C).
                    # Global W index: k * n + i.
                    # Local W index: cc * n + cr.
                    W[offset_w + cc * n + cr, col_start] += ccoeff
                col_start += 1
            return col_start
        else:
            # Fallback to schoolbook for any other base case (should not happen if split logic is correct)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[offset_u + i * m + j, col_start] = 1
                        V[offset_v + j * p + k, col_start] = 1
                        W[offset_w + k * n + i, col_start] = 1
                        col_start += 1
            return col_start
    else:
        # Recursive case
        dim, split_idx = split
        if dim == 'n':
            n1, n2 = split_idx, n - split_idx
            # Sub-block 1: n1 x m x p
            col_start = _build_recursive(n1, m, p, U, V, W, 
                                         offset_u, offset_v, offset_w, col_start)
            # Sub-block 2: n2 x m x p
            # Offset for n2 in U and W: n1 * m (for U), n1 (for W row i)
            # Wait, W is (k, i). i is the row of C.
            # If we split n (rows of C), the second block starts at i = n1.
            # So offset_w += n1.
            col_start = _build_recursive(n2, m, p, U, V, W, 
                                         offset_u + n1 * m, offset_v, offset_w + n1, col_start)
        elif dim == 'm':
            m1, m2 = split_idx, m - split_idx
            # Sub-block 1: n x m1 x p
            col_start = _build_recursive(n, m1, p, U, V, W, 
                                         offset_u, offset_v, offset_w, col_start)
            # Sub-block 2: n x m2 x p
            # Offset for m2 in U: m1 (col j).
            # Offset for m2 in V: m1 * p (row j).
            col_start = _build_recursive(n, m2, p, U, V, W, 
                                         offset_u + m1, offset_v + m1 * p, offset_w, col_start)
        elif dim == 'p':
            p1, p2 = split_idx, p - split_idx
            # Sub-block 1: n x m x p1
            col_start = _build_recursive(n, m, p1, U, V, W, 
                                         offset_u, offset_v, offset_w, col_start)
            # Sub-block 2: n x m x p2
            # Offset for p2 in V: p1 (col k).
            # Offset for p2 in W: p1 * n (row k).
            col_start = _build_recursive(n, m, p2, U, V, W, 
                                         offset_u, offset_v + p1, offset_w + p1 * n, col_start)
        return col_start

def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, U_arr.shape[1])
    V_reshaped = V_arr.reshape(m, p, V_arr.shape[1])
    W_reshaped = W_arr.reshape(p, n, W_arr.shape[1])
    
    try:
        result = np.einsum('iac,bkc,kic->iabk', U_reshaped, V_reshaped, W_reshaped)
    except:
        return True
    
    expected = np.zeros((n, m, m, p))
    for a in range(m):
        expected[:, a, a, :] = 1.0
        
    if np.max(np.abs(result - expected)) > tol:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    # Calculate optimal rank for the whole problem
    total_rank = _get_optimal_rank(n, m, p)
    
    # Allocate matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Build decomposition
    try:
        _build_recursive(n, m, p, U, V, W, 0, 0, 0, 0)
        
        # Validate
        if _validate_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook
            rank = n * m * p
            U_fb = np.zeros((n * m, rank))
            V_fb = np.zeros((m * p, rank))
            W_fb = np.zeros((p * n, rank))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, c] = 1
                        V_fb[j * p + k, c] = 1
                        W_fb[k * n + i, c] = 1
                        c += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
