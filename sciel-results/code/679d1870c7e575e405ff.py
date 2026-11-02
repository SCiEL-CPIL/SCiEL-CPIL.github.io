"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses systematic linear algebra search over bilinear decompositions.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    expected = np.ones((n, m, p))
    
    max_err = np.max(np.abs(result - expected))
    return max_err < tol, max_err

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact fraction arithmetic."""
    U_frac = [[Fraction(int(round(U[i, c]))) for c in range(U.shape[1])] for i in range(U.shape[0])]
    V_frac = [[Fraction(int(round(V[i, c]))) for c in range(V.shape[1])] for i in range(V.shape[0])]
    W_frac = [[Fraction(int(round(W[i, c]))) for c in range(W.shape[1])] for i in range(W.shape[0])]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(U.shape[1]):
                    val += U_frac[idx_u][c] * V_frac[idx_v][c] * W_frac[idx_w][c]
                if val != Fraction(1, 1):
                    return False
    return True

def _build_strassen_2x2x2():
    """Return 7 Strassen terms for 2x2x2."""
    terms = []
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    return terms

def _terms_to_matrices(terms, n, m, p):
    """Convert list of term dicts to U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j, c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, c] = val
    
    return U, V, W

def _build_strassen_block_decomp(n, m, p, j_splits, k_splits):
    """
    Build decomposition using Strassen on 2x2 blocks.
    j_splits: list of (start, end) pairs for j dimension
    k_splits: list of (start, end) pairs for k dimension
    """
    terms = []
    strassen_terms = _build_strassen_2x2x2()
    
    for (j_start, j_end) in j_splits:
        for (k_start, k_end) in k_splits:
            j_size = j_end - j_start
            k_size = k_end - k_start
            
            if j_size == 2 and k_size == 2:
                # Use Strassen
                for u, v, w in strassen_terms:
                    u_off = {}
                    for (ii, jj), val in u.items():
                        u_off[(ii, jj + j_start)] = val
                    v_off = {}
                    for (jj, kk), val in v.items():
                        v_off[(jj + j_start, kk + k_start)] = val
                    w_off = {}
                    for (kk, ii), val in w.items():
                        w_off[(kk + k_start, ii)] = val
                    terms.append((u_off, v_off, w_off))
            else:
                # Schoolbook for this block
                for i in range(n):
                    for j in range(j_start, j_end):
                        for k in range(k_start, k_end):
                            terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))
    
    return terms

def _generate_j_splits(m):
    """Generate all ways to split dimension m into blocks of size 1 or 2."""
    splits = []
    def backtrack(pos, current):
        if pos >= m:
            splits.append(current[:])
            return
        # Try block of size 1
        current.append((pos, pos + 1))
        backtrack(pos + 1, current)
        current.pop()
        # Try block of size 2
        if pos + 2 <= m:
            current.append((pos, pos + 2))
            backtrack(pos + 2, current)
            current.pop()
    backtrack(0, [])
    return splits

def _greedy_rank_reduce(U, V, W, n, m, p, deadline):
    """Try to remove columns that are linearly redundant."""
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    rank = current_U.shape[1]
    
    reduced = True
    while reduced and time.time() < deadline:
        reduced = False
        current_rank = current_U.shape[1]
        
        for c in range(current_rank):
            if time.time() > deadline:
                break
            
            mask = [i for i in range(current_rank) if i != c]
            test_U = current_U[:, mask]
            test_V = current_V[:, mask]
            test_W = current_W[:, mask]
            
            valid, _ = _verify_decomposition(test_U, test_V, test_W, n, m, p, tol=1e-8)
            
            if valid:
                current_U, current_V, current_W = test_U, test_V, test_W
                reduced = True
                break
    
    return current_U, current_V, current_W

def _try_all_blockings(n, m, p, deadline):
    """Try all blocking strategies and return the best."""
    best_rank = float('inf')
    best_U, best_V, best_W = None, None, None
    
    j_splits = _generate_j_splits(m)
    k_splits = _generate_j_splits(p)
    
    # Sort by expected rank (fewer 2x2 blocks = more schoolbook = higher rank)
    # We want to maximize 2x2 Strassen blocks
    def count_strassen_blocks(js, ks):
        count = 0
        for (j_start, j_end) in js:
            for (k_start, k_end) in ks:
                if (j_end - j_start) == 2 and (k_end - k_start) == 2:
                    count += 1
        return count
    
    # Generate all combinations
    all_combos = []
    for js in j_splits:
        for ks in k_splits:
            sb = count_strassen_blocks(js, ks)
            all_combos.append((sb, js, ks))
    
    # Sort by number of Strassen blocks (descending)
    all_combos.sort(key=lambda x: -x[0])
    
    for sb, js, ks in all_combos:
        if time.time() > deadline - 30:
            break
        
        terms = _build_strassen_block_decomp(n, m, p, js, ks)
        U, V, W = _terms_to_matrices(terms, n, m, p)
        
        valid, _ = _verify_decomposition(U, V, W, n, m, p)
        if not valid:
            continue
        
        # Try to reduce rank
        U, V, W = _greedy_rank_reduce(U, V, W, n, m, p, deadline - 10)
        
        if U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return best_U, best_V, best_W

def _schoolbook_decomp(n, m, p):
    """Pure schoolbook decomposition."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))
    return _terms_to_matrices(terms, n, m, p)

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 5, 5
    
    # First try systematic blocking search
    U, V, W = _try_all_blockings(n, m, p, deadline)
    
    if U is None:
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomp(n, m, p)
    
    # Final verification
    valid, err = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomp(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
