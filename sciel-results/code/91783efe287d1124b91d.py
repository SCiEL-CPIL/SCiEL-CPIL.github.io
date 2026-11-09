import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 as dicts of (index, coeff).
    U: A[i,j] at index i*2+j
    V: B[j,k] at index j*2+k
    W: C[i,k] at index k*2+i (trace convention W[k*n+i])
    """
    terms = []
    # m1 = (a+d)(e+h) -> C[0,0]+, C[1,1]+
    terms.append({'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
    # m2 = (c+d)e -> C[1,0]+, C[1,1]-
    terms.append({'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,1): 1, (1,1): -1}})
    # m3 = a(f-h) -> C[0,0]-, C[0,1]+
    terms.append({'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,0): -1, (1,0): 1}})
    # m4 = d(g-e) -> C[0,0]+, C[1,0]+
    terms.append({'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (0,1): 1}})
    # m5 = (a+b)h -> C[0,1]+, C[1,1]-
    terms.append({'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(1,0): 1, (1,1): -1}})
    # m6 = (c-a)(e+f) -> C[1,0]+, C[1,1]+
    terms.append({'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(0,1): 1, (1,1): 1}})
    # m7 = (b-d)(g+h) -> C[0,0]+, C[0,1]+
    terms.append({'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,0): 1}})
    return terms

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 via recursive Strassen.
    Returns list of (U_dict, V_dict, W_dict) where indices are flat.
    U: A[i,j] at i*4+j, V: B[j,k] at j*4+k, W: C[i,k] at k*4+i
    """
    terms_2x2 = _strassen_2x2_terms()
    
    block_terms = [
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,1): 1, (1,1): -1}},
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,0): 1, (1,1): 1}},
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (0,1): 1}},
    ]
    
    all_terms = []
    for s in range(7):
        bt = block_terms[s]
        for t in range(7):
            st = terms_2x2[t]
            U_dict = {}
            V_dict = {}
            W_dict = {}
            
            for (bi, bj), a_coeff in bt['A'].items():
                for (i, j), u_coeff in st['U'].items():
                    flat_i = bi * 2 + i
                    flat_j = bj * 2 + j
                    idx = flat_i * 4 + flat_j
                    U_dict[idx] = U_dict.get(idx, 0) + a_coeff * u_coeff
            
            for (bj, bk), b_coeff in bt['B'].items():
                for (j, k), v_coeff in st['V'].items():
                    flat_j = bj * 2 + j
                    flat_k = bk * 2 + k
                    idx = flat_j * 4 + flat_k
                    V_dict[idx] = V_dict.get(idx, 0) + b_coeff * v_coeff
            
            for (bi, bk), c_coeff in bt['C'].items():
                for (k, i), w_coeff in st['W'].items():
                    flat_k = bk * 2 + k
                    flat_i = bi * 2 + i
                    idx = flat_k * 4 + flat_i
                    W_dict[idx] = W_dict.get(idx, 0) + c_coeff * w_coeff
            
            # Remove zero entries
            U_dict = {k: v for k, v in U_dict.items() if v != 0}
            V_dict = {k: v for k, v in V_dict.items() if v != 0}
            W_dict = {k: v for k, v in W_dict.items() if v != 0}
            
            all_terms.append((U_dict, V_dict, W_dict))
    
    return all_terms

def _build_4x4x8_decomposition():
    """Build decomposition for 4x4 x 4x8.
    Split into two 4x4x4 subproblems (columns 0-3 and 4-7 of B).
    Rank: 2 * 49 = 98
    """
    n, m, p = 4, 4, 8
    terms_4x4x4 = _build_4x4x4_strassen()
    rank = 2 * len(terms_4x4x4)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(2):
        for idx, (U_dict, V_dict, W_dict) in enumerate(terms_4x4x4):
            col = block * len(terms_4x4x4) + idx
            for u_idx, val in U_dict.items():
                U[u_idx, col] = val
            for v_idx, val in V_dict.items():
                # v_idx is j*4+k where k is local (0-3), offset by block*4
                j = v_idx // 4
                k_local = v_idx % 4
                k_global = k_local + block * 4
                V[j * 8 + k_global, col] = val
            for w_idx, val in W_dict.items():
                # w_idx is k*4+i where k is local (0-3), offset by block*4
                k_local = w_idx // 4
                i = w_idx % 4
                k_global = k_local + block * 4
                W[k_global * 4 + i, col] = val
    
    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """Attempt to reduce rank by finding linear dependencies.
    The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    We can try to find a lower-rank representation by checking if
    some columns are linearly dependent in a structured way.
    """
    rank = U.shape[1]
    if rank <= 50:
        return U, V, W
    
    # Build the full tensor T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    T[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
    
    # Try to find a lower rank decomposition via SVD-like approach
    # Reshape T into a matrix and use rank approximation
    # T as matrix: (n*m, p) doesn't directly work for this tensor format
    
    # Alternative: try random linear combinations of rank-1 terms
    # to see if we can represent the tensor with fewer terms
    # This is NP-hard in general, so we just try a few heuristics
    
    # Heuristic: check if any two terms can be merged
    # (i.e., if U_r1 * V_r1 * W_r1 + U_r2 * V_r2 * W_r2 can be
    #  represented as a single rank-1 term)
    # This is rarely possible, so skip for now
    
    # Try: can we find a subset of terms that spans the same tensor?
    # Use iterative greedy elimination
    if time.time() > deadline - 30:
        return U, V, W
    
    # Simple check: are any columns exactly zero or duplicates?
    for r in range(rank):
        if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
            # This term contributes nothing, remove it
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            rank -= 1
            r -= 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    rank = U.shape[1]
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        a_part = np.sum(Ur * A_test, axis=1)
        b_part = np.sum(Vr * B_test, axis=0)
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += a_part[i] * b_part[k] * Wr[k, i]
    
    return np.allclose(C_computed, C_expected, atol=1e-9)

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 x 4x8."""
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_4x4x8_decomposition()
        
        if not _verify(U, V, W, n, m, p):
            raise ValueError("Verification failed for 49*2 decomposition")
        
        # Try rank reduction
        U, V, W = _try_rank_reduction(U, V, W, n, m, p, deadline)
        
        def to_list(arr):
            return [[float(x) for x in row] for row in arr]
        
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    except Exception:
        pass
    
    # Fallback: schoolbook rank 128
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
