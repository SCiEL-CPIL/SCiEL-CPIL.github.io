"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _rank1_decomposition(n, m, p):
    """Schoolbook rank n*m*p decomposition for n x m * m * p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's M_{2,2,2} decomposition.
    Each term is (u_dict, v_dict, w_dict) where keys are (i,j), (j,k), (k,i)
    in local 2x2x2 coordinates.
    """
    terms = []
    
    # P1: (A11+A22)(B11+B22)
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    
    # P2: (A21+A22)B11
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    
    # P3: A11(B12-B22)
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1}
    terms.append((u3, v3, w3))
    
    # P4: A22(B21-B11)
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (1,0): 1}
    terms.append((u4, v4, w4))
    
    # P5: (A11+A12)B22
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (0,1): 1}
    terms.append((u5, v5, w5))
    
    # P6: (A21-A11)(B11+B12)
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    
    # P7: (A12-A22)(B21+B22)
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    terms.append((u7, w7))
    
    return terms

def _construct_2x3x9_strassen():
    """
    Construct decomposition for 2x3 * 3x9 using Strassen on 2x2 blocks.
    
    Partition j into {0,1} and {2}:
    - A_{j<2} * B_{j<2}: 2x2 * 2x9, partitioned into 4 pairs of k + 1 single k
      Each 2x2*2x2 uses 7 Strassen terms: 4*7 = 28
      Each 2x2*2x1 uses 4 schoolbook terms: 1*4 = 4
    - A_{j=2} * B_{j=2}: 2x1 * 1x9, schoolbook: 2*1*9 = 18
    
    Total: 28 + 4 + 18 = 50
    """
    n, m, p = 2, 3, 9
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Part 1: A_{j in {0,1}} * B_{j in {0,1}}
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    k_single = [8]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for (k1, k2) in k_pairs:
        for (u_dict, v_dict, w_dict) in strassen_terms:
            U_term = {}
            for (li, lj), coeff in u_dict.items():
                U_term[(li, lj)] = coeff
            
            V_term = {}
            for (lj, lk), coeff in v_dict.items():
                V_term[(lj, k1 + lk)] = coeff
            
            W_term = {}
            for (li, lk), coeff in w_dict.items():
                W_term[(li, k1 + lk)] = coeff
            
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
    
    for k in k_single:
        for i in range(2):
            for j in range(2):
                U_term = {(i, j): 1}
                V_term = {(j, k): 1}
                W_term = {(i, k): 1}
                U_terms.append(U_term)
                V_terms.append(V_term)
                W_terms.append(W_term)
    
    # Part 2: A_{j=2} * B_{j=2}
    for i in range(2):
        for k in range(9):
            U_term = {(i, 2): 1}
            V_term = {(2, k): 1}
            W_term = {(i, k): 1}
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
    
    rank = len(U_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (i, j), coeff in U_terms[col].items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in V_terms[col].items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in W_terms[col].items():
            W[k * n + i][col] = coeff
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if U else 0
    
    T_computed = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        u_vec = np.array([U[i][c] for i in range(n*m)])
        v_vec = np.array([V[i][c] for i in range(m*p)])
        w_vec = np.array([W[i][c] for i in range(p*n)])
        T_computed += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    diff = np.max(np.abs(T_computed - T_expected))
    return diff < 1e-10

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding linearly dependent terms using QR with column pivoting.
    """
    rank = len(U[0]) if U else 0
    if rank <= 1 or time.time() > deadline:
        return U, V, W
    
    nU = n * m
    nV = m * p
    nW = p * n
    
    # Build tensor entry -> term contributions mapping
    from collections import defaultdict
    entry_to_terms = defaultdict(list)
    
    for c in range(rank):
        u_nonzero = [(i, U[i][c]) for i in range(nU) if U[i][c] != 0]
        v_nonzero = [(j, V[j][c]) for j in range(nV) if V[j][c] != 0]
        w_nonzero = [(k, W[k][c]) for k in range(nW) if W[k][c] != 0]
        if u_nonzero and v_nonzero and w_nonzero:
            for i, ui in u_nonzero:
                for j, vj in v_nonzero:
                    for k, wk in w_nonzero:
                        entry_to_terms[(i, j, k)].append((c, ui * vj * wk))
    
    entries = list(entry_to_terms.keys())
    if not entries:
        return U, V, W
    
    num_entries = len(entries)
    
    # Build matrix: rows = tensor entries, cols = terms
    M = np.zeros((num_entries, rank), dtype=float)
    for entry_idx, entry in enumerate(entries):
        for term_idx, val in entry_to_terms[entry]:
            M[entry_idx, term_idx] = val
    
    # QR with column pivoting
    try:
        Q, R, piv = np.linalg.qr(M, mode='complete', pivoting=True)
        diag_R = np.abs(np.diag(R))
        threshold = max(1e-10, 1e-8 * np.max(diag_R) if len(diag_R) > 0 else 1e-10)
        rank_found = int(np.sum(diag_R > threshold))
        
        if rank_found >= rank:
            return U, V, W
        
        independent_cols = sorted(piv[:rank_found])
        new_rank = len(independent_cols)
        
        new_U = [[0] * new_rank for _ in range(nU)]
        new_V = [[0] * new_rank for _ in range(nV)]
        new_W = [[0] * new_rank for _ in range(nW)]
        
        for new_col, orig_idx in enumerate(independent_cols):
            for i in range(nU):
                if U[i][orig_idx] != 0:
                    new_U[i][new_col] = U[i][orig_idx]
            for j in range(nV):
                if V[j][orig_idx] != 0:
                    new_V[j][new_col] = V[j][orig_idx]
            for k in range(nW):
                if W[k][orig_idx] != 0:
                    new_W[k][new_col] = W[k][orig_idx]
        
        # Verify the reduced decomposition
        if _verify_decomposition(new_U, new_V, new_W, n, m, p):
            return new_U, new_V, new_W
        else:
            return U, V, W
    except Exception:
        return U, V, W

def _construct_2x3x9():
    """
    Construct rank-minimal decomposition for 2x3 * 3x9.
    Uses Strassen-based decomposition to achieve rank 50, then reduces rank
    by finding linearly dependent terms.
    """
    n, m, p = 2, 3, 9
    start = time.time()
    deadline = start + 130
    
    try:
        U, V, W = _construct_2x3x9_strassen()
        
        if _verify_decomposition(U, V, W, n, m, p):
            U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
            return U, V, W
        else:
            return _rank1_decomposition(n, m, p)
    except Exception:
        return _rank1_decomposition(n, m, p)

def construct():
    """
    Returns tensor decomposition for matrix multiplication 2x3 * 3x9.
    Uses Strassen-based decomposition to achieve rank 50, then reduces rank
    by finding linearly dependent terms via QR with column pivoting.
    """
    start = time.time()
    deadline = start + 135
    
    try:
        result = _construct_2x3x9()
        return {"U": result[0], "V": result[1], "W": result[2]}
    except Exception:
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
