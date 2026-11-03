"""Matrix multiplication 3,4,7 - rank minimization via block decomposition."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _strassen_2x2_local():
    """Return the 7 Strassen rank-1 terms for 2x2x2 as sparse (positions, values) triples.
    Each term: (list of (pos,val) for U, list for V, list for W) with local indices.
    Local: u[i*2+j], v[j*2+k], w[k*2+i]
    """
    terms = []
    # M1: (a11+a22)(b11+b22)
    terms.append(([ (0,1.0),(3,1.0) ], [ (0,1.0),(3,1.0) ], [ (0,1.0),(3,1.0) ]))
    # M2: (a21+a22)b11
    terms.append(([ (2,1.0),(3,1.0) ], [ (0,1.0) ], [ (1,1.0),(3,-1.0) ]))
    # M3: a11(b12-b22)
    terms.append(([ (0,1.0) ], [ (1,1.0),(3,-1.0) ], [ (2,1.0),(3,1.0) ]))
    # M4: a22(b21-b11)
    terms.append(([ (3,1.0) ], [ (2,1.0),(0,-1.0) ], [ (0,1.0),(1,1.0) ]))
    # M5: (a11+a12)b22
    terms.append(([ (0,1.0),(1,1.0) ], [ (3,1.0) ], [ (0,-1.0),(2,1.0) ]))
    # M6: (a21-a11)(b11+b12)
    terms.append(([ (2,1.0),(0,-1.0) ], [ (0,1.0),(1,1.0) ], [ (3,1.0) ]))
    # M7: (a12-a22)(b21+b22)
    terms.append(([ (1,1.0),(3,-1.0) ], [ (2,1.0),(3,1.0) ], [ (0,1.0) ]))
    return terms

def _schoolbook_sparse(n, m, p):
    """Return schoolbook terms as sparse (pos, val) triples for the full n x m x p problem."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(([ (i*m+j, 1.0) ], [ (j*p+k, 1.0) ], [ (k*n+i, 1.0) ]))
    return terms

def _expand_sparse_term(term_sparse, n, m, p, i0, j0, k0):
    """Expand a sparse local term to global indices.
    term_sparse: (u_pairs, v_pairs, w_pairs) where each pair is (local_pos, val)
    i0, j0, k0: offsets for the block.
    Local: u[i*m+j], v[j*p+k], w[k*n+i]
    Global: u[(i0+i)*m + (j0+j)], v[(j0+j)*p + (k0+k)], w[(k0+k)*n + (i0+i)]
    """
    u_pairs, v_pairs, w_pairs = term_sparse
    u_global = []
    v_global = []
    w_global = []
    for (pos, val) in u_pairs:
        i = pos // m
        j = pos % m
        gpos = (i0 + i) * m + (j0 + j)
        u_global.append((gpos, val))
    for (pos, val) in v_pairs:
        j = pos // p
        k = pos % p
        gpos = (j0 + j) * p + (k0 + k)
        v_global.append((gpos, val))
    for (pos, val) in w_pairs:
        k = pos // n
        i = pos % n
        gpos = (k0 + k) * n + (i0 + i)
        w_global.append((gpos, val))
    return (u_global, v_global, w_global)

def _sparsify_strassen_block(n_local, m_local, p_local, i0, j0, k0, n, m, p):
    """Get Strassen terms for a 2x2x2 block at offset (i0,j0,k0)."""
    if n_local != 2 or m_local != 2 or p_local != 2:
        return _schoolbook_sparse(n_local, m_local, p_local)
    local_terms = _strassen_2x2_local()
    global_terms = []
    for t in local_terms:
        global_terms.append(_expand_sparse_term(t, n_local, m_local, p_local, i0, j0, k0))
    return global_terms

def _schoolbook_block(n_local, m_local, p_local, i0, j0, k0, n, m, p):
    """Get schoolbook terms for a block at offset (i0,j0,k0)."""
    local_terms = _schoolbook_sparse(n_local, m_local, p_local)
    global_terms = []
    for t in local_terms:
        global_terms.append(_expand_sparse_term(t, n_local, m_local, p_local, i0, j0, k0))
    return global_terms

def _assemble_terms(all_sparse_terms, n, m, p):
    """Convert list of sparse terms to dense U, V, W matrices."""
    rank = len(all_sparse_terms)
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    for r, (u_pairs, v_pairs, w_pairs) in enumerate(all_sparse_terms):
        for (pos, val) in u_pairs:
            U[pos, r] = val
        for (pos, val) in v_pairs:
            V[pos, r] = val
        for (pos, val) in w_pairs:
            W[pos, r] = val
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    # The multiplication tensor T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    # should equal 1 if u=i*m+j, v=j*p+k, w=k*n+i for valid (i,j,k), else 0.
    # Check a few random cases
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[u, r] * V[v, r] * W[w, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    # Check that non-matching indices give 0
    import itertools
    for (u, v, w) in itertools.product(range(n*m), range(m*p), range(p*n)):
        i_u = u // m
        j_u = u % m
        j_v = v // p
        k_v = v % p
        k_w = w // n
        i_w = w % n
        # Valid if j_u == j_v and i_u == i_w and k_v == k_w
        if j_u == j_v and i_u == i_w and k_v == k_w:
            continue  # Already checked
        val = 0.0
        for r in range(U.shape[1]):
            val += U[u, r] * V[v, r] * W[w, r]
        if abs(val) > 1e-10:
            return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    all_terms = []
    
    # Block decomposition:
    # Split A rows: [0:2], [2:3]
    # Split B cols: [0:4], [4:7]
    # Split A cols: [0:2], [2:4]
    # Split B rows: [0:2], [2:4]
    
    # Block (0,0): A[0:2, :] (2x4) * B[:, 0:4] (4x4) -> C[0:2, 0:4] (2x4)
    #   Sub-block (0,0,a): A[0:2, 0:2] (2x2) * B[0:2, 0:4] (2x4)
    #     Sub-sub (0,0,a,1): A[0:2, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> Strassen
    #     Sub-sub (0,0,a,2): A[0:2, 0:2] (2x2) * B[0:2, 2:4] (2x2) -> Strassen
    #   Sub-block (0,0,b): A[0:2, 2:4] (2x2) * B[2:4, 0:4] (2x4)
    #     Sub-sub (0,0,b,1): A[0:2, 2:4] (2x2) * B[2:4, 0:2] (2x2) -> Strassen
    #     Sub-sub (0,0,b,2): A[0:2, 2:4] (2x2) * B[2:4, 2:4] (2x2) -> Strassen
    #   Total: 4 * 7 = 28
    
    # Strassen blocks for (0,0):
    all_terms += _sparsify_strassen_block(2, 2, 2, i0=0, j0=0, k0=0, n=n, m=m, p=p)
    all_terms += _sparsify_strassen_block(2, 2, 2, i0=0, j0=0, k0=2, n=n, m=m, p=p)
    all_terms += _sparsify_strassen_block(2, 2, 2, i0=0, j0=2, k0=0, n=n, m=m, p=p)
    all_terms += _sparsify_strassen_block(2, 2, 2, i0=0, j0=2, k0=2, n=n, m=m, p=p)
    
    # Block (0,1): A[0:2, :] (2x4) * B[:, 4:7] (4x3) -> C[0:2, 4:7] (2x3)
    #   Sub-block (0,1,a): A[0:2, 0:2] (2x2) * B[0:2, 4:7] (2x3)
    #     Split B cols 4:7 into 4:5 and 5:7:
    #       (0,1,a,1): 2x2x1 -> schoolbook (4 terms)
    #       (0,1,a,2): 2x2x2 -> Strassen (7 terms)
    #   Sub-block (0,1,b): A[0:2, 2:4] (2x2) * B[2:4, 4:7] (2x3)
    #     Split B cols 4:7 into 4:5 and 5:7:
    #       (0,1,b,1): 2x2x1 -> schoolbook (4 terms)
    #       (0,1,b,2): 2x2x2 -> Strassen (7 terms)
    #   Total: 2*(4+7) = 22
    
    all_terms += _schoolbook_block(2, 2, 1, i0=0, j0=0, k0=4, n=n, m=m, p=p)
    all_terms += _sparsify_strassen_block(2, 2, 2, i0=0, j0=0, k0=5, n=n, m=m, p=p)
    all_terms += _schoolbook_block(2, 2, 1, i0=0, j0=2, k0=4, n=n, m=m, p=p)
    all_terms += _sparsify_strassen_block(2, 2, 2, i0=0, j0=2, k0=5, n=n, m=m, p=p)
    
    # Block (1,0): A[2:3, :] (1x4) * B[:, 0:4] (4x4) -> C[2:3, 0:4] (1x4)
    #   Sub-block (1,0,a): A[2:3, 0:2] (1x2) * B[0:2, 0:4] (2x4)
    #     Split B cols 0:4 into 0:2 and 2:4:
    #       (1,0,a,1): 1x2x2 -> schoolbook (4 terms)
    #       (1,0,a,2): 1x2x2 -> schoolbook (4 terms)
    #   Sub-block (1,0,b): A[2:3, 2:4] (1x2) * B[2:4, 0:4] (2x4)
    #     Split B cols 0:4 into 0:2 and 2:4:
    #       (1,0,b,1): 1x2x2 -> schoolbook (4 terms)
    #       (1,0,b,2): 1x2x2 -> schoolbook (4 terms)
    #   Total: 4*4 = 16
    
    all_terms += _schoolbook_block(1, 2, 2, i0=2, j0=0, k0=0, n=n, m=m, p=p)
    all_terms += _schoolbook_block(1, 2, 2, i0=2, j0=0, k0=2, n=n, m=m, p=p)
    all_terms += _schoolbook_block(1, 2, 2, i0=2, j0=2, k0=0, n=n, m=m, p=p)
    all_terms += _schoolbook_block(1, 2, 2, i0=2, j0=2, k0=2, n=n, m=m, p=p)
    
    # Block (1,1): A[2:3, :] (1x4) * B[:, 4:7] (4x3) -> C[2:3, 4:7] (1x3)
    #   Sub-block (1,1,a): A[2:3, 0:2] (1x2) * B[0:2, 4:7] (2x3)
    #     Split B cols 4:7 into 4:5 and 5:7:
    #       (1,1,a,1): 1x2x1 -> schoolbook (2 terms)
    #       (1,1,a,2): 1x2x2 -> schoolbook (4 terms)
    #   Sub-block (1,1,b): A[2:3, 2:4] (1x2) * B[2:4, 4:7] (2x3)
    #     Split B cols 4:7 into 4:5 and 5:7:
    #       (1,1,b,1): 1x2x1 -> schoolbook (2 terms)
    #       (1,1,b,2): 1x2x2 -> schoolbook (4 terms)
    #   Total: 2*(2+4) = 12
    
    all_terms += _schoolbook_block(1, 2, 1, i0=2, j0=0, k0=4, n=n, m=m, p=p)
    all_terms += _schoolbook_block(1, 2, 2, i0=2, j0=0, k0=5, n=n, m=m, p=p)
    all_terms += _schoolbook_block(1, 2, 1, i0=2, j0=2, k0=4, n=n, m=m, p=p)
    all_terms += _schoolbook_block(1, 2, 2, i0=2, j0=2, k0=5, n=n, m=m, p=p)
    
    # Total rank: 28 + 22 + 16 + 12 = 78
    rank = len(all_terms)
    
    U, V, W = _assemble_terms(all_terms, n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify correctness (quick check)
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _assemble_terms(_schoolbook_sparse(n, m, p), n, m, p)
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
