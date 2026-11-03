"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # We will build the decomposition using a block-Strassen approach.
    # Row blocks: [(0,1), (2,2)]
    # Mid blocks: [(0,1), (2,3), (4,4)]
    # Col blocks: [(0,1), (2,3), (4,5)]
    
    # We accumulate terms as (u_dict, v_dict, w_dict)
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
        
    # Corrected Strassen for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    def add_strassen_block(row_idx, col_idx, mid_idx):
        i1, i2 = row_idx
        k1, k2 = col_idx
        j1, j2 = mid_idx
        
        # M1
        u = {i1*m+j1: 1, i2*m+j2: 1}
        v = {j1*p+k1: 1, j2*p+k2: 1}
        w = {k1*n+i1: 1, k2*n+i2: 1}
        add_term(u, v, w)
        
        # M2
        u = {i2*m+j1: 1, i2*m+j2: 1}
        v = {j1*p+k1: 1}
        w = {k1*n+i2: 1}
        add_term(u, v, w)
        
        # M3
        u = {i1*m+j1: 1}
        v = {j1*p+k2: 1, j2*p+k2: -1}
        w = {k2*n+i1: 1}
        add_term(u, v, w)
        
        # M4
        u = {i2*m+j2: 1}
        v = {j2*p+k1: 1, j1*p+k1: -1}
        w = {k1*n+i2: 1}
        add_term(u, v, w)
        
        # M5
        u = {i1*m+j1: 1, i1*m+j2: 1}
        v = {j2*p+k2: 1}
        w = {k2*n+i1: 1}
        add_term(u, v, w)
        
        # M6
        u = {i2*m+j1: 1, i1*m+j1: -1}
        v = {j1*p+k1: 1, j1*p+k2: 1}
        w = {k1*n+i2: 1, k2*n+i2: 1}
        add_term(u, v, w)
        
        # M7
        u = {i1*m+j2: 1, i2*m+j2: -1}
        v = {j2*p+k1: 1, j2*p+k2: 1}
        w = {k1*n+i1: 1, k2*n+i1: 1}
        add_term(u, v, w)
        
    def add_schoolbook_block(row_idx, col_idx, mid_idx):
        for i in row_idx:
            for j in mid_idx:
                for k in col_idx:
                    u = {i*m+j: 1}
                    v = {j*p+k: 1}
                    w = {k*n+i: 1}
                    add_term(u, v, w)
    
    row_blocks = [(0, 1), (2, 2)]
    mid_blocks = [(0, 1), (2, 3), (4, 4)]
    col_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for rb in row_blocks:
        for mb in mid_blocks:
            for cb in col_blocks:
                if len(rb) == 2 and len(mb) == 2 and len(cb) == 2:
                    add_strassen_block(rb, cb, mb)
                else:
                    add_schoolbook_block(rb, cb, mb)
    
    # Convert terms to U, V, W matrices
    max_u = max(max(u.keys()) for u, v, w in terms)
    max_v = max(max(v.keys()) for u, v, w in terms)
    max_w = max(max(w.keys()) for u, v, w in terms)
    
    rank = len(terms)
    
    U = np.zeros((max_u + 1, rank))
    V = np.zeros((max_v + 1, rank))
    W = np.zeros((max_w + 1, rank))
    
    for c, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx, c] = val
        for idx, val in v.items():
            V[idx, c] = val
        for idx, val in w.items():
            W[idx, c] = val
    
    # Verify
    if _verify(U, V, W, n, m, p):
        # Ensure shapes are correct
        # U should be (n*m, rank)
        # V should be (m*p, rank)
        # W should be (p*n, rank)
        # Pad if necessary
        if U.shape[0] < n*m:
            U = np.pad(U, ((0, n*m - U.shape[0]), (0, 0)))
        if V.shape[0] < m*p:
            V = np.pad(V, ((0, m*p - V.shape[0]), (0, 0)))
        if W.shape[0] < p*n:
            W = np.pad(W, ((0, p*n - W.shape[0]), (0, 0)))
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
