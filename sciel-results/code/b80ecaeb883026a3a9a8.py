import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-124 decomposition for 5x5x5 matrix multiplication.
    Uses Strassen (rank 7) for 2x2 blocks and Schoolbook for others.
    """
    n, m, p = DIMENSIONS
    # We use a 2+3 split
    n1, n2 = 2, 3
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    
    # Global dimensions
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # We will build the decomposition term by term.
    # Each term is a tuple (u_vec, v_vec, w_vec)
    terms = []
    
    # Helper to add a term to the list
    def add_term(u, v, w):
        # u, v, w are lists of length dim_U, dim_V, dim_W
        terms.append((u, v, w))
        
    # Helper for Schoolbook decomposition of m x n * n x p -> m x p
    def schoolbook(m, n, p, u_offset, v_offset, w_offset):
        # u_offset: index in U where this block starts
        # v_offset: index in V where this block starts
        # w_offset: index in W where this block starts
        # Returns list of (u_local, v_local, w_local)
        local_terms = []
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    u_local = [0] * (m * n)
                    v_local = [0] * (n * p)
                    w_local = [0] * (p * m)
                    
                    u_local[i * n + j] = 1
                    v_local[j * p + k] = 1
                    # W index convention: W[k*n+i] in global?
                    # In local block, C is m x p.
                    # The global W index is w_offset + k * m + i ?
                    # Let's check global W index: W[k*n + i].
                    # Here k is local k, i is local i.
                    # Global k = k_offset + k_local
                    # Global i = i_offset + i_local
                    # So global index = (k_offset + k) * n + (i_offset + i)
                    # This is NOT simply w_offset + k*m + i.
                    # We need to pass the offsets for k and i separately to W.
                    
                    # Let's refactor add_term to take global indices directly.
                    pass
                    
    # Refactored approach:
    # Define global indices for each block.
    
    # Block A22: i in 0..1, j in 0..1. U rows 0..3.
    # Block A23: i in 0..1, j in 2..4. U rows 4..9.
    # Block A32: i in 2..4, j in 0..1. U rows 10..17.
    # Block A33: i in 2..4, j in 2..4. U rows 18..24.
    
    # Block B22: j in 0..1, k in 0..1. V rows 0..3.
    # Block B23: j in 0..1, k in 2..4. V rows 4..9.
    # Block B32: j in 2..4, k in 0..1. V rows 10..17.
    # Block B33: j in 2..4, k in 2..4. V rows 18..24.
    
    # Block C22: i in 0..1, k in 0..1. W indices: k*5 + i.
    # Block C23: i in 0..1, k in 2..4.
    # Block C32: i in 2..4, k in 0..1.
    # Block C33: i in 2..4, k in 2..4.
    
    # We will generate terms for each block product.
    
    # 1. A22 * B22 -> C22 (2x2 * 2x2 -> 2x2)
    # Use Strassen
    # Local indices:
    # A: 0,1,2,3 (i,j) -> 0,1,2,3
    # B: 0,1,2,3 (j,k) -> 0,1,2,3
    # C: 0,1,2,3 (k,i) -> k*2+i
    
    # Strassen terms (u, v, w) in local 2x2 coords
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0])
    ]
    
    # Map local to global
    for u_l, v_l, w_l in strassen_terms:
        u_g = [0] * 25
        v_g = [0] * 25
        w_g = [0] * 25
        
        # U: A22 rows 0..3
        for idx, val in enumerate(u_l):
            if val != 0:
                u_g[idx] = val
                
        # V: B22 rows 0..3
        for idx, val in enumerate(v_l):
            if val != 0:
                v_g[idx] = val
                
        # W: C22. Local C index k*2+i. Global W index k*5+i.
        # k is 0..1, i is 0..1.
        for idx, val in enumerate(w_l):
            if val != 0:
                k_l = idx // 2
                i_l = idx % 2
                global_w_idx = k_l * 5 + i_l
                w_g[global_w_idx] = val
                
        terms.append((u_g, v_g, w_g))
        
    # 2. A23 * B32 -> C22 (2x3 * 3x2 -> 2x2)
    # Schoolbook
    # A23: i in 0..1, j in 2..4. U rows 4..9.
    # B32: j in 2..4, k in 0..1. V rows 10..17.
    # C22: i in 0..1, k in 0..1. W indices k*5+i.
    
    for i in range(2):
        for j in range(3):
            for k in range(2):
                u_g = [0] * 25
                v_g = [0] * 25
                w_g = [0] * 25
                
                # U index: i*5 + (j+2)
                u_idx = i * 5 + (j + 2)
                u_g[u_idx] = 1
                
                # V index: (j+2)*5 + k
                v_idx = (j + 2) * 5 + k
                v_g[v_idx] = 1
                
                # W index: k*5 + i
                w_idx = k * 5 + i
                w_g[w_idx] = 1
                
                terms.append((u_g, v_g, w_g))
                
    # 3. A22 * B23 -> C23 (2x2 * 2x3 -> 2x3)
    # Schoolbook
    # A22: i in 0..1, j in 0..1. U rows 0..3.
    # B23: j in 0..1, k in 2..4. V rows 4..9.
    # C23: i in 0..1, k in 2..4. W indices k*5+i.
    
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_g = [0] * 25
                v_g = [0] * 25
                w_g = [0] * 25
                
                # U index: i*5 + j
                u_idx = i * 5 + j
                u_g[u_idx] = 1
                
                # V index: j*5 + (k+2)
                v_idx = j * 5 + (k + 2)
                v_g[v_idx] = 1
                
                # W index: (k+2)*5 + i
                w_idx = (k + 2) * 5 + i
                w_g[w_idx] = 1
                
                terms.append((u_g, v_g, w_g))
                
    # 4. A23 * B33 -> C23 (2x3 * 3x3 -> 2x3)
    # Schoolbook
    # A23: i in 0..1, j in 2..4. U rows 4..9.
    # B33: j in 2..4, k in 2..4. V rows 18..24.
    # C23: i in 0..1, k in 2..4. W indices k*5+i.
    
    for i in range(2):
        for j in range(3):
            for k in range(3):
                u_g = [0] * 25
                v_g = [0] * 25
                w_g = [0] * 25
                
                # U index: i*5 + (j+2)
                u_idx = i * 5 + (j + 2)
                u_g[u_idx] = 1
                
                # V index: (j+2)*5 + (k+2)
                v_idx = (j + 2) * 5 + (k + 2)
                v_g[v_idx] = 1
                
                # W index: (k+2)*5 + i
                w_idx = (k + 2) * 5 + i
                w_g[w_idx] = 1
                
                terms.append((u_g, v_g, w_g))
                
    # 5. A32 * B22 -> C32 (3x2 * 2x2 -> 3x2)
    # Schoolbook
    # A32: i in 2..4, j in 0..1. U rows 10..17.
    # B22: j in 0..1, k in 0..1. V rows 0..3.
    # C32: i in 2..4, k in 0..1. W indices k*5+i.
    
    for i in range(3):
        for j in range(2):
            for k in range(2):
                u_g = [0] * 25
                v_g = [0] * 25
                w_g = [0] * 25
                
                # U index: (i+2)*5 + j
                u_idx = (i + 2) * 5 + j
                u_g[u_idx] = 1
                
                # V index: j*5 + k
                v_idx = j * 5 + k
                v_g[v_idx] = 1
                
                # W index: k*5 + (i+2)
                w_idx = k * 5 + (i + 2)
                w_g[w_idx] = 1
                
                terms.append((u_g, v_g, w_g))
                
    # 6. A33 * B32 -> C32 (3x3 * 3x2 -> 3x2)
    # Schoolbook
    # A33: i in 2..4, j in 2..4. U rows 18..24.
    # B32: j in 2..4, k in 0..1. V rows 10..17.
    # C32: i in 2..4, k in 0..1. W indices k*5+i.
    
    for i in range(3):
        for j in range(3):
            for k in range(2):
                u_g = [0] * 25
                v_g = [0] * 25
                w_g = [0] * 25
                
                # U index: (i+2)*5 + (j+2)
                u_idx = (i + 2) * 5 + (j + 2)
                u_g[u_idx] = 1
                
                # V index: (j+2)*5 + k
                v_idx = (j + 2) * 5 + k
                v_g[v_idx] = 1
                
                # W index: k*5 + (i+2)
                w_idx = k * 5 + (i + 2)
                w_g[w_idx] = 1
                
                terms.append((u_g, v_g, w_g))
                
    # 7. A32 * B23 -> C33 (3x2 * 2x3 -> 3x3)
    # Schoolbook
    # A32: i in 2..4, j in 0..1. U rows 10..17.
    # B23: j in 0..1, k in 2..4. V rows 4..9.
    # C33: i in 2..4, k in 2..4. W indices k*5+i.
    
    for i in range(3):
        for j in range(2):
            for k in range(3):
                u_g = [0] * 25
                v_g = [0] * 25
                w_g = [0] * 25
                
                # U index: (i+2)*5 + j
                u_idx = (i + 2) * 5 + j
                u_g[u_idx] = 1
                
                # V index: j*5 + (k+2)
                v_idx = j * 5 + (k + 2)
                v_g[v_idx] = 1
                
                # W index: (k+2)*5 + (i+2)
                w_idx = (k + 2) * 5 + (i + 2)
                w_g[w_idx] = 1
                
                terms.append((u_g, v_g, w_g))
                
    # 8. A33 * B33 -> C33 (3x3 * 3x3 -> 3x3)
    # Schoolbook
    # A33: i in 2..4, j in 2..4. U rows 18..24.
    # B33: j in 2..4, k in 2..4. V rows 18..24.
    # C33: i in 2..4, k in 2..4. W indices k*5+i.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_g = [0] * 25
                v_g = [0] * 25
                w_g = [0] * 25
                
                # U index: (i+2)*5 + (j+2)
                u_idx = (i + 2) * 5 + (j + 2)
                u_g[u_idx] = 1
                
                # V index: (j+2)*5 + (k+2)
                v_idx = (j + 2) * 5 + (k + 2)
                v_g[v_idx] = 1
                
                # W index: (k+2)*5 + (i+2)
                w_idx = (k + 2) * 5 + (i + 2)
                w_g[w_idx] = 1
                
                terms.append((u_g, v_g, w_g))
                
    rank = len(terms)
    
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    for t, (u, v, w) in enumerate(terms):
        for i in range(25):
            U[i][t] = u[i]
            V[i][t] = v[i]
            W[i][t] = w[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
