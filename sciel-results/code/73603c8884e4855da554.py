import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        add_rank1(u_vec, v_vec, w_vec)
    
    # Part 1: 4x4x4 multiplication using Strassen (7 outer x 7 inner = 49 terms)
    strassen_products = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}},
        {'A': {2: 1, 3: 1}, 'B': {0: 1}},
        {'A': {0: 1}, 'B': {1: 1, 3: -1}},
        {'A': {3: 1}, 'B': {2: 1, 0: -1}},
        {'A': {0: 1, 1: 1}, 'B': {3: 1}},
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}},
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}},
    ]
    
    c_contributions = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]
    
    def a_global(block, li, lk):
        if block == 0: return li, lk
        elif block == 1: return li, lk + 2
        elif block == 2: return li + 2, lk
        elif block == 3: return li + 2, lk + 2
    
    def b_global(block, lk, lj):
        if block == 0: return lk, lj
        elif block == 1: return lk, lj + 2
        elif block == 2: return lk + 2, lj
        elif block == 3: return lk + 2, lj + 2
    
    def c_global(block, li, lj):
        if block == 0: return li, lj
        elif block == 1: return li, lj + 2
        elif block == 2: return li + 2, lj
        elif block == 3: return li + 2, lj + 2
    
    inner_strassen = [
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
        {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
    ]
    
    for prod_idx, prod in enumerate(strassen_products):
        a_form = prod['A']
        b_form = prod['B']
        c_contrib = c_contributions[prod_idx]
        
        for inner_idx, inner in enumerate(inner_strassen):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            for a_block, a_sign in a_form.items():
                for li, lk, inner_sign in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign
            
            for b_block, b_sign in b_form.items():
                for lk, lj, inner_sign in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign
            
            for c_block, c_sign in c_contrib.items():
                for li, lj, inner_sign in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign
            
            add_rank1(u_vec, v_vec, w_vec)
    
    # Part 2: 4x4x1 (k in 0..3, j=4) - schoolbook
    for i in range(4):
        for k in range(4):
            if time.time() - _START > _DEADLINE - 10:
                break
            add_term(i, k, 4, 1.0, 1.0, 1.0)
    
    # Part 3: 4x1x5 (k=4, j in 0..4) - schoolbook
    for i in range(4):
        for j in range(5):
            if time.time() - _START > _DEADLINE - 10:
                break
            add_term(i, 4, j, 1.0, 1.0, 1.0)
    
    rank = len(U_cols)
    
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1
    
    # Aggressive null-space reduction: iteratively remove linearly dependent terms
    if len(U_cols) > 1 and time.time() - _START < _DEADLINE - 20:
        # Precompute test matrices for verification
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        def verify_decomposition_fast(Uc, Vc, Wc, atol=1e-6):
            """Vectorized verification."""
            num_terms = len(Uc)
            if num_terms == 0:
                return np.allclose(np.zeros((n, p)), C_expected, atol=atol)
            
            U_mat = np.array(Uc, dtype=float).T  # (n*m, rank)
            V_mat = np.array(Vc, dtype=float).T  # (m*p, rank)
            W_mat = np.array(Wc, dtype=float).T  # (p*n, rank)
            
            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(p):
                    u_slice = U_mat[i*m:(i+1)*m, :]  # (m, rank)
                    v_indices = np.arange(m) * p + j
                    v_slice = V_mat[v_indices, :]  # (m, rank)
                    w_val = W_mat[j*n + i, :]  # (rank,)
                    C_computed[i, j] = np.sum(u_slice * v_slice * w_val[np.newaxis, :])
            
            return np.allclose(C_computed, C_expected, atol=atol)
        
        # Iteratively find and remove null-space vectors
        max_iterations = 30
        for iteration in range(max_iterations):
            if time.time() - _START > _DEADLINE - 10:
                break
            if len(U_cols) <= 1:
                break
            
            U_mat = np.array(U_cols, dtype=float).T  # (n*m, rank)
            
            # Use U matrix for null-space detection
            s_vals = np.linalg.svd(U_mat, compute_uv=False)
            tol = 1e-9
            num_indep = int(np.sum(s_vals > tol))
            
            if num_indep >= len(U_cols):
                break
            
            # Get null-space vectors from U
            U_svd, s, Vt = np.linalg.svd(U_mat, full_matrices=True)
            null_vecs = Vt[num_indep:]
            
            found_removal = False
            for ns_vec in null_vecs:
                if time.time() - _START > _DEADLINE - 10:
                    break
                
                # Try removing the term with largest coefficient in null vector
                drop_idx = int(np.argmax(np.abs(ns_vec)))
                if abs(ns_vec[drop_idx]) < 1e-10:
                    continue
                
                # Build candidate with removed term
                new_U_cols = [U_cols[i] for i in range(len(U_cols)) if i != drop_idx]
                new_V_cols = [V_cols[i] for i in range(len(V_cols)) if i != drop_idx]
                new_W_cols = [W_cols[i] for i in range(len(W_cols)) if i != drop_idx]
                
                # Verify correctness
                if verify_decomposition_fast(new_U_cols, new_V_cols, new_W_cols):
                    U_cols = new_U_cols
                    V_cols = new_V_cols
                    W_cols = new_W_cols
                    found_removal = True
                    break
            
            if not found_removal:
                break
    
    U = np.stack(U_cols, axis=1)
    V = np.stack(V_cols, axis=1)
    W = np.stack(W_cols, axis=1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
