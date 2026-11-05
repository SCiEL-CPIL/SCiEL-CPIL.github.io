import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(np.array(u_vec, dtype=float))
        V_cols.append(np.array(v_vec, dtype=float))
        W_cols.append(np.array(w_vec, dtype=float))
    
    # --- Part 1: 4x4x4 core via nested Strassen (28 terms) ---
    # This handles i in 0..3, k in 0..3, j in 0..3
    
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
    
    # --- Part 2: k in 0..3, j=4 (4 terms) ---
    # For each k in 0..3, we need: C[i,4] += A[i,k] * B[k,4]
    # This is a rank-1 term per k: u has 1 at (i,k) for all i, v has 1 at (k,4), w has 1 at (4,i) for all i
    for k in range(4):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        for i in range(4):
            u_vec[i * m + k] = 1.0
        v_vec[k * p + 4] = 1.0
        for i in range(4):
            w_vec[4 * n + i] = 1.0
        add_rank1(u_vec, v_vec, w_vec)
    
    # --- Part 3: k=4, j in 0..4 (1 term) ---
    # C[i,j] += A[i,4] * B[4,j]
    # u has 1 at (i,4) for all i, v has 1 at (4,j) for all j, w has 1 at (j,i) for all i,j
    u_vec = np.zeros(n * m)
    v_vec = np.zeros(m * p)
    w_vec = np.zeros(p * n)
    for i in range(4):
        u_vec[i * m + 4] = 1.0
    for j in range(5):
        v_vec[4 * p + j] = 1.0
    for j in range(5):
        for i in range(4):
            w_vec[j * n + i] = 1.0
    add_rank1(u_vec, v_vec, w_vec)
    
    # --- Attempt rank reduction via SVD-based null space search ---
    if time.time() - _START < _DEADLINE - 15:
        def verify_decomposition(U_cols, V_cols, W_cols, n, m, p, seed=42):
            np.random.seed(seed)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            C_computed = np.zeros((n, p))
            for r in range(len(U_cols)):
                u_flat = U_cols[r]
                v_flat = V_cols[r]
                w_flat = W_cols[r]
                for i in range(n):
                    for j in range(p):
                        s = 0.0
                        for k in range(m):
                            s += u_flat[i*m+k] * v_flat[k*p+j]
                        C_computed[i,j] += s * w_flat[j*n+i]
            return np.allclose(C_computed, C_expected, atol=1e-6)
        
        # Try to remove terms where U columns are linearly dependent
        for iteration in range(30):
            if time.time() - _START > _DEADLINE - 10:
                break
            
            U_mat = np.array(U_cols, dtype=float)
            num_terms = len(U_cols)
            if num_terms <= 30:
                break
            
            s_vals = np.linalg.svd(U_mat, compute_uv=False)
            tol = 1e-9
            num_indep_U = np.sum(s_vals > tol)
            
            if num_indep_U >= num_terms:
                break
            
            U_t = U_mat.T
            U_svd, s, Vt = np.linalg.svd(U_t, full_matrices=True)
            null_vecs = Vt[num_indep_U:]
            
            removed = False
            for ns_vec in null_vecs:
                if time.time() - _START > _DEADLINE - 10:
                    break
                
                drop_idx = int(np.argmax(np.abs(ns_vec)))
                if abs(ns_vec[drop_idx]) < 1e-10:
                    continue
                
                new_U_cols = [U_cols[i] for i in range(num_terms) if i != drop_idx]
                new_V_cols = [V_cols[i] for i in range(num_terms) if i != drop_idx]
                new_W_cols = [W_cols[i] for i in range(num_terms) if i != drop_idx]
                
                if verify_decomposition(new_U_cols, new_V_cols, new_W_cols, n, m, p):
                    U_cols = new_U_cols
                    V_cols = new_V_cols
                    W_cols = new_W_cols
                    removed = True
                    break
            
            if not removed:
                break
    
    U = np.stack(U_cols, axis=1)
    V = np.stack(V_cols, axis=1)
    W = np.stack(W_cols, axis=1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
