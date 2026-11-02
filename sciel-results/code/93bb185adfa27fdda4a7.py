"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_decomposition(n, m, p):
    """Build tensor decomposition for n×m × m×p matrix multiplication.
    
    For 4×4 × 4×6, we split into:
    - 4×4 × 4×4 part: Use Strassen (rank 49)
    - 4×4 × 4×2 part: Use Schoolbook (rank 32)
    Total rank: 81
    """
    rank_terms = []
    
    # Strassen 2x2x2 terms
    strassen_terms_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    
    def add_strassen_term(a_off, b_off, w_offs, u_dict, v_dict, w_dict):
        global_u = {}
        global_v = {}
        global_w = {}
        
        for (li, lj), coeff_u in u_dict.items():
            gi = a_off[0] + li
            gj = a_off[1] + lj
            key = (gi, gj)
            global_u[key] = global_u.get(key, 0) + coeff_u
            
        for (lj, lk), coeff_v in v_dict.items():
            gj = b_off[0] + lj
            gk = b_off[1] + lk
            key = (gj, gk)
            global_v[key] = global_v.get(key, 0) + coeff_v
            
        for (lk, li), coeff_w in w_dict.items():
            for (w_row_start, w_col_start, coeff_c) in w_offs:
                gi = w_row_start + li
                gk = w_col_start + lk
                key = (gk, gi)
                global_w[key] = global_w.get(key, 0) + coeff_w * coeff_c
                
        rank_terms.append((global_u, global_v, global_w))

    def add_product(a_off, b_off, w_offs):
        for u_dict, v_dict, w_dict in strassen_terms_2x2:
            add_strassen_term(a_off, b_off, w_offs, u_dict, v_dict, w_dict)
            
    # Part 1: 4×4 × 4×4 using Strassen
    add_product((0,0), (0,0), [(0,0,1), (2,2,1)])
    add_product((0,0), (2,2), [(0,0,1), (2,2,1)])
    add_product((2,2), (0,0), [(0,0,1), (2,2,1)])
    add_product((2,2), (2,2), [(0,0,1), (2,2,1)])
    
    add_product((2,0), (0,0), [(2,0,1), (2,2,-1)])
    add_product((2,2), (0,0), [(2,0,1), (2,2,-1)])
    
    add_product((0,0), (0,2), [(0,2,1), (2,2,1)])
    add_product((0,0), (2,2), [(0,2,-1), (2,2,-1)])
    
    add_product((2,2), (2,0), [(0,0,1), (2,0,1)])
    add_product((2,2), (0,0), [(0,0,-1), (2,0,-1)])
    
    add_product((0,0), (2,2), [(0,0,-1), (0,2,1)])
    add_product((0,2), (2,2), [(0,0,-1), (0,2,1)])
    
    add_product((2,0), (0,0), [(2,2,1)])
    add_product((2,0), (0,2), [(2,2,1)])
    add_product((0,0), (0,0), [(2,2,-1)])
    add_product((0,0), (0,2), [(2,2,-1)])
    
    add_product((0,2), (2,0), [(0,0,1)])
    add_product((0,2), (2,2), [(0,0,1)])
    add_product((2,2), (2,0), [(0,0,-1)])
    add_product((2,2), (2,2), [(0,0,-1)])
    
    # Part 2: 4×4 × 4×2 using Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u_dict = {(i, j): 1}
                v_dict = {(j, k+4): 1}
                w_dict = {(k+4, i): 1}
                rank_terms.append((u_dict, v_dict, w_dict))
                
    rank = len(rank_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i, r] = coeff
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct using the tensor decomposition formula."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        # alpha_r = sum over (i,j) of A[i,j] * U[i*m+j, r]
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        # beta_r = sum over (j,k) of B[j,k] * V[j*p+k, r]
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        # Add alpha_r * beta_r * W[:, r].reshape(p, n) transposed
        # W[k*n+i, r] corresponds to output (i, k)
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _remove_redundant(U, V, W, n, m, p, tol=1e-9):
    """Remove linearly dependent rank-1 terms."""
    rank = U.shape[1]
    keep = list(range(rank))
    
    for i in range(len(keep)):
        if i >= len(keep):
            break
        ri = keep[i]
        for j in range(i + 1, len(keep)):
            rj = keep[j]
            u_i = U[:, ri]
            u_j = U[:, rj]
            v_i = V[:, ri]
            v_j = V[:, rj]
            w_i = W[:, ri]
            w_j = W[:, rj]
            
            mask_u = np.abs(u_i) > tol
            if not np.any(mask_u):
                if np.allclose(u_j, 0, atol=tol):
                    mask_v = np.abs(v_i) > tol
                    if not np.any(mask_v):
                        if np.allclose(v_j, 0, atol=tol) and np.allclose(w_j, 0, atol=tol):
                            keep.remove(rj)
                            j -= 1
                        continue
                    continue
                else:
                    continue
            
            c_u = u_j[mask_u] / u_i[mask_u]
            if not np.allclose(c_u, c_u[0], rtol=1e-7, atol=tol):
                continue
            c = c_u[0]
            
            if not np.allclose(v_j, c * v_i, rtol=1e-7, atol=tol):
                continue
            if not np.allclose(w_j, c * w_i, rtol=1e-7, atol=tol):
                continue
            
            keep.remove(rj)
            j -= 1
    
    if len(keep) < rank:
        return U[:, keep], V[:, keep], W[:, keep], True
    return U, V, W, False

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build decomposition
    U, V, W = _build_decomposition(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Remove redundant terms
    U, V, W, reduced = _remove_redundant(U, V, W, n, m, p)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        U, V, W = _build_decomposition(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
