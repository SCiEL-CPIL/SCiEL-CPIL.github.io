import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _build_block_decomposition(n, m, p):
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    def schoolbook_block(i_off, j_off, k_off, bn, bm, bp):
        for i in range(bn):
            for j in range(bm):
                for k in range(bp):
                    ig = i_off + i
                    jg = j_off + j
                    kg = k_off + k
                    add_term(
                        {ig * m + jg: 1},
                        {jg * p + kg: 1},
                        {kg * n + ig: 1}
                    )
    
    def strassen_block(i_off, j_off, k_off):
        def gi(i, j): return (i_off + i) * m + (j_off + j)
        def gj(j, k): return (j_off + j) * p + (k_off + k)
        def gk(k, i): return (k_off + k) * n + (i_off + i)
        
        add_term(
            {gi(0,0): 1, gi(1,1): 1},
            {gj(0,0): 1, gj(1,1): 1},
            {gk(0,0): 1, gk(1,1): 1}
        )
        add_term(
            {gi(1,0): 1, gi(1,1): 1},
            {gj(0,0): 1},
            {gk(0,1): 1, gk(1,1): -1}
        )
        add_term(
            {gi(0,0): 1},
            {gj(0,1): 1, gj(1,1): -1},
            {gk(1,0): 1, gk(1,1): 1}
        )
        add_term(
            {gi(1,1): 1},
            {gj(1,0): 1, gj(0,0): -1},
            {gk(0,0): 1, gk(0,1): 1}
        )
        add_term(
            {gi(0,0): 1, gi(0,1): 1},
            {gj(1,1): 1},
            {gk(0,0): -1, gk(1,0): 1}
        )
        add_term(
            {gi(1,0): 1, gi(0,0): -1},
            {gj(0,0): 1, gj(0,1): 1},
            {gk(1,1): 1}
        )
        add_term(
            {gi(0,1): 1, gi(1,1): -1},
            {gj(1,0): 1, gj(1,1): 1},
            {gk(0,0): 1}
        )
    
    i_parts = [(0, 2), (2, 2)]
    j_parts = [(0, 2), (2, 2), (4, 1)]
    k_parts = [(0, 2), (2, 2), (4, 1)]
    
    for i_off, i_sz in i_parts:
        for j_off, j_sz in j_parts:
            for k_off, k_sz in k_parts:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    strassen_block(i_off, j_off, k_off)
                else:
                    schoolbook_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
    
    return terms

def _terms_to_matrices(terms, n, m, p):
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, c] = val
        for idx, val in v_dict.items():
            V[idx, c] = val
        for idx, val in w_dict.items():
            W[idx, c] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=tol)

def _try_eliminate_columns(U, V, W, n, m, p, deadline):
    """Try to eliminate redundant columns by checking if removing one column
    still gives a valid decomposition. Iterate until no more can be removed."""
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    
    while time.time() < deadline:
        found_elimination = False
        for col in range(current_rank):
            if time.time() > deadline:
                return current_U, current_V, current_W, current_rank
            
            mask = np.ones(current_rank, dtype=bool)
            mask[col] = False
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            if _verify_decomposition(U_test, V_test, W_test, n, m, p):
                current_U, current_V, current_W = U_test, V_test, W_test
                current_rank = U_test.shape[1]
                found_elimination = True
                break
        
        if not found_elimination:
            break
    
    return current_U, current_V, current_W, current_rank

def _try_pairwise_elimination(U, V, W, n, m, p, deadline):
    """Try to eliminate pairs of columns simultaneously."""
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    
    while time.time() < deadline:
        found_elimination = False
        for col1 in range(current_rank):
            if time.time() > deadline:
                return current_U, current_V, current_W, current_rank
            for col2 in range(col1 + 1, current_rank):
                if time.time() > deadline:
                    return current_U, current_V, current_W, current_rank
                
                mask = np.ones(current_rank, dtype=bool)
                mask[col1] = False
                mask[col2] = False
                U_test = current_U[:, mask]
                V_test = current_V[:, mask]
                W_test = current_W[:, mask]
                
                if _verify_decomposition(U_test, V_test, W_test, n, m, p):
                    current_U, current_V, current_W = U_test, V_test, W_test
                    current_rank = U_test.shape[1]
                    found_elimination = True
                    break
            if found_elimination:
                break
        
        if not found_elimination:
            break
    
    return current_U, current_V, current_W, current_rank

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Build structured decomposition (92 rank)
    terms = _build_block_decomposition(n, m, p)
    U, V, W = _terms_to_matrices(terms, n, m, p)
    
    # Verify structured solution
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try single column elimination
    if time.time() < deadline:
        U_e, V_e, W_e, rank_e = _try_eliminate_columns(U, V, W, n, m, p, deadline)
        if rank_e < best_rank:
            best_U, best_V, best_W = U_e, V_e, W_e
            best_rank = rank_e
    
    # Try pairwise elimination on the best result so far
    if time.time() < deadline:
        U_p, V_p, W_p, rank_p = _try_pairwise_elimination(best_U, best_V, best_W, n, m, p, deadline)
        if rank_p < best_rank:
            best_U, best_V, best_W = U_p, V_p, W_p
            best_rank = rank_p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
