import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[ _round_to_half_integer(x) for x in row] for row in M])

def _strassen_construct(n, m, p):
    """Construct the Strassen-based decomposition for 4x4x5.
    
    Uses rank-7 Strassen for 2x2x2 blocks and rank-8 for 2x2x3 blocks.
    Total: 4*7 + 4*8 = 28 + 32 = 60 (if we can achieve rank-8 for 2x2x3)
    Fallback: 4*7 + 4*9 = 64
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    def add_strassen_2x2_2(ai, aj, bj, bk):
        # Standard Strassen rank-7 for 2x2x2
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai+1)] = 1; w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk+1)] = 1; v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = -1; w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = 1; u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    def add_2x2x3_rank8(ai, aj, bj, bk):
        """Rank-8 scheme for 2x2x3 matrix multiplication.
        
        For C = AB where A is 2x2, B is 2x3:
        C[0,0] = a*b + c*e
        C[0,1] = a*c + c*f  
        C[0,2] = a*d + c*g
        C[1,0] = b*b + d*e
        C[1,1] = b*c + d*f
        C[1,2] = b*d + d*g
        
        Use Strassen-like combinations for the first two columns (rank 7)
        plus one more term for the third column.
        """
        # Use Strassen for columns bk, bk+1 (2x2x2 subproblem)
        # Then add terms for column bk+2
        
        # M1 = (a+d)(e+h) -> contributes to C[0,0], C[0,1], C[1,0], C[1,1]
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai+1)] = 1; w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk+1)] = 1; v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = -1; w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = 1; u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)
        
        # M8 for third column: need to handle C[0,2] and C[1,2]
        # C[0,2] = a*d + c*g, C[1,2] = b*d + d*g
        # Use one shared term: (a+b+d+c) won't work simply
        # Try: M8 = (a+c) * something? No, let's use two naive terms or find a combo
        # Actually for rank 8, we need one more term. Let's use:
        # M8 handles both: use (a+b) for U, d for V, and split W
        # But that gives C[0,2] += a*d and C[1,2] += b*d, missing c*g and d*g
        # 
        # Alternative: use two separate naive terms for column bk+2
        # That would be rank 9. For rank 8, try:
        # M8 = (a+c+d+b) * d? No.
        # Let's just use 2 naive terms (rank 9) for reliability
        
        # Naive for col bk+2, row ai
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk+2)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+2, ai)] = 1
        add_term(u, v, w)
        
        # Naive for col bk+2, row ai+1
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk+2)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+2, ai+1)] = 1
        add_term(u, v, w)
        
        # Naive for col bk+2, row ai, col aj+1
        u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+2)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+2, ai)] = 1
        add_term(u, v, w)
        
        # Naive for col bk+2, row ai+1, col aj+1
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+2)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+2, ai+1)] = 1
        add_term(u, v, w)

    def add_2x2x3_rank9(ai, aj, bj, bk):
        """Rank-9 scheme: Strassen (7) + 2 naive for third column."""
        add_strassen_2x2_2(ai, aj, bj, bk)
        # Naive for col bk+2
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk+2, ai+di)] = 1
                add_term(u, v, w)

    # C00 = A00 B00 + A01 B10 (2x2x2 + 2x2x2 = rank 14)
    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    # C01 = A00 B01 + A01 B11 (2x2x3 + 2x2x3)
    add_2x2x3_rank9(0, 0, 0, 2)
    add_2x2x3_rank9(0, 2, 2, 2)
    # C10 = A10 B00 + A11 B10 (2x2x2 + 2x2x2 = rank 14)
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    # C11 = A10 B01 + A11 B11 (2x2x3 + 2x2x3)
    add_2x2x3_rank9(2, 0, 0, 2)
    add_2x2x3_rank9(2, 2, 2, 2)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _remove_zero_columns(U, V, W):
    """Remove columns that are all zeros in any of U, V, or W."""
    rank = U.shape[1]
    keep = []
    for r in range(rank):
        if np.any(U[:, r] != 0) and np.any(V[:, r] != 0) and np.any(W[:, r] != 0):
            keep.append(r)
    U = U[:, keep]
    V = V[:, keep]
    W = W[:, keep]
    return U, V, W

def _als_decompose(n, m, p, rank, seed=42, max_iters=300, tol=1e-9, init_U=None, init_V=None, init_W=None):
    """Try to find a rank-r decomposition using alternating least squares."""
    np.random.seed(seed)
    
    if init_U is not None:
        U = np.array(init_U, dtype=float).copy()
        V = np.array(init_V, dtype=float).copy()
        W = np.array(init_W, dtype=float).copy()
    else:
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
    
    for it in range(max_iters):
        # Optimize U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V: for each (j,k)
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W: for each (i,k)
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
    
    # Round to half-integers
    U_r = _round_matrix_to_half_int(U)
    V_r = _round_matrix_to_half_int(V)
    W_r = _round_matrix_to_half_int(W)
    
    # Remove zero columns
    U_r, V_r, W_r = _remove_zero_columns(U_r, V_r, W_r)
    
    # Verify
    if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _polish_decomposition(U, V, W, n, m, p, max_iters=50, tol=1e-12):
    """Final high-precision polish pass to tighten the decomposition."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    rank = U.shape[1]
    
    for it in range(max_iters):
        # Optimize U with high precision
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=1e-14)[0]
                except:
                    pass
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=1e-14)[0]
                except:
                    pass
        
        # Optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=1e-14)[0]
                except:
                    pass
    
    # Round to half-integers
    U_r = _round_matrix_to_half_int(U)
    V_r = _round_matrix_to_half_int(V)
    W_r = _round_matrix_to_half_int(W)
    
    U_r, V_r, W_r = _remove_zero_columns(U_r, V_r, W_r)
    
    if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Get the Strassen-based decomposition as baseline (rank 64)
    U_base, V_base, W_base = _strassen_construct(n, m, p)
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Try to find lower-rank decompositions using ALS
    # Start from best_rank-1 and go down
    min_rank_to_try = max(40, best_rank - 25)
    
    for rank in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 40:
            break
        
        found = False
        # Try multiple seeds and initializations
        for seed in range(8):
            if time.time() > deadline - 40:
                break
            
            # Alternate between random init and Strassen-based init
            if seed % 2 == 0:
                result = _als_decompose(n, m, p, rank, seed=seed, max_iters=150)
            else:
                # Initialize from Strassen structure, truncated
                U_init = np.zeros((n*m, rank))
                V_init = np.zeros((m*p, rank))
                W_init = np.zeros((p*n, rank))
                # Copy first 'rank' columns from Strassen if available
                n_copy = min(rank, best_rank)
                U_init[:, :n_copy] = np.array(U_base)[:, :n_copy]
                V_init[:, :n_copy] = np.array(V_base)[:, :n_copy]
                W_init[:, :n_copy] = np.array(W_base)[:, :n_copy]
                result = _als_decompose(n, m, p, rank, seed=seed, max_iters=150,
                                       init_U=U_init, init_V=V_init, init_W=W_init)
            
            if result is not None:
                U, V, W = result
                # Try to polish
                polished = _polish_decomposition(U, V, W, n, m, p, max_iters=30)
                if polished is not None:
                    U, V, W = polished
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
