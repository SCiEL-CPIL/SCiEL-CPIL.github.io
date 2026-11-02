"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

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

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
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
    
    # Strassen 2x2x2 for A block at rows (ai, ai+1), cols (aj, aj+1)
    # B block at rows (bj, bj+1), cols (bk, bk+1)
    # C block at rows (ai, ai+1), cols (bk, bk+1)
    def add_strassen_2x2_2(ai, aj, bj, bk):
        # M1 = (a+d)(e+h) -> C11, C22
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e -> C21 (+1), C22 (-1)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai+1)] = 1; w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        
        # M3 = a(f-h) -> C12 (+1), C22 (+1)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk+1)] = 1; v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e) -> C11 (+1), C21 (+1)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h -> C11 (-1), C12 (+1)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = -1; w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f) -> C22 (+1)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h) -> C11 (+1)
        u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = 1; u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)
    
    # Naive 2x2x1: A (2x2) * B_col (2x1) -> C_col (2x1)
    # 4 terms: A[i,0]*B[0,k], A[i,1]*B[1,k] for i=0,1
    def add_naive_2x2_1(ai, aj, bj, bk):
        # A rows ai, ai+1; A cols aj, aj+1; B rows bj, bj+1; B col bk
        for ii in range(2):
            for jj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai+ii, aj+jj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+jj, bk)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk, ai+ii)] = 1
                add_term(u, v, w)
    
    # Naive 2x1x1: A (2x1) * B (1x1) -> C (2x1) -- not needed directly
    
    # Block decomposition:
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01, B02, B03, B04],  (2x5)
    #      [B10, B11, B12, B13, B14]]  (2x5)
    # C = A * B (4x5)
    # C_top = A00*B00 + A01*B10 (2x5)
    # C_bot = A10*B00 + A11*B10 (2x5)
    
    # For each 2x2 * 2x5 multiplication, use:
    # Strassen on first 2 cols of B (7 terms) + naive on remaining 3 cols (2*3=6 terms) = 13 terms
    # Total: 4 such multiplications * 13 = 52 terms
    
    # A00 (rows 0-1, cols 0-1) * B00 (rows 0-1, cols 0-4) -> C rows 0-1, cols 0-4
    # Strassen on B cols 0-1, naive on B cols 2-4
    add_strassen_2x2_2(0, 0, 0, 0)  # A00 * B00 cols 0-1
    for bk in range(2, 5):
        add_naive_2x2_1(0, 0, 0, bk)  # A00 * B00 col bk
    
    # A01 (rows 0-1, cols 2-3) * B10 (rows 2-3, cols 0-4) -> C rows 0-1, cols 0-4
    add_strassen_2x2_2(0, 2, 2, 0)  # A01 * B10 cols 0-1
    for bk in range(2, 5):
        add_naive_2x2_1(0, 2, 2, bk)  # A01 * B10 col bk
    
    # A10 (rows 2-3, cols 0-1) * B00 (rows 0-1, cols 0-4) -> C rows 2-3, cols 0-4
    add_strassen_2x2_2(2, 0, 0, 0)  # A10 * B00 cols 0-1
    for bk in range(2, 5):
        add_naive_2x2_1(2, 0, 0, bk)  # A10 * B00 col bk
    
    # A11 (rows 2-3, cols 2-3) * B10 (rows 2-3, cols 0-4) -> C rows 2-3, cols 0-4
    add_strassen_2x2_2(2, 2, 2, 0)  # A11 * B10 cols 0-1
    for bk in range(2, 5):
        add_naive_2x2_1(2, 2, 2, bk)  # A11 * B10 col bk
    
    rank = len(U_cols)
    
    # Build matrices
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for r in range(rank):
        U[:, r] = U_cols[r]
        V[:, r] = V_cols[r]
        W[:, r] = W_cols[r]
    
    # Verify
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Round to half-integers (should already be integers)
        U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
        V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
        W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Fallback: schoolbook
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
