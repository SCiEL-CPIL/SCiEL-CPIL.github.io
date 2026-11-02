"""Matrix multiplication 2x3x10 with Strassen-based block decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (20, rank)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def U_idx(i, j):
        return i * m + j  # 0..5
    
    def V_idx(j, k):
        return j * p + k  # 0..29
    
    def W_idx(k, i):
        return k * n + i  # 0..19
    
    def add_term(u_vals, v_vals, w_vals):
        """Add a term with sparse indices and values."""
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for idx, val in u_vals:
            u[idx] = val
        for idx, val in v_vals:
            v[idx] = val
        for idx, val in w_vals:
            w[idx] = val
        U_rows.append(u)
        V_rows.append(v)
        W_rows.append(w)
    
    # --- Block 1: First 2 columns (k=0,1) using Strassen for 2x2 part + schoolbook for 2x1x2 part ---
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B (first 2 cols) = [[b00, b01], [b10, b11], [b20, b21]]
    # 
    # Strassen for A1 (2x2) * B11 (2x2):
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6
    
    # M1 = (a00+a11)*(b00+b11)
    add_term(
        [(U_idx(0,0), 1), (U_idx(1,1), 1)],
        [(V_idx(0,0), 1), (V_idx(1,1), 1)],
        [(W_idx(0,0), 1), (W_idx(0,1), 1), (W_idx(1,0), 1), (W_idx(1,1), 1)]
    )
    
    # M2 = (a10+a11)*b00
    add_term(
        [(U_idx(1,0), 1), (U_idx(1,1), 1)],
        [(V_idx(0,0), 1)],
        [(W_idx(0,1), 1), (W_idx(1,1), -1)]
    )
    
    # M3 = a00*(b01-b11)
    add_term(
        [(U_idx(0,0), 1)],
        [(V_idx(0,1), 1), (V_idx(1,1), -1)],
        [(W_idx(1,0), 1)]
    )
    
    # M4 = a11*(b10-b00)
    add_term(
        [(U_idx(1,1), 1)],
        [(V_idx(1,0), 1), (V_idx(0,0), -1)],
        [(W_idx(0,1), 1)]
    )
    
    # M5 = (a00+a01)*b11
    add_term(
        [(U_idx(0,0), 1), (U_idx(0,1), 1)],
        [(V_idx(1,1), 1)],
        [(W_idx(0,0), -1), (W_idx(1,0), 1)]
    )
    
    # M6 = (a10-a00)*(b00+b01)
    add_term(
        [(U_idx(1,0), 1), (U_idx(0,0), -1)],
        [(V_idx(0,0), 1), (V_idx(0,1), 1)],
        [(W_idx(1,1), 1)]
    )
    
    # M7 = (a01-a11)*(b10+b11)
    add_term(
        [(U_idx(0,1), 1), (U_idx(1,1), -1)],
        [(V_idx(1,0), 1), (V_idx(1,1), 1)],
        [(W_idx(0,0), 1)]
    )
    
    # Schoolbook for A2 (2x1) * B12 (1x2):
    # a02*b20 -> C[0,0]
    add_term(
        [(U_idx(0,2), 1)],
        [(V_idx(2,0), 1)],
        [(W_idx(0,0), 1)]
    )
    
    # a02*b21 -> C[0,1]
    add_term(
        [(U_idx(0,2), 1)],
        [(V_idx(2,1), 1)],
        [(W_idx(1,0), 1)]
    )
    
    # a12*b20 -> C[1,0]
    add_term(
        [(U_idx(1,2), 1)],
        [(V_idx(2,0), 1)],
        [(W_idx(0,1), 1)]
    )
    
    # a12*b21 -> C[1,1]
    add_term(
        [(U_idx(1,2), 1)],
        [(V_idx(2,1), 1)],
        [(W_idx(1,1), 1)]
    )
    
    # --- Block 2: Remaining 8 columns (k=2..9) using schoolbook, rank 48 ---
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                add_term(
                    [(U_idx(i, j), 1)],
                    [(V_idx(j, k), 1)],
                    [(W_idx(k, i), 1)]
                )
    
    total_rank = len(U_rows)
    
    # Build final matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for t in range(total_rank):
        for idx in range(n * m):
            U[idx][t] = U_rows[t][idx]
        for idx in range(m * p):
            V[idx][t] = V_rows[t][idx]
        for idx in range(p * n):
            W[idx][t] = W_rows[t][idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
