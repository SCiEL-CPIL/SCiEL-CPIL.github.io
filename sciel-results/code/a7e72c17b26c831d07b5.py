import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    # Partition sizes for 5x5x5
    sizes = [2, 2, 1]
    starts = [0, 2, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        
        for idx, coeff in u_coeffs:
            U_col[idx] = coeff
        for idx, coeff in v_coeffs:
            V_col[idx] = coeff
        for idx, coeff in w_coeffs:
            W_col[idx] = coeff
            
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    def map_u(i, k, p_local, q_local):
        return (starts[i] + p_local) * m + (starts[k] + q_local)

    def map_v(k, j, q_local, t_local):
        return (starts[k] + q_local) * p + (starts[j] + t_local)

    def map_w(i, j, p_local, t_local):
        return (starts[j] + t_local) * n + (starts[i] + p_local)

    def strassen_terms(i, j, k):
        """Generate the 7 Strassen terms for a 2x2x2 block."""
        # Term 1: (a00+a11)(b00+b11) -> c00, c11
        add_term(
            [(map_u(i, k, 0, 0), 1), (map_u(i, k, 1, 1), 1)],
            [(map_v(k, j, 0, 0), 1), (map_v(k, j, 1, 1), 1)],
            [(map_w(i, j, 0, 0), 1), (map_w(i, j, 1, 1), 1)]
        )
        # Term 2: (a10+a11)(b00) -> c10, c11
        add_term(
            [(map_u(i, k, 1, 0), 1), (map_u(i, k, 1, 1), 1)],
            [(map_v(k, j, 0, 0), 1)],
            [(map_w(i, j, 1, 0), 1), (map_w(i, j, 1, 1), -1)]
        )
        # Term 3: (a00)(b01-b11) -> c01, c11
        add_term(
            [(map_u(i, k, 0, 0), 1)],
            [(map_v(k, j, 0, 1), 1), (map_v(k, j, 1, 1), -1)],
            [(map_w(i, j, 0, 1), 1), (map_w(i, j, 1, 1), 1)]
        )
        # Term 4: (a11)(b10-b00) -> c00, c10
        add_term(
            [(map_u(i, k, 1, 1), 1)],
            [(map_v(k, j, 1, 0), 1), (map_v(k, j, 0, 0), -1)],
            [(map_w(i, j, 0, 0), 1), (map_w(i, j, 1, 0), 1)]
        )
        # Term 5: (a00+a01)(b11) -> c00, c01
        add_term(
            [(map_u(i, k, 0, 0), 1), (map_u(i, k, 0, 1), 1)],
            [(map_v(k, j, 1, 1), 1)],
            [(map_w(i, j, 0, 0), -1), (map_w(i, j, 0, 1), 1)]
        )
        # Term 6: (a10-a00)(b00+b01) -> c11
        add_term(
            [(map_u(i, k, 1, 0), 1), (map_u(i, k, 0, 0), -1)],
            [(map_v(k, j, 0, 0), 1), (map_v(k, j, 0, 1), 1)],
            [(map_w(i, j, 1, 1), 1)]
        )
        # Term 7: (a01-a11)(b10+b11) -> c00
        add_term(
            [(map_u(i, k, 0, 1), 1), (map_u(i, k, 1, 1), -1)],
            [(map_v(k, j, 1, 0), 1), (map_v(k, j, 1, 1), 1)],
            [(map_w(i, j, 0, 0), 1)]
        )

    def schoolbook_terms(i, j, k, r, s, c):
        """Generate schoolbook terms for an r x s x c block."""
        for p_local in range(r):
            for q_local in range(s):
                for t_local in range(c):
                    add_term(
                        [(map_u(i, k, p_local, q_local), 1)],
                        [(map_v(k, j, q_local, t_local), 1)],
                        [(map_w(i, j, p_local, t_local), 1)]
                    )

    # Iterate over output blocks i, j
    for i in range(3):
        for j in range(3):
            if time.time() - start_time > DEADLINE:
                break
            for k in range(3):
                r = sizes[i]
                s = sizes[k]
                c = sizes[j]
                
                if r == 2 and s == 2 and c == 2:
                    strassen_terms(i, j, k)
                else:
                    schoolbook_terms(i, j, k, r, s, c)
    
    rank = len(U_cols)
    
    # Convert to standard format
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
