"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W with a dynamic list of columns
    U = []
    V = []
    W = []
    
    def add_factor(u_vals, v_vals, w_vals):
        # u_vals: dict {(i,j): val}
        # v_vals: dict {(j,k): val}
        # w_vals: dict {(k,i): val}
        u_row = [0] * (n * m)
        v_row = [0] * (m * p)
        w_row = [0] * (p * n)
        
        for (i, j), val in u_vals.items():
            u_row[i * m + j] = val
        for (j, k), val in v_vals.items():
            v_row[j * p + k] = val
        for (k, i), val in w_vals.items():
            w_row[k * n + i] = val
            
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)

    # Helper to add schoolbook term for specific i, j, k
    def add_schoolbook(i, j, k):
        u_vals = {(i, j): 1}
        v_vals = {(j, k): 1}
        w_vals = {(k, i): 1}
        add_factor(u_vals, v_vals, w_vals)

    # Helper to add Strassen block for specific offset
    def add_strassen_block(i_off, j_off, k_off):
        # i in {i_off, i_off+1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
        
        # M1: (A00+A11)(B00+B11)
        u = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
        v = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
        w = {(k_off, i_off): 1, (k_off+1, i_off+1): 1}
        add_factor(u, v, w)
        
        # M2: (A10+A11)B00
        u = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
        v = {(j_off, k_off): 1}
        w = {(k_off, i_off+1): 1, (k_off+1, i_off+1): -1}
        add_factor(u, v, w)
        
        # M3: A00(B10-B11)
        u = {(i_off, j_off): 1}
        v = {(j_off+1, k_off): 1, (j_off+1, k_off+1): -1}
        w = {(k_off+1, i_off): 1}
        add_factor(u, v, w)
        
        # M4: A11(B01-B00)
        u = {(i_off+1, j_off+1): 1}
        v = {(j_off, k_off+1): 1, (j_off, k_off): -1}
        w = {(k_off, i_off): 1, (k_off, i_off+1): 1}
        add_factor(u, v, w)
        
        # M5: (A00+A01)B11
        u = {(i_off, j_off): 1, (i_off, j_off+1): 1}
        v = {(j_off+1, k_off+1): 1}
        w = {(k_off, i_off): -1, (k_off+1, i_off): 1}
        add_factor(u, v, w)
        
        # M6: (A10-A00)(B00+B01)
        u = {(i_off+1, j_off): 1, (i_off, j_off): -1}
        v = {(j_off, k_off): 1, (j_off, k_off+1): 1}
        w = {(k_off+1, i_off+1): 1}
        add_factor(u, v, w)
        
        # M7: (A01-A11)(B10+B11)
        u = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
        v = {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}
        w = {(k_off, i_off): 1, (k_off+1, i_off): 1}
        add_factor(u, v, w)

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Use Strassen for j in {0,1}
    add_strassen_block(0, 0, 0)
    
    # Rest of Block 1: i in {0,1}, k in {0,1}, j in {2,3}
    for i in [0, 1]:
        for j in [2, 3]:
            for k in [0, 1]:
                add_schoolbook(i, j, k)

    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    # Use Strassen for j in {2,3}
    add_strassen_block(0, 2, 2)
    
    # Rest of Block 2: i in {0,1}, k in {2,3}, j in {0,1}
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [2, 3]:
                add_schoolbook(i, j, k)

    # Remaining parts: i = 2
    for i in [2]:
        for j in range(m):
            for k in range(p):
                add_schoolbook(i, j, k)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
