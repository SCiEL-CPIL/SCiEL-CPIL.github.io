"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 5
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal r
        # u_indices: list of (idx, val)
        # v_indices: list of (idx, val)
        # w_indices: list of (idx, val)
        for idx, val in u_indices:
            U[idx][r] += val
        for idx, val in v_indices:
            V[idx][r] += val
        for idx, val in w_indices:
            W[idx][r] += val
        r += 1

    def add_strassen(i_base, k_base, j_base):
        # i_base: 0 or 1 (row of A)
        # k_base: 0, 2 (start of k)
        # j_base: 0, 2 (start of j)
        
        # Local indices for 2x2x2
        # A: i in {0,1}, k in {0,1}
        # B: k in {0,1}, j in {0,1}
        # C: i in {0,1}, j in {0,1}
        
        # Global U index: i_global * m + k_global
        # Global V index: k_global * p + j_global
        # Global W index: j_global * n + i_global
        
        def ui(i_l, k_l): return (i_base + i_l) * m + (k_base + k_l)
        def vi(k_l, j_l): return (k_base + k_l) * p + (j_base + j_l)
        def wi(j_l, i_l): return (j_base + j_l) * n + (i_base + i_l)
        
        # M1
        add_term([(ui(0,0), 1), (ui(1,1), 1)], 
                 [(vi(0,0), 1), (vi(1,1), 1)], 
                 [(wi(0,0), 1), (wi(1,1), 1)])
        # M2
        add_term([(ui(1,0), 1), (ui(1,1), 1)], 
                 [(vi(0,0), 1)], 
                 [(wi(0,1), 1), (wi(1,1), -1)])
        # M3
        add_term([(ui(0,0), 1)], 
                 [(vi(0,1), 1), (vi(1,1), -1)], 
                 [(wi(0,0), 1), (wi(1,1), 1)])
        # M4
        add_term([(ui(1,1), 1)], 
                 [(vi(1,0), 1), (vi(0,0), -1)], 
                 [(wi(0,0), 1), (wi(0,1), 1)])
        # M5
        add_term([(ui(0,0), 1), (ui(0,1), 1)], 
                 [(vi(1,1), 1)], 
                 [(wi(0,0), -1), (wi(0,1), 1)])
        # M6
        add_term([(ui(1,0), 1), (ui(0,0), -1)], 
                 [(vi(0,0), 1), (vi(0,1), 1)], 
                 [(wi(1,1), 1)])
        # M7
        add_term([(ui(0,1), 1), (ui(1,1), -1)], 
                 [(vi(1,0), 1), (vi(1,1), 1)], 
                 [(wi(0,0), 1)])

    def add_schoolbook_2x2x1(i_base, k_base, j_base):
        # A: 2x2, B: 2x1, C: 2x1
        # C_i0 = A_i0 B_00 + A_i1 B_10
        def ui(i_l, k_l): return (i_base + i_l) * m + (k_base + k_l)
        def vi(k_l, j_l): return (k_base + k_l) * p + (j_base + j_l)
        def wi(j_l, i_l): return (j_base + j_l) * n + (i_base + i_l)
        
        # T1: A_00 B_00 -> C_00
        add_term([(ui(0,0), 1)], [(vi(0,0), 1)], [(wi(0,0), 1)])
        # T2: A_01 B_10 -> C_00
        add_term([(ui(0,1), 1)], [(vi(1,0), 1)], [(wi(0,0), 1)])
        # T3: A_10 B_00 -> C_10
        add_term([(ui(1,0), 1)], [(vi(0,0), 1)], [(wi(0,1), 1)])
        # T4: A_11 B_10 -> C_10
        add_term([(ui(1,1), 1)], [(vi(1,0), 1)], [(wi(0,1), 1)])

    def add_schoolbook_2x1x5(i_base, k_base, j_base):
        # A: 2x1, B: 1x5, C: 2x5
        def ui(i_l, k_l): return (i_base + i_l) * m + (k_base + k_l)
        def vi(k_l, j_l): return (k_base + k_l) * p + (j_base + j_l)
        def wi(j_l, i_l): return (j_base + j_l) * n + (i_base + i_l)
        
        for i_l in range(2):
            for j_l in range(5):
                add_term([(ui(i_l, 0), 1)], [(vi(0, j_l), 1)], [(wi(j_l, i_l), 1)])

    # Block 1: A1 (k=0,1)
    add_strassen(0, 0, 0)       # A1 * P (j=0,1)
    add_strassen(0, 0, 2)       # A1 * Q (j=2,3)
    add_schoolbook_2x2x1(0, 0, 4) # A1 * S (j=4)
    
    # Block 2: A2 (k=2,3)
    add_strassen(0, 2, 0)       # A2 * P' (j=0,1)
    add_strassen(0, 2, 2)       # A2 * Q' (j=2,3)
    add_schoolbook_2x2x1(0, 2, 4) # A2 * S' (j=4)
    
    # Block 3: A3 (k=4)
    add_schoolbook_2x1x5(0, 4, 0) # A3 * R3 (j=0..4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
