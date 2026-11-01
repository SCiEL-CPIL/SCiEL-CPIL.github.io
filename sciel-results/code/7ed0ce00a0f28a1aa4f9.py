"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions: n=2, m=5, p=6
    # Strategy:
    # Split m=5 into m1=2, m2=2, m3=1.
    # Part 1 (m3=1): Rank 12.
    # Part 2 (m1=2): Rank 21 (using Strassen on 2x2x6 split into 3x 2x2x2).
    # Part 3 (m2=2): Rank 21.
    # Total Rank = 12 + 21 + 21 = 54.
    
    rank = 54
    
    # Initialize matrices
    # U: (n*m, rank) = (10, 54)
    # V: (m*p, rank) = (30, 54)
    # W: (p*n, rank) = (12, 54)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank 1 term
    def add_term(u_indices, v_indices, w_indices, val=1.0):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += val
        for idx in v_indices:
            V[idx][col] += val
        for idx in w_indices:
            W[idx][col] += val
        col += 1
        
    # Part 1: j = 4 (the 1x1 block)
    # A[i, 4] * B[4, k] -> C[i, k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            add_term([u_idx], [v_idx], [w_idx])
            
    # Part 2 & 3: j in {0,1} and j in {2,3}
    # We define the Strassen terms for a 2x2x2 block.
    # Local indices for A: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # Local indices for B: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # Local indices for C: (0,0)=C00, (0,1)=C01, (1,0)=C10, (1,1)=C11
    
    # Strassen Terms:
    # Each term is (u_local, v_local, w_local, sign)
    # u_local: list of (li, lj, val)
    # v_local: list of (lj, lk, val)
    # w_local: list of (li, lk, val)
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        (
            [(0,0,1), (1,1,1)], # a, d
            [(0,0,1), (1,1,1)], # e, h
            [(0,0,1), (1,1,1)]  # C00, C11
        ),
        # M2 = (c+d)e
        (
            [(1,0,1), (1,1,1)], # c, d
            [(0,0,1)],          # e
            [(1,0,1), (1,1,-1)] # C10, C11
        ),
        # M3 = a(f-h)
        (
            [(0,0,1)],          # a
            [(0,1,1), (1,1,-1)],# f, -h
            [(0,1,1), (1,1,1)]  # C01, C11
        ),
        # M4 = d(g-e)
        (
            [(1,1,1)],          # d
            [(1,0,1), (0,0,-1)],# g, -e
            [(0,0,1), (1,0,1)]  # C00, C10
        ),
        # M5 = (a+b)h
        (
            [(0,0,1), (0,1,1)], # a, b
            [(1,1,1)],          # h
            [(0,0,-1), (0,1,1)] # C00, C01
        ),
        # M6 = (c-a)(e+f)
        (
            [(1,0,1), (0,0,-1)],# c, -a
            [(0,0,1), (0,1,1)], # e, f
            [(1,1,1)]           # C11
        ),
        # M7 = (b-d)(g+h)
        (
            [(0,1,1), (1,1,-1)],# b, -d
            [(1,0,1), (1,1,1)], # g, h
            [(0,0,1)]           # C00
        ),
    ]
    
    # Process the two 2x2 blocks for j
    # Block 1: j = 0, 1
    # Block 2: j = 2, 3
    
    for block_idx in range(2):
        j_base = block_idx * 2
        
        # Split k into 3 pairs of 2
        for k_pair in range(3):
            k_base = k_pair * 2
            
            for term in strassen_terms:
                u_list, v_list, w_list = term
                
                # Map local indices to global indices for U
                u_indices = []
                for (li, lj, val) in u_list:
                    global_i = li
                    global_j = j_base + lj
                    u_indices.append(global_i * m + global_j)
                    
                # Map local indices to global indices for V
                v_indices = []
                for (lj, lk, val) in v_list:
                    global_j = j_base + lj
                    global_k = k_base + lk
                    v_indices.append(global_j * p + global_k)
                    
                # Map local indices to global indices for W
                w_indices = []
                for (li, lk, val) in w_list:
                    global_i = li
                    global_k = k_base + lk
                    w_indices.append(global_k * n + global_i)
                    
                # We need to handle the values (coefficients)
                # The helper add_term currently assumes val=1 for all.
                # We need to modify the helper or inline the logic to handle coefficients.
                
                # Let's inline the logic for clarity and correctness with coefficients
                for (li, lj, val_u) in u_list:
                    global_i = li
                    global_j = j_base + lj
                    U[global_i * m + global_j][col] += val_u
                for (lj, lk, val_v) in v_list:
                    global_j = j_base + lj
                    global_k = k_base + lk
                    V[global_j * p + global_k][col] += val_v
                for (li, lk, val_w) in w_list:
                    global_i = li
                    global_k = k_base + lk
                    W[global_k * n + global_i][col] += val_w
                    
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
