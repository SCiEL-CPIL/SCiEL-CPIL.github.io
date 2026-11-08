def construct():
    n, m, p = 4, 5, 5
    U_entries = []
    V_entries = []
    W_entries = []
    
    def add_tensor(U_idx, U_val, V_idx, V_val, W_idx, W_val):
        U_entries.append((U_idx, U_val))
        V_entries.append((V_idx, V_val))
        W_entries.append((W_idx, W_val))
        
    # Schoolbook
    def add_schoolbook(A_set, B_set):
        for i, j in A_set:
            for j2, k in B_set:
                if j == j2:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_tensor([u_idx], [1], [v_idx], [1], [w_idx], [1])
                    
    # Strassen
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        # M1
        add_tensor([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        # M2
        add_tensor([a10, a11], [1, 1], [b00], [1], [c10], [1])
        # M3
        add_tensor([a00], [1], [b01, b11], [1, -1], [c01], [1])
        # M4
        add_tensor([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
        # M5
        add_tensor([a00, a01], [1, 1], [b11], [1], [c01, c00], [1, -1])
        # M6
        add_tensor([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        # M7
        add_tensor([a01, a11], [1, -1], [b10, b11], [1, 1], [c00, c11], [1, -1])

    # Define blocks
    # A: 4x5
    # Rows 0-1, 2-3. Cols 0-1, 2-4.
    A00 = [(0,0), (0,1), (1,0), (1,1)]
    A01 = [(0,2), (0,3), (0,4), (1,2), (1,3), (1,4)]
    A10 = [(2,0), (2,1), (3,0), (3,1)]
    A11 = [(2,2), (2,3), (2,4), (3,2), (3,3), (3,4)]
    
    # B: 5x5
    # Rows 0-1, 2-4. Cols 0-1, 2-4.
    B00 = [(0,0), (0,1), (1,0), (1,1)]
    B01 = [(0,2), (0,3), (0,4), (1,2), (1,3), (1,4)]
    B10 = [(2,0), (2,1), (3,0), (3,1), (4,0), (4,1)]
    B11 = [(2,2), (2,3), (2,4), (3,2), (3,3), (3,4), (4,2), (4,3), (4,4)]
    
    # C indices for Strassen (2x2 blocks)
    # C00: rows 0-1, cols 0-1
    c00_00 = (0,0) # k=0, i=0 -> w_idx = 0*4+0=0
    c00_01 = (0,1) # k=0, i=1 -> w_idx = 0*4+1=1
    c00_10 = (1,0) # k=1, i=0 -> w_idx = 1*4+0=4
    c00_11 = (1,1) # k=1, i=1 -> w_idx = 1*4+1=5
    
    # Wait, Strassen is for 2x2 blocks.
    # The C block is 2x2.
    # So c00 is the top-left 1x1 of the C block?
    # No, in the Strassen call, a00 is the top-left 1x1 of the A block.
    # So for the first Strassen call (A00 * B00), the A block is 2x2, B block is 2x2, C block is 2x2.
    # So a00 is (0,0), a01 is (0,1), a10 is (1,0), a11 is (1,1).
    # b00 is (0,0), etc.
    # c00 is (0,0) in C block, which is global (0,0).
    
    # Let's extract the single indices for Strassen.
    # A00 block indices:
    a00_00 = (0,0)
    a00_01 = (0,1)
    a00_10 = (1,0)
    a00_11 = (1,1)
    
    # B00 block indices:
    b00_00 = (0,0)
    b00_01 = (0,1)
    b00_10 = (1,0)
    b00_11 = (1,1)
    
    # C00 block indices (k, i):
    c00_00 = (0,0)
    c00_01 = (0,1)
    c00_10 = (1,0)
    c00_11 = (1,1)
    
    add_strassen(a00_00, a00_01, a00_10, a00_11,
                 b00_00, b00_01, b00_10, b00_11,
                 c00_00, c00_01, c00_10, c00_11)
                 
    # A10 block indices (rows 2-3, cols 0-1)
    a10_00 = (2,0)
    a10_01 = (2,1)
    a10_10 = (3,0)
    a10_11 = (3,1)
    
    # C10 block indices (rows 2-3, cols 0-1) -> k in 0-1, i in 2-3
    c10_00 = (0,2)
    c10_01 = (0,3)
    c10_10 = (1,2)
    c10_11 = (1,3)
    
    add_strassen(a10_00, a10_01, a10_10, a10_11,
                 b00_00, b00_01, b00_10, b00_11,
                 c10_00, c10_01, c10_10, c10_11)
                 
    # Schoolbook parts
    add_schoolbook(A01, B10)
    add_schoolbook(A00, B01)
    add_schoolbook(A01, B11)
    add_schoolbook(A11, B10)
    add_schoolbook(A10, B01)
    add_schoolbook(A11, B11)
    
    # Build matrices
    rank = len(U_entries)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for u_idx, val in U_entries[c]:
            U[u_idx][c] += val
        for v_idx, val in V_entries[c]:
            V[v_idx][c] += val
        for w_idx, val in W_entries[c]:
            W[w_idx][c] += val
            
    return {"U": U, "V": V, "W": W}
