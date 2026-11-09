"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen(u_rows, v_rows, w_rows):
        nonlocal col
        # U rows: [a00, a01, a10, a11]
        # V rows: [b00, b01, b10, b11]
        # W rows: [c00, c01, c10, c11]
        
        # M1
        U[u_rows[0]][col] += 1; U[u_rows[3]][col] += 1
        V[v_rows[0]][col] += 1; V[v_rows[3]][col] += 1
        W[w_rows[0]][col] += 1; W[w_rows[3]][col] += 1
        col += 1
        
        # M2
        U[u_rows[2]][col] += 1; U[u_rows[3]][col] += 1
        V[v_rows[0]][col] += 1
        W[w_rows[2]][col] += 1; W[w_rows[3]][col] += -1
        col += 1
        
        # M3
        U[u_rows[0]][col] += 1
        V[v_rows[2]][col] += 1; V[v_rows[3]][col] += -1
        W[w_rows[1]][col] += 1; W[w_rows[3]][col] += 1
        col += 1
        
        # M4
        U[u_rows[3]][col] += 1
        V[v_rows[1]][col] += 1; V[v_rows[0]][col] += -1
        W[w_rows[0]][col] += 1; W[w_rows[2]][col] += 1
        col += 1
        
        # M5
        U[u_rows[0]][col] += 1; U[u_rows[1]][col] += 1
        V[v_rows[3]][col] += 1
        W[w_rows[0]][col] += -1; W[w_rows[1]][col] += 1
        col += 1
        
        # M6
        U[u_rows[2]][col] += 1; U[u_rows[0]][col] += -1
        V[v_rows[0]][col] += 1; V[v_rows[1]][col] += 1
        W[w_rows[3]][col] += 1
        col += 1
        
        # M7
        U[u_rows[1]][col] += 1; U[u_rows[3]][col] += -1
        V[v_rows[2]][col] += 1; V[v_rows[3]][col] += 1
        W[w_rows[0]][col] += 1
        col += 1

    # Block 1: A11 (2x2) x B111 (2x2) -> C111 (2x2)
    # A11 rows: i=0,1; j=0,1. Indices: 0,1,4,5
    # B111 rows: j=0,1; k=0,1. Indices: 0,1,5,6
    # C111 rows: k=0,1; i=0,1. Indices: 0,1,3,4
    add_strassen([0, 1, 4, 5], [0, 1, 5, 6], [0, 1, 3, 4])
    
    # Block 2: A11 (2x2) x B112 (2x2) -> C112 (2x2)
    # B112 rows: j=0,1; k=2,3. Indices: 10,11,15,16
    # C112 rows: k=2,3; i=0,1. Indices: 6,7,9,10
    add_strassen([0, 1, 4, 5], [10, 11, 15, 16], [6, 7, 9, 10])
    
    # Block 3: A11 (2x2) x B113 (2x1) -> C113 (2x1)
    # B113 rows: j=0,1; k=4. Indices: 4, 9
    # C113 rows: k=4; i=0,1. Indices: 12, 13
    u_rows = [0, 1, 4, 5]
    v_rows = [4, 9]
    w_rows = [12, 13]
    
    # C0 = a00 b00 + a01 b10
    U[u_rows[0]][col] += 1; V[v_rows[0]][col] += 1; W[w_rows[0]][col] += 1
    col += 1
    U[u_rows[1]][col] += 1; V[v_rows[1]][col] += 1; W[w_rows[0]][col] += 1
    col += 1
    # C1 = a10 b00 + a11 b10
    U[u_rows[2]][col] += 1; V[v_rows[0]][col] += 1; W[w_rows[1]][col] += 1
    col += 1
    U[u_rows[3]][col] += 1; V[v_rows[1]][col] += 1; W[w_rows[1]][col] += 1
    col += 1
    
    # Block 4: A21 (1x2) x B21 (2x5) -> C21 (1x5)
    # A21 rows: i=2; j=2,3. Indices: 10, 11
    # B21 rows: j=2,3; k=0..4. Indices: 10..19
    # C21 rows: k=0..4; i=2. Indices: 2, 5, 8, 11, 14
    u_rows = [10, 11]
    w_rows = [2, 5, 8, 11, 14]
    
    for k in range(5):
        v_idx0 = 10 + k # j=2, k
        v_idx1 = 15 + k # j=3, k
        w_idx = w_rows[k]
        
        # Ck = a00 b0k + a01 b1k
        U[u_rows[0]][col] += 1; V[v_idx0][col] += 1; W[w_idx][col] += 1
        col += 1
        U[u_rows[1]][col] += 1; V[v_idx1][col] += 1; W[w_idx][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
