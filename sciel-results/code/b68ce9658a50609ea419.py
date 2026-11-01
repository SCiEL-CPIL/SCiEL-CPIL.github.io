"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 7
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    # Helper to add a 2x2x2 Strassen block
    # a_idx: list of 4 U indices [A11, A12, A21, A22]
    # b_idx: list of 4 V indices [B11, B12, B21, B22]
    # c_idx: list of 4 W indices [C11, C12, C21, C22]
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # M1: A11(B11+B12)
        U[a_idx[0]][col] = 1
        V[b_idx[0]][col] = 1; V[b_idx[1]][col] = 1
        W[c_idx[0]][col] = 1; W[c_idx[1]][col] = 1
        col += 1
        
        # M2: (A21+A22)B11
        U[a_idx[2]][col] = 1; U[a_idx[3]][col] = 1
        V[b_idx[0]][col] = 1
        W[c_idx[2]][col] = 1
        col += 1
        
        # M3: A11(B21-B11)
        U[a_idx[0]][col] = 1
        V[b_idx[0]][col] = -1; V[b_idx[2]][col] = 1
        W[c_idx[0]][col] = 1
        col += 1
        
        # M4: A22(B12-B22)
        U[a_idx[3]][col] = 1
        V[b_idx[1]][col] = 1; V[b_idx[3]][col] = -1
        W[c_idx[3]][col] = 1
        col += 1
        
        # M5: (A11+A22)(B11+B22)
        U[a_idx[0]][col] = 1; U[a_idx[3]][col] = 1
        V[b_idx[0]][col] = 1; V[b_idx[3]][col] = 1
        W[c_idx[0]][col] = 1; W[c_idx[1]][col] = 1; W[c_idx[2]][col] = 1; W[c_idx[3]][col] = 1
        col += 1
        
        # M6: (A12-A22)(B21+B22)
        U[a_idx[1]][col] = 1; U[a_idx[3]][col] = -1
        V[b_idx[2]][col] = 1; V[b_idx[3]][col] = 1
        W[c_idx[1]][col] = 1; W[c_idx[3]][col] = 1
        col += 1
        
        # M7: (A21-A11)(B11+B12)
        U[a_idx[0]][col] = -1; U[a_idx[2]][col] = 1
        V[b_idx[0]][col] = 1; V[b_idx[1]][col] = 1
        W[c_idx[2]][col] = 1; W[c_idx[3]][col] = 1
        col += 1

    def add_schoolbook_2x2x1(a_idx, b_idx, c_idx):
        nonlocal col
        # A 2x2, B 2x1, C 2x1
        # b_idx: [B1, B2]
        # c_idx: [C1, C2]
        
        # M1: A11 B1
        U[a_idx[0]][col] = 1
        V[b_idx[0]][col] = 1
        W[c_idx[0]][col] = 1
        col += 1
        
        # M2: A12 B2
        U[a_idx[1]][col] = 1
        V[b_idx[1]][col] = 1
        W[c_idx[0]][col] = 1
        col += 1
        
        # M3: A21 B1
        U[a_idx[2]][col] = 1
        V[b_idx[0]][col] = 1
        W[c_idx[1]][col] = 1
        col += 1
        
        # M4: A22 B2
        U[a_idx[3]][col] = 1
        V[b_idx[1]][col] = 1
        W[c_idx[1]][col] = 1
        col += 1

    # Part 1: A1 B1
    # A1: U indices [0, 1, 2, 3]
    # B1: V indices for k=0,1 are [0, 1, 7, 8]? No.
    # V index = j*p + k.
    # B1 uses j=0,1.
    # k=0: j=0->V[0], j=1->V[7].
    # k=1: j=0->V[1], j=1->V[8].
    # So for k=0,1 block:
    # B11 (j=0,k=0) -> V[0]
    # B12 (j=0,k=1) -> V[1]
    # B21 (j=1,k=0) -> V[7]
    # B22 (j=1,k=1) -> V[8]
    
    # C1: W indices for k=0,1.
    # W index = k*n + i.
    # k=0: i=0->W[0], i=1->W[1].
    # k=1: i=0->W[2], i=1->W[3].
    # C11 (k=0,i=0) -> W[0]
    # C12 (k=1,i=0) -> W[2]
    # C21 (k=0,i=1) -> W[1]
    # C22 (k=1,i=1) -> W[3]
    
    a1_idx = [0, 1, 2, 3]
    
    # Block 1: k=0,1
    b1_idx = [0, 1, 7, 8]
    c1_idx = [0, 2, 1, 3]
    add_strassen(a1_idx, b1_idx, c1_idx)
    
    # Block 2: k=2,3
    # B11 (j=0,k=2) -> V[2]
    # B12 (j=0,k=3) -> V[3]
    # B21 (j=1,k=2) -> V[9]
    # B22 (j=1,k=3) -> V[10]
    b2_idx = [2, 3, 9, 10]
    # C11 (k=2,i=0) -> W[4]
    # C12 (k=3,i=0) -> W[6]
    # C21 (k=2,i=1) -> W[5]
    # C22 (k=3,i=1) -> W[7]
    c2_idx = [4, 6, 5, 7]
    add_strassen(a1_idx, b2_idx, c2_idx)
    
    # Block 3: k=4,5
    b3_idx = [4, 5, 11, 12]
    c3_idx = [8, 10, 9, 11]
    add_strassen(a1_idx, b3_idx, c3_idx)
    
    # Block 4: k=6
    # B1 (j=0,k=6) -> V[6]
    # B2 (j=1,k=6) -> V[13]
    b4_idx = [6, 13]
    # C1 (k=6,i=0) -> W[12]
    # C2 (k=6,i=1) -> W[13]
    c4_idx = [12, 13]
    add_schoolbook_2x2x1(a1_idx, b4_idx, c4_idx)

    # Part 2: A2 B2
    # A2: U indices [4, 5, 6, 7]
    # B2: V indices for j=2,3.
    # j=2: k=0->V[14], k=1->V[15], k=2->V[16], k=3->V[17], k=4->V[18], k=5->V[19], k=6->V[20]
    # j=3: k=0->V[21], k=1->V[22], k=2->V[23], k=3->V[24], k=4->V[25], k=5->V[26], k=6->V[27]
    
    a2_idx = [4, 5, 6, 7]
    
    # Block 1: k=0,1
    b5_idx = [14, 15, 21, 22]
    c5_idx = [0, 2, 1, 3]
    add_strassen(a2_idx, b5_idx, c5_idx)
    
    # Block 2: k=2,3
    b6_idx = [16, 17, 23, 24]
    c6_idx = [4, 6, 5, 7]
    add_strassen(a2_idx, b6_idx, c6_idx)
    
    # Block 3: k=4,5
    b7_idx = [18, 19, 25, 26]
    c7_idx = [8, 10, 9, 11]
    add_strassen(a2_idx, b7_idx, c7_idx)
    
    # Block 4: k=6
    b8_idx = [20, 27]
    c8_idx = [12, 13]
    add_schoolbook_2x2x1(a2_idx, b8_idx, c8_idx)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
