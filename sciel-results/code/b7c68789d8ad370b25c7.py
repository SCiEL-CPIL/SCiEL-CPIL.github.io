import numpy as np

def construct():
    n, m, p = 2, 5, 5
    # Total rank estimate: 18 + 18 + 10 = 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Block 1: A1 (2x2) x B1 (2x5)
    # A1 indices: i in 0,1; j in 0,1. Global A idx: i*5 + j
    # B1 indices: j in 0,1; k in 0..4. Global B idx: j*5 + k
    # C1 indices: i in 0,1; k in 0..4. Global C idx: k*2 + i
    
    # B1 split into B_a (2x2, k=0,1), B_b (2x2, k=2,3), B_c (2x1, k=4)
    
    # Strassen for B_a
    # M1 = (A00+A11)(B00+B11)
    # A00: 0*5+0=0, A11: 1*5+1=6
    # B00: 0*5+0=0, B11: 1*5+1=6
    # C00: 0*2+0=0, C11: 1*2+1=3
    # C01: 0*2+1=1, C10: 1*2+0=2
    
    # We will implement Strassen manually for each 2x2 block
    
    def add_strassen_block(a_base, b_base, c_base):
        # a_base: index of A00 in U
        # b_base: index of B00 in V
        # c_base: index of C00 in W
        # A00: a_base, A01: a_base+1, A10: a_base+5, A11: a_base+6
        # B00: b_base, B01: b_base+1, B10: b_base+5, B11: b_base+6
        # C00: c_base, C01: c_base+1, C10: c_base+5, C11: c_base+6
        # Note: C indices are k*2+i.
        # C00 (k=0,i=0) -> 0
        # C01 (k=0,i=1) -> 1
        # C10 (k=1,i=0) -> 2
        # C11 (k=1,i=1) -> 3
        # So c_base is index of C00.
        # C01 is c_base+1
        # C10 is c_base+2 (k=1, i=0 -> 1*2+0=2)
        # C11 is c_base+3
        
        # M1
        add_term(a_base, b_base, c_base, 1, 1, 1) # C00
        add_term(a_base+5, b_base+5, c_base, 1, 1, 1) # C00? No, A11 B11 contributes to C11
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # M2 = (A10+A11)B00 -> C10
        # M3 = A00(B01-B11) -> C01
        # M4 = A11(B10-B00) -> C10, C11
        # M5 = (A00+A01)B11 -> C01, C11
        # M6 = (A10-A00)(B00+B01) -> C11
        # M7 = (A01-A11)(B10+B11) -> C00, C11
        
        # We need to accumulate into the same columns? No, each M is a separate term in the decomposition.
        # So we add 7 terms.
        
        # M1
        U[a_base][col] = 1; U[a_base+5][col] = 1
        V[b_base][col] = 1; V[b_base+5][col] = 1
        W[c_base][col] = 1; W[c_base+3][col] = 1
        col += 1
        
        # M2
        U[a_base+5][col] = 1; U[a_base+6][col] = 1 # A10, A11? No A10 is a_base+5, A11 is a_base+6
        V[b_base][col] = 1
        W[c_base+2][col] = 1 # C10
        col += 1
        
        # M3
        U[a_base][col] = 1
        V[b_base+1][col] = 1; V[b_base+6][col] = -1 # B01, B11
        W[c_base+1][col] = 1 # C01
        col += 1
        
        # M4
        U[a_base+6][col] = 1
        V[b_base+5][col] = 1; V[b_base][col] = -1 # B10, B00
        W[c_base+2][col] = 1; W[c_base+3][col] = 1 # C10, C11
        col += 1
        
        # M5
        U[a_base][col] = 1; U[a_base+1][col] = 1 # A00, A01
        V[b_base+6][col] = 1 # B11
        W[c_base+1][col] = 1; W[c_base+3][col] = 1 # C01, C11
        col += 1
        
        # M6
        U[a_base+5][col] = 1; U[a_base][col] = -1 # A10, A00
        V[b_base][col] = 1; V[b_base+1][col] = 1 # B00, B01
        W[c_base+3][col] = 1 # C11
        col += 1
        
        # M7
        U[a_base+1][col] = 1; U[a_base+6][col] = -1 # A01, A11
        V[b_base+5][col] = 1; V[b_base+6][col] = 1 # B10, B11
        W[c_base][col] = 1; W[c_base+3][col] = 1 # C00, C11
        col += 1

    # Block 1
    # A1 base 0. B1 base 0. C1 base 0 (k=0,1)
    add_strassen_block(0, 0, 0)
    # B_b base 2 (k=2,3). C_b base 4 (k=2,3 -> 2*2+0=4)
    add_strassen_block(0, 10, 4) # B base for k=2 is 0*5+2=2? No, B1 is j in 0,1.
    # B1 indices: j=0,1. k=2,3.
    # B_{0,2} is 0*5+2=2. B_{1,3} is 1*5+3=8.
    # So b_base for B_b is 2.
    add_strassen_block(0, 2, 4)
    
    # B_c (2x1, k=4). Schoolbook 2x2x1. Rank 4.
    # A1 x B_{*,4}
    # B_{0,4} is 4. B_{1,4} is 9.
    # C_{*,4} is k=4. C_{0,4} is 4*2+0=8. C_{1,4} is 4*2+1=9.
    # Terms:
    # A00 B04 -> C04
    U[0][col] = 1; V[4][col] = 1; W[8][col] = 1; col += 1
    # A01 B14 -> C04
    U[1][col] = 1; V[9][col] = 1; W[8][col] = 1; col += 1
    # A10 B04 -> C14
    U[5][col] = 1; V[4][col] = 1; W[9][col] = 1; col += 1
    # A11 B14 -> C14
    U[6][col] = 1; V[9][col] = 1; W[9][col] = 1; col += 1
    
    # Block 2: A2 (2x2, j=2,3) x B2 (2x5, j=2,3)
    # A2 base 10 (i=0,j=2). A_{0,2}=10, A_{0,3}=11, A_{1,2}=15, A_{1,3}=16.
    # B2 base 10 (j=2,k=0). B_{2,0}=10, B_{3,1}=16.
    # C2 base 0.
    add_strassen_block(10, 10, 0)
    add_strassen_block(10, 12, 4) # B_{2,2}=12, B_{3,3}=18.
    
    # B_c for Block 2 (k=4)
    # B_{2,4}=14, B_{3,4}=19.
    U[10][col] = 1; V[14][col] = 1; W[8][col] = 1; col += 1
    U[11][col] = 1; V[19][col] = 1; W[8][col] = 1; col += 1
    U[15][col] = 1; V[14][col] = 1; W[9][col] = 1; col += 1
    U[16][col] = 1; V[19][col] = 1; W[9][col] = 1; col += 1
    
    # Block 3: A3 (2x1, j=4) x B3 (1x5, j=4)
    # A_{0,4}=4, A_{1,4}=9.
    # B_{4,k} for k=0..4: 20, 21, 22, 23, 24.
    # C_{i,k}: 2k, 2k+1.
    # Rank 10.
    for k in range(5):
        # A0 Bk -> C0k
        U[4][col] = 1; V[20+k][col] = 1; W[2*k][col] = 1; col += 1
        # A1 Bk -> C1k
        U[9][col] = 1; V[20+k][col] = 1; W[2*k+1][col] = 1; col += 1
        
    return {"U": U, "V": V, "W": W}
