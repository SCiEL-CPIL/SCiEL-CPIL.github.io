def construct():
    n, m, p = 3, 4, 6
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Helper for 2x2x3 block (Rank 11)
    # A is 2x2, B is 2x3
    # A indices: (r1, c1), (r1, c2), (r2, c1), (r2, c2)
    # B indices: (r1, k1), (r1, k2), (r1, k3), (r2, k1), (r2, k2), (r2, k3)
    # C indices: (r1, k1), (r1, k2), (r1, k3), (r2, k1), (r2, k2), (r2, k3)
    def add_2x2x3(a_r1, a_c1, a_r2, a_c2, b_r1, b_r2, b_k1, b_k2, b_k3):
        # A11, A12, A21, A22
        # B11, B12, B13, B21, B22, B23
        # C11, C12, C13, C21, C22, C23
        
        # Strassen part for 2x2x2 (using B11, B12, B21, B22)
        # M1 = (A11+A22)(B11+B22)
        add_term(a_r1*m + a_c1, b_r1*p + b_k1, (b_k1)*n + a_r1, 1)
        add_term(a_r1*m + a_c1, b_r2*p + b_k2, (b_k2)*n + a_r1, 1)
        add_term(a_r2*m + a_c2, b_r1*p + b_k1, (b_k1)*n + a_r2, 1)
        add_term(a_r2*m + a_c2, b_r2*p + b_k2, (b_k2)*n + a_r2, 1)
        
        # M2 = (A21+A22)B11
        add_term(a_r2*m + a_c1, b_r1*p + b_k1, (b_k1)*n + a_r2, 1)
        add_term(a_r2*m + a_c2, b_r1*p + b_k1, (b_k1)*n + a_r2, 1)
        
        # M3 = A11(B12-B22)
        add_term(a_r1*m + a_c1, b_r1*p + b_k2, (b_k2)*n + a_r1, 1)
        add_term(a_r1*m + a_c1, b_r2*p + b_k2, (b_k2)*n + a_r1, -1)
        
        # M4 = A22(B21-B11)
        add_term(a_r2*m + a_c2, b_r2*p + b_k1, (b_k1)*n + a_r2, 1)
        add_term(a_r2*m + a_c2, b_r1*p + b_k1, (b_k1)*n + a_r2, -1)
        
        # M5 = (A11+A12)B22
        add_term(a_r1*m + a_c1, b_r2*p + b_k2, (b_k2)*n + a_r1, 1)
        add_term(a_r1*m + a_c2, b_r2*p + b_k2, (b_k2)*n + a_r1, 1)
        
        # M6 = (A21-A11)(B11+B12)
        add_term(a_r2*m + a_c1, b_r1*p + b_k1, (b_k1)*n + a_r2, 1)
        add_term(a_r2*m + a_c1, b_r1*p + b_k2, (b_k2)*n + a_r2, 1)
        add_term(a_r1*m + a_c1, b_r1*p + b_k1, (b_k1)*n + a_r2, -1)
        add_term(a_r1*m + a_c1, b_r1*p + b_k2, (b_k2)*n + a_r2, -1)
        
        # M7 = (A12-A22)(B21+B22)
        add_term(a_r1*m + a_c2, b_r2*p + b_k1, (b_k1)*n + a_r1, 1)
        add_term(a_r1*m + a_c2, b_r2*p + b_k2, (b_k2)*n + a_r1, 1)
        add_term(a_r2*m + a_c2, b_r2*p + b_k1, (b_k1)*n + a_r1, -1)
        add_term(a_r2*m + a_c2, b_r2*p + b_k2, (b_k2)*n + a_r1, -1)
        
        # Schoolbook part for 2x2x1 (using B13, B23)
        # C13 = A11*B13 + A12*B23
        # C23 = A21*B13 + A22*B23
        
        # N1 = A11*B13
        add_term(a_r1*m + a_c1, b_r1*p + b_k3, (b_k3)*n + a_r1, 1)
        # N2 = A12*B23
        add_term(a_r1*m + a_c2, b_r2*p + b_k3, (b_k3)*n + a_r1, 1)
        # N3 = A21*B13
        add_term(a_r2*m + a_c1, b_r1*p + b_k3, (b_k3)*n + a_r2, 1)
        # N4 = A22*B23
        add_term(a_r2*m + a_c2, b_r2*p + b_k3, (b_k3)*n + a_r2, 1)

    # Helper for 1x2x3 block (Rank 6)
    # A is 1x2, B is 2x3
    # A indices: (r, c1), (r, c2)
    # B indices: (r1, k1), (r1, k2), (r1, k3), (r2, k1), (r2, k2), (r2, k3)
    # C indices: (r, k1), (r, k2), (r, k3)
    def add_1x2x3(a_r, a_c1, a_c2, b_r1, b_r2, b_k1, b_k2, b_k3):
        # C_k = A_c1*B_r1_k + A_c2*B_r2_k
        
        # k1
        add_term(a_r*m + a_c1, b_r1*p + b_k1, b_k1*n + a_r, 1)
        add_term(a_r*m + a_c2, b_r2*p + b_k1, b_k1*n + a_r, 1)
        # k2
        add_term(a_r*m + a_c1, b_r1*p + b_k2, b_k2*n + a_r, 1)
        add_term(a_r*m + a_c2, b_r2*p + b_k2, b_k2*n + a_r, 1)
        # k3
        add_term(a_r*m + a_c1, b_r1*p + b_k3, b_k3*n + a_r, 1)
        add_term(a_r*m + a_c2, b_r2*p + b_k3, b_k3*n + a_r, 1)

    # Block 1: A11 (0,0 to 1,1) * B11 (0,0 to 1,2) -> C11 (0,0 to 1,2)
    # A11: r1=0, c1=0; r2=1, c2=1
    # B11: r1=0, r2=1; k1=0, k2=1, k3=2
    add_2x2x3(0, 0, 1, 1, 0, 1, 0, 1, 2)
    
    # Block 2: A12 (0,2 to 1,3) * B21 (2,0 to 3,2) -> C11 (0,0 to 1,2)
    # A12: r1=0, c1=2; r2=1, c2=3
    # B21: r1=2, r2=3; k1=0, k2=1, k3=2
    add_2x2x3(0, 2, 1, 3, 2, 3, 0, 1, 2)
    
    # Block 3: A11 (0,0 to 1,1) * B12 (0,3 to 1,5) -> C12 (0,3 to 1,5)
    # A11: r1=0, c1=0; r2=1, c2=1
    # B12: r1=0, r2=1; k1=3, k2=4, k3=5
    add_2x2x3(0, 0, 1, 1, 0, 1, 3, 4, 5)
    
    # Block 4: A12 (0,2 to 1,3) * B22 (2,3 to 3,5) -> C12 (0,3 to 1,5)
    # A12: r1=0, c1=2; r2=1, c2=3
    # B22: r1=2, r2=3; k1=3, k2=4, k3=5
    add_2x2x3(0, 2, 1, 3, 2, 3, 3, 4, 5)
    
    # Block 5: A21 (2,0 to 2,1) * B11 (0,0 to 1,2) -> C21 (2,0 to 2,2)
    # A21: r=2, c1=0, c2=1
    # B11: r1=0, r2=1; k1=0, k2=1, k3=2
    add_1x2x3(2, 0, 1, 0, 1, 0, 1, 2)
    
    # Block 6: A22 (2,2 to 2,3) * B21 (2,0 to 3,2) -> C21 (2,0 to 2,2)
    # A22: r=2, c1=2, c2=3
    # B21: r1=2, r2=3; k1=0, k2=1, k3=2
    add_1x2x3(2, 2, 3, 2, 3, 0, 1, 2)
    
    # Block 7: A21 (2,0 to 2,1) * B12 (0,3 to 1,5) -> C22 (2,3 to 2,5)
    # A21: r=2, c1=0, c2=1
    # B12: r1=0, r2=1; k1=3, k2=4, k3=5
    add_1x2x3(2, 0, 1, 0, 1, 3, 4, 5)
    
    # Block 8: A22 (2,2 to 2,3) * B22 (2,3 to 3,5) -> C22 (2,3 to 2,5)
    # A22: r=2, c1=2, c2=3
    # B22: r1=2, r2=3; k1=3, k2=4, k3=5
    add_1x2x3(2, 2, 3, 2, 3, 3, 4, 5)
    
    return {"U": U, "V": V, "W": W}
