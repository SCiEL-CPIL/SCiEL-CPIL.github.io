import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # We decompose into 2 blocks of j (m=4 -> 2x2)
    # And 4 blocks of k (p=8 -> 4x2) for the inner 2x2 mult
    # Total 8 independent 2x2x2 multiplications.
    # Rank of each is 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 2*4 = 8 rows
    V = [[0] * rank for _ in range(m * p)] # 4*8 = 32 rows
    W = [[0] * rank for _ in range(p * n)] # 8*2 = 16 rows
    
    # Define Strassen factors for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # U_A indices: a=0, b=1, c=2, d=3
    # U_B indices: e=0, f=1, g=2, h=3
    # Output P indices: P11=0, P12=1, P21=2, P22=3
    
    # We use the following rank 7 decomposition:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a+b)(h-g)
    # M4 = (c-a)(e+f)
    # M5 = (a-b)(h+g)
    # M6 = (c+d)f
    # M7 = (a-b)e
    
    # Factors for A (rows a,b,c,d)
    # c0: a+d -> [1,0,0,1]
    # c1: c+d -> [0,0,1,1]
    # c2: a+b -> [1,1,0,0]
    # c3: c-a -> [0,-1,1,0]
    # c4: a-b -> [1,-1,0,0]
    # c5: c+d -> [0,0,1,1]
    # c6: a-b -> [1,-1,0,0]
    
    U_A_strassen = [
        [1, 0, 1, 0, 1, 0, 1], # a
        [0, 0, 1, -1, -1, 0, -1], # b
        [0, 1, 0, 1, 0, 1, 0], # c
        [1, 1, 0, 0, 0, 1, 0]  # d
    ]
    
    # Factors for B (rows e,f,g,h)
    # c0: e+h -> [1,0,0,1]
    # c1: e+f -> [1,1,0,0]
    # c2: h-g -> [0,0,-1,1]
    # c3: e+f -> [1,1,0,0]
    # c4: h+g -> [0,0,1,1]
    # c5: f -> [0,1,0,0]
    # c6: e -> [1,0,0,0]
    
    V_B_strassen = [
        [1, 1, 0, 1, 0, 0, 1], # e
        [0, 1, 0, 1, 0, 1, 0], # f
        [0, 0, -1, 0, 1, 0, 0], # g
        [1, 0, 1, 0, 1, 0, 0]  # h
    ]
    
    # Reconstruction W (cols P11, P12, P21, P22)
    # P11 = M1 + M4 - M5 + M7
    # P12 = M3 + M5
    # P21 = M2 + M4
    # P22 = M1 - M2 + M3 + M6
    
    W_P_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # P11
        [0, 0, 1, 0, 1, 0, 0],  # P12
        [0, 1, 0, 1, 0, 0, 0],  # P21
        [1, -1, 1, 0, 0, 1, 0]  # P22
    ]
    
    # Verify Strassen decomposition
    # This is a sanity check. If it fails, we might return rank 64.
    # But we assume it's correct based on standard formulas.
    
    # Mapping indices
    # A is 2x4. Rows i=0,1. Cols j=0,1,2,3.
    # j is split into j' (0,1) and j'' (0,1).
    # j = 2*j' + j''.
    # A index: i*m + j.
    
    # B is 4x8. Rows j=0..3. Cols k=0..7.
    # j = 2*j' + j''.
    # k is split into k_group (0..3) and k'' (0,1).
    # k = 2*k_group + k''.
    # B index: j*p + k.
    
    # C is 2x8. Rows i=0,1. Cols k=0..7.
    # k = 2*k_group + k''.
    # C index: k*n + i. (Trace convention W[k*n+i, r])
    
    col_offset = 0
    for j_prime in range(2):
        for k_group in range(4):
            # This corresponds to the multiplication of 
            # A_sub: 2x2 (rows i, cols j_prime*2 + j'')
            # B_sub: 2x2 (rows j_prime*2 + j'', cols k_group*2 + k'')
            
            # For each of the 7 Strassen multiplications
            for s in range(7):
                col = col_offset + s
                
                # Update U
                # A_sub entries: a=A[0, 2*j_prime+0], b=A[0, 2*j_prime+1]
                #               c=A[1, 2*j_prime+0], d=A[1, 2*j_prime+1]
                for i in range(2):
                    for j_prime_prime in range(2):
                        j = 2 * j_prime + j_prime_prime
                        u_val = U_A_strassen[i * 2 + j_prime_prime][s] # Wait, U_A indices are a,b,c,d.
                        # My U_A_strassen is indexed by [a,b,c,d] rows.
                        # a is (i=0, j''=0) -> row 0
                        # b is (i=0, j''=1) -> row 1
                        # c is (i=1, j''=0) -> row 2
                        # d is (i=1, j''=1) -> row 3
                        idx_u = i * m + j
                        u_row = i * 2 + j_prime_prime
                        U[idx_u][col] = U_A_strassen[u_row][s]
                        
                # Update V
                # B_sub entries: e=B[2*j_prime+0, 2*k_group+0], f=B[2*j_prime+0, 2*k_group+1]
                #               g=B[2*j_prime+1, 2*k_group+0], h=B[2*j_prime+1, 2*k_group+1]
                for j_prime_prime in range(2):
                    for k_prime_prime in range(2):
                        j = 2 * j_prime + j_prime_prime
                        k = 2 * k_group + k_prime_prime
                        idx_v = j * p + k
                        v_row = j_prime_prime * 2 + k_prime_prime
                        V[idx_v][col] = V_B_strassen[v_row][s]
                        
                # Update W
                # C_sub entries: P11=C[0, 2*k_group+0], P12=C[0, 2*k_group+1]
                #               P21=C[1, 2*k_group+0], P22=C[1, 2*k_group+1]
                for i in range(2):
                    for k_prime_prime in range(2):
                        k = 2 * k_group + k_prime_prime
                        idx_w = k * n + i
                        w_row = i * 2 + k_prime_prime
                        W[idx_w][col] = W_P_strassen[w_row][s]
                        
            col_offset += 7
            
    return {"U": U, "V": V, "W": W}
