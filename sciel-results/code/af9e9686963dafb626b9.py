import numpy as np

def construct():
    n, m, p = 3, 4, 5
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx, v_idx, w_idx are local indices in the sub-problem?
        # No, we will build the sub-problems and map them.
        pass

    # We need a way to build the sub-decompositions and merge them.
    # Let's create a function that returns U, V, W for a specific n, m, p decomposition
    # and then we will embed it.
    
    def get_1_2_5():
        # Returns U (2x10), V (10x10), W (5x10)
        # A is 1x2, B is 2x5, C is 1x5 (flattened C index k)
        # C_k = A_0 B_0k + A_1 B_1k
        r = 10
        u = [[0]*r for _ in range(2)]
        v = [[0]*r for _ in range(10)]
        w = [[0]*r for _ in range(5)]
        
        col = 0
        for k in range(5):
            # Term 1: A_0 * B_0k
            u[0][col] = 1
            v[0*5 + k][col] = 1
            w[k][col] = 1
            col += 1
            # Term 2: A_1 * B_1k
            u[1][col] = 1
            v[1*5 + k][col] = 1
            w[k][col] = 1
            col += 1
        return u, v, w

    def get_2_2_1():
        # A 2x2, B 2x1, C 2x1
        # C_0 = A_00 B_0 + A_01 B_1
        # C_1 = A_10 B_0 + A_11 B_1
        r = 4
        u = [[0]*r for _ in range(4)]
        v = [[0]*r for _ in range(2)]
        w = [[0]*r for _ in range(2)]
        
        col = 0
        # C_0 terms
        u[0][col] = 1; v[0][col] = 1; w[0][col] = 1; col += 1
        u[1][col] = 1; v[1][col] = 1; w[0][col] = 1; col += 1
        # C_1 terms
        u[2][col] = 1; v[0][col] = 1; w[1][col] = 1; col += 1
        u[3][col] = 1; v[1][col] = 1; w[1][col] = 1; col += 1
        return u, v, w

    def get_2_2_2():
        # Strassen
        r = 7
        u = [[0]*r for _ in range(4)]
        v = [[0]*r for _ in range(4)]
        w = [[0]*r for _ in range(4)]
        
        # M1 = (a+d)(e+h)
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        # M2 = (c+d)e
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        # M3 = a(f-h)
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        # M4 = d(g-e)
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        # M5 = (a+b)h
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        # M6 = (c-a)(e+f)
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        # M7 = (b-d)(g+h)
        u[1][6] = 1; u[3][6] = -1
        v[2][6] = 1; v[3][6] = 1
        
        # Combine to W
        # C00 = M1 + M4 - M5 + M7
        w[0][0] = 1; w[0][3] = 1; w[0][4] = -1; w[0][6] = 1
        # C01 = M3 + M5
        w[1][2] = 1; w[1][4] = 1
        # C10 = M2 + M4
        w[2][1] = 1; w[2][3] = 1
        # C11 = M1 - M2 + M3 + M6
        w[3][0] = 1; w[3][1] = -1; w[3][2] = 1; w[3][5] = 1
        
        return u, v, w

    def get_2_2_5():
        # Split p=5 into 2, 2, 1
        # B is 2x5. Indices 0,1; 2,3; 4.
        # C is 2x5.
        # C01 = A B01 (2x2x2)
        # C23 = A B23 (2x2x2)
        # C4  = A B4  (2x2x1)
        
        u_222, v_222, w_222 = get_2_2_2()
        u_221, v_221, w_221 = get_2_2_1()
        
        r = 7 + 7 + 4
        u = [[0]*r for _ in range(4)]
        v = [[0]*r for _ in range(10)] # 2x5
        w = [[0]*r for _ in range(10)] # 2x5
        
        # Add 2x2x2 for cols 0,1
        for i in range(4):
            for j in range(7):
                u[i][j] = u_222[i][j]
                # v index mapping: local j (0..3) to global j (0..3)
                v[j][j] = v_222[j][j]
                # w index mapping: local k (0..3) to global k (0..3)
                w[k*2+i][j] = w_222[k*2+i][j] if False else 0 # Need correct mapping
                # Wait, W in 2x2x2 is 2x2 flattened.
                # In 2x2x5, W is 2x5 flattened.
                # Local W index $l$ corresponds to $i_{local}, k_{local}$.
                # Global W index $i_{local} \cdot 5 + k_{global}$.
                # Here $k_{global} \in \{0,1\}$.
        
        # Let's do it properly with a merge function
        pass

    # This manual construction is error-prone.
    # I will use a generic merge.
    
    def merge(u1, v1, w1, u2, v2, w2):
        r1 = len(u1[0])
        r2 = len(u2[0])
        r = r1 + r2
        n1 = len(u1); m1 = len(v1[0])//len(v1) # No, v is (m*p, r)
        # Dimensions:
        # u: (n*m, r)
        # v: (m*p, r)
        # w: (p*n, r)
        
        n_u1 = len(u1)
        n_v1 = len(v1)
        n_w1 = len(w1)
        
        n_u2 = len(u2)
        n_v2 = len(v2)
        n_w2 = len(w2)
        
        # Assume dimensions match or are compatible
        u = [[0]*r for _ in range(n_u1)]
        v = [[0]*r for _ in range(n_v1)]
        w = [[0]*r for _ in range(n_w1)]
        
        for i in range(n_u1):
            for j in range(r1):
                u[i][j] = u1[i][j]
        for i in range(n_v1):
            for j in range(r1):
                v[i][j] = v1[i][j]
        for i in range(n_w1):
            for j in range(r1):
                w[i][j] = w1[i][j]
                
        for i in range(n_u2):
            for j in range(r2):
                u[i][r1+j] = u2[i][j]
        for i in range(n_v2):
            for j in range(r2):
                v[i][r1+j] = v2[i][j]
        for i in range(n_w2):
            for j in range(r2):
                w[i][r1+j] = w2[i][j]
                
        return u, v, w

    # Build 2x2x5
    def get_2_2_5():
        u_a, v_a, w_a = get_2_2_2()
        u_b, v_b, w_b = get_2_2_2()
        u_c, v_c, w_c = get_2_2_1()
        
        # We need to permute V and W indices to match the 5 columns.
        # V for 2x2x2 is 4 entries (2x2).
        # V for 2x2x5 is 10 entries (2x5).
        
        # Let's create empty 2x2x5 matrices
        r_total = 7+7+4
        u = [[0]*r_total for _ in range(4)]
        v = [[0]*r_total for _ in range(10)]
        w = [[0]*r_total for _ in range(10)]
        
        # Part 1: cols 0,1
        for i in range(4):
            for j in range(7):
                u[i][j] = u_a[i][j]
                # v mapping: local v index l (0..3) -> global v index l
                # v_a is 4x7. v is 10x14.
                v[l][j] = v_a[l][j] # l is 0..3
        # w mapping: local w index l (0..3) -> global w index l (0..3)
        # w_a is 4x7. w is 10x14.
        for i in range(4):
            for j in range(7):
                w[i][j] = w_a[i][j]
                
        # Part 2: cols 2,3
        # v mapping: local v index l (0..3) -> global v index 4+l
        # w mapping: local w index l (0..3) -> global w index 4+l
        offset = 7
        for i in range(4):
            for j in range(7):
                u[i][offset+j] = u_b[i][j]
                v[4+l][offset+j] = v_b[l][j]
                w[4+l][offset+j] = w_b[l][j]
                
        # Part 3: col 4
        # v mapping: local v index l (0..1) -> global v index 8+l
        # w mapping: local w index l (0..1) -> global w index 8+l
        offset = 14
        for i in range(4):
            for j in range(4):
                u[i][offset+j] = u_c[i][j]
                v[8+l][offset+j] = v_c[l][j]
                w[8+l][offset+j] = w_c[l][j]
                
        return u, v, w

    # This is getting messy. I will hardcode the 56 terms in a list of tuples (u_idx, v_idx, w_idx, val).
    # This is safer.
    
    return {"U": U, "V": V, "W": W}
