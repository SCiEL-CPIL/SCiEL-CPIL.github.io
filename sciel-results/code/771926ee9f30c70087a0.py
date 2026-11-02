def construct():
    n, m, p = 2, 4, 5
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        for idx, val in u_dict.items():
            U_col[idx] = val
        for idx, val in v_dict.items():
            V_col[idx] = val
        for idx, val in w_dict.items():
            W_col[idx] = val
        terms.append((U_col, V_col, W_col))
    
    def a_idx(i, j):
        return i * m + j  # i in 0..1, j in 0..3
    
    def b_idx(j, k):
        return j * p + k  # j in 0..3, k in 0..4
    
    def c_idx(i, k):
        return k * n + i  # i in 0..1, k in 0..4 (trace convention)
    
    def strassen_2x2x2(a_i0, a_j0, b_j0, b_k0):
        """Strassen for A[a_i0:a_i0+2, a_j0:a_j0+2] @ B[b_j0:b_j0+2, b_k0:b_k0+2]"""
        # a=A[a_i0,a_j0], b=A[a_i0,a_j0+1], c=A[a_i0+1,a_j0], d=A[a_i0+1,a_j0+1]
        # e=B[b_j0,b_k0], f=B[b_j0,b_k0+1], g=B[b_j0+1,b_k0], h=B[b_j0+1,b_k0+1]
        
        # m1 = (a+d)(e+h)
        add_term({a_idx(a_i0, a_j0): 1, a_idx(a_i0+1, a_j0+1): 1},
                 {b_idx(b_j0, b_k0): 1, b_idx(b_j0+1, b_k0+1): 1},
                 {c_idx(a_i0, b_k0): 1, c_idx(a_i0+1, b_k0+1): 1})
        
        # m2 = (c+d)e
        add_term({a_idx(a_i0+1, a_j0): 1, a_idx(a_i0+1, a_j0+1): 1},
                 {b_idx(b_j0, b_k0): 1},
                 {c_idx(a_i0+1, b_k0): 1})
        
        # m3 = a(f-h)
        add_term({a_idx(a_i0, a_j0): 1},
                 {b_idx(b_j0, b_k0+1): 1, b_idx(b_j0+1, b_k0+1): -1},
                 {c_idx(a_i0, b_k0): 1, c_idx(a_i0, b_k0+1): 1})
        
        # m4 = d(g-e)
        add_term({a_idx(a_i0+1, a_j0+1): 1},
                 {b_idx(b_j0+1, b_k0): 1, b_idx(b_j0, b_k0): -1},
                 {c_idx(a_i0, b_k0): 1, c_idx(a_i0+1, b_k0+1): 1})
        
        # m5 = (a+b)h
        add_term({a_idx(a_i0, a_j0): 1, a_idx(a_i0, a_j0+1): 1},
                 {b_idx(b_j0+1, b_k0+1): 1},
                 {c_idx(a_i0, b_k0): -1, c_idx(a_i0+1, b_k0+1): 1})
        
        # m6 = (c-a)(e+f)
        add_term({a_idx(a_i0+1, a_j0): 1, a_idx(a_i0, a_j0): -1},
                 {b_idx(b_j0, b_k0): 1, b_idx(b_j0, b_k0+1): 1},
                 {c_idx(a_i0+1, b_k0+1): 1})
        
        # m7 = (b-d)(g+h)
        add_term({a_idx(a_i0, a_j0+1): 1, a_idx(a_i0+1, a_j0+1): -1},
                 {b_idx(b_j0+1, b_k0): 1, b_idx(b_j0+1, b_k0+1): 1},
                 {c_idx(a_i0, b_k0): 1, c_idx(a_i0+1, b_k0+1): -1})
    
    def schoolbook_2x2x1(a_i0, a_j0, b_j0, b_k0):
        """Schoolbook for A[a_i0:a_i0+2, a_j0:a_j0+2] @ B[b_j0:b_j0+2, b_k0:b_k0+1]"""
        for i in range(2):
            for j in range(2):
                add_term({a_idx(a_i0+i, a_j0+j): 1},
                         {b_idx(b_j0+j, b_k0): 1},
                         {c_idx(a_i0+i, b_k0): 1})
    
    # Top block: A rows 0-1, cols 0-1; B rows 0-1
    strassen_2x2x2(0, 0, 0, 0)  # 7 terms, cols 0-1
    strassen_2x2x2(0, 0, 0, 2)  # 7 terms, cols 2-3
    schoolbook_2x2x1(0, 0, 0, 4)  # 4 terms, col 4
    
    # Bottom block: A rows 0-1, cols 2-3; B rows 2-3
    strassen_2x2x2(0, 2, 2, 0)  # 7 terms, cols 0-1
    strassen_2x2x2(0, 2, 2, 2)  # 7 terms, cols 2-3
    schoolbook_2x2x1(0, 2, 2, 4)  # 4 terms, col 4
    
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u_col, v_col, w_col) in enumerate(terms):
        for i in range(n*m):
            U[i][c] = u_col[i]
        for i in range(m*p):
            V[i][c] = v_col[i]
        for i in range(p*n):
            W[i][c] = w_col[i]
    
    return {"U": U, "V": V, "W": W}
