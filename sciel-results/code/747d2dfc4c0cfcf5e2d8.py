import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    # Schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is part of a Strassen block
                is_strassen = False
                # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
                # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
                # Block 3: i in {0,1}, j in {0,1}, k in {4,5}
                # Block 4: i in {0,1}, j in {2,3}, k in {0,1}
                # Block 5: i in {0,1}, j in {2,3}, k in {2,3}
                # Block 6: i in {0,1}, j in {2,3}, k in {4,5}
                
                if i < 2 and j < 2 and k < 2:
                    is_strassen = True
                elif i < 2 and j < 2 and 2 <= k < 4:
                    is_strassen = True
                elif i < 2 and j < 2 and 4 <= k < 6:
                    is_strassen = True
                elif i < 2 and 2 <= j < 4 and k < 2:
                    is_strassen = True
                elif i < 2 and 2 <= j < 4 and 2 <= k < 4:
                    is_strassen = True
                elif i < 2 and 2 <= j < 4 and 4 <= k < 6:
                    is_strassen = True
                
                if not is_strassen:
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    add_term(u, v, w)

    # Strassen blocks
    # Define blocks: (i_start, j_start, k_start)
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    for i_s, j_s, k_s in blocks:
        # Indices
        i0, i1 = i_s, i_s + 1
        j0, j1 = j_s, j_s + 1
        k0, k1 = k_s, k_s + 1
        
        # A indices: a=(i0,j0), b=(i0,j1), c=(i1,j0), d=(i1,j1)
        # B indices: e=(j0,k0), f=(j0,k1), g=(j1,k0), h=(j1,k1)
        
        def get_a(i, j): return i*m + j
        def get_b(j, k): return j*p + k
        def get_c(i, k): return k*n + i
        
        # M1 = (a+d)(e+h)
        # Expand: a(e+h) + d(e+h) = ae + ah + de + dh
        # This is not a single term in the bilinear form sense unless we consider the product of sums.
        # The tensor decomposition requires linear forms.
        # M1 is a product of two linear forms.
        # L1 = a + d
        # L2 = e + h
        # Term: L1 * L2
        # In U, V, W, this corresponds to one column where U has 1s at a and d, V has 1s at e and h, W has 1 at C11?
        # No, W depends on the output.
        # C11 = M1 + M4 - M5 + M7
        # So M1 contributes +1 to C11.
        # M4 contributes +1 to C11.
        # M5 contributes -1 to C11.
        # M7 contributes +1 to C11.
        
        # We need to add 7 terms for the block.
        # Term 1: L1=L_A1, L2=L_B1, L_C=1 at (i0,k0)
        # L_A1 = A[i0,j0] + A[i1,j1]
        # L_B1 = B[j0,k0] + B[j1,k1]
        
        # Let's define the 7 terms explicitly.
        
        # 1. (A_i0_j0 + A_i1_j1) * (B_j0_k0 + B_j1_k1) -> C_i0_k0 (+1)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[get_a(i0,j0)] = 1; u[get_a(i1,j1)] = 1
        v[get_b(j0,k0)] = 1; v[get_b(j1,k1)] = 1
        w[get_c(i0,k0)] = 1
        add_term(u, v, w)
        
        # 2. (A_i1_j0 + A_i1_j1) * B_j0_k0 -> C_i1_k0 (+1)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[get_a(i1,j0)] = 1; u[get_a(i1,j1)] = 1
        v[get_b(j0,k0)] = 1
        w[get_c(i1,k0)] = 1
        add_term(u, v, w)
        
        # 3. A_i0_j0 * (B_j0_k1 - B_j1_k1) -> C_i0_k1 (+1)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[get_a(i0,j0)] = 1
        v[get_b(j0,k1)] = 1; v[get_b(j1,k1)] = -1
        w[get_c(i0,k1)] = 1
        add_term(u, v, w)
        
        # 4. A_i1_j1 * (B_j1_k0 - B_j0_k0) -> C_i1_k0 (+1) AND C_i0_k0 (+1)?
        # M4 = d(g-e).
        # C11 += M4. C21 += M4.
        # So this term contributes to both C_i0_k0 and C_i1_k0?
        # No, C11 is (i0,k0). C21 is (i1,k0).
        # So w has 1 at (i0,k0) and 1 at (i1,k0).
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[get_a(i1,j1)] = 1
        v[get_b(j1,k0)] = 1; v[get_b(j0,k0)] = -1
        w[get_c(i0,k0)] = 1
        w[get_c(i1,k0)] = 1
        add_term(u, v, w)
        
        # 5. (A_i0_j0 + A_i0_j1) * B_j1_k1 -> C_i0_k0 (-1)
        # M5 = (a+b)h. C11 -= M5.
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[get_a(i0,j0)] = 1; u[get_a(i0,j1)] = 1
        v[get_b(j1,k1)] = 1
        w[get_c(i0,k0)] = -1
        add_term(u, v, w)
        
        # 6. (A_i1_j0 - A_i0_j0) * (B_j0_k0 + B_j0_k1) -> C_i1_k1 (+1)
        # M6 = (c-a)(e+f). C22 += M6.
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[get_a(i1,j0)] = 1; u[get_a(i0,j0)] = -1
        v[get_b(j0,k0)] = 1; v[get_b(j0,k1)] = 1
        w[get_c(i1,k1)] = 1
        add_term(u, v, w)
        
        # 7. (A_i0_j1 - A_i1_j1) * (B_j1_k0 + B_j1_k1) -> C_i0_k0 (+1)
        # M7 = (b-d)(g+h). C11 += M7.
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[get_a(i0,j1)] = 1; u[get_a(i1,j1)] = -1
        v[get_b(j1,k0)] = 1; v[get_b(j1,k1)] = 1
        w[get_c(i0,k0)] = 1
        add_term(u, v, w)
        
    rank = len(U)
    return {"U": U, "V": V, "W": W}

