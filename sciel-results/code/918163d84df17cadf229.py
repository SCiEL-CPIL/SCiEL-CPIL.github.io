import time

def construct():
    n, m, p = 3, 4, 6
    # We will construct a decomposition of rank 66.
    # Part 1: i=2 (1x4x6) schoolbook, rank 24
    # Part 2: i in {0,1} (2x4x6)
    #   Split j into {0,1} and {2,3}
    #   For each j-block (2x2x6), split k into {0,1}, {2,3}, {4,5}
    #   For each k-block (2x2x2), use Strassen (rank 7)
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Part 1: i=2, schoolbook for j=0..3, k=0..5
    for j in range(4):
        for k in range(6):
            u = [0] * (n*m)
            v = [0] * (m*p)
            w = [0] * (p*n)
            u[2*m + j] = 1
            v[j*p + k] = 1
            w[k*n + 2] = 1
            add_term(u, v, w)
            
    # Part 2: i in {0,1}
    # j-blocks: {0,1} and {2,3}
    # k-blocks: {0,1}, {2,3}, {4,5}
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    strassen_coeffs = [
        # (u_indices, v_indices, w_indices, coeffs)
        # u: 2x2, v: 2x2, w: 2x2
        # u[0]=A00, u[1]=A01, u[2]=A10, u[3]=A11
        # v[0]=B00, v[1]=B01, v[2]=B10, v[3]=B11
        # w[0]=C00, w[1]=C01, w[2]=C10, w[3]=C11
        
        # M1
        ([1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], 1),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0], 1),
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1], 1),
        # M4
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 0, 0], 1),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], 1),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1], 1),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, 0], 1),
    ]
    
    for j_block in range(2): # 0: {0,1}, 1: {2,3}
        for k_block in range(3): # 0: {0,1}, 1: {2,3}, 2: {4,5}
            j0 = j_block * 2
            j1 = j_block * 2 + 1
            k0 = k_block * 2
            k1 = k_block * 2 + 1
            
            for coeff in strassen_coeffs:
                u_local, v_local, w_local, scale = coeff
                u = [0] * (n*m)
                v = [0] * (m*p)
                w = [0] * (p*n)
                
                # Map local indices to global
                # u: i in {0,1}, j in {j0, j1}
                # u_local[0] -> i=0, j=j0
                # u_local[1] -> i=0, j=j1
                # u_local[2] -> i=1, j=j0
                # u_local[3] -> i=1, j=j1
                u[0*m + j0] = u_local[0] * scale
                u[0*m + j1] = u_local[1] * scale
                u[1*m + j0] = u_local[2] * scale
                u[1*m + j1] = u_local[3] * scale
                
                # v: j in {j0, j1}, k in {k0, k1}
                # v_local[0] -> j=j0, k=k0
                # v_local[1] -> j=j0, k=k1
                # v_local[2] -> j=j1, k=k0
                # v_local[3] -> j=j1, k=k1
                v[j0*p + k0] = v_local[0] * scale
                v[j0*p + k1] = v_local[1] * scale
                v[j1*p + k0] = v_local[2] * scale
                v[j1*p + k1] = v_local[3] * scale
                
                # w: k in {k0, k1}, i in {0,1}
                # w_local[0] -> k=k0, i=0
                # w_local[1] -> k=k0, i=1
                # w_local[2] -> k=k1, i=0
                # w_local[3] -> k=k1, i=1
                # Note: W[k*n+i, r]
                w[k0*n + 0] = w_local[0] * scale
                w[k0*n + 1] = w_local[1] * scale
                w[k1*n + 0] = w_local[2] * scale
                w[k1*n + 1] = w_local[3] * scale
                
                add_term(u, v, w)
                
    return {"U": U, "V": V, "W": W}
