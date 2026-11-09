import numpy as np

def construct():
    n, m, p = 3, 5, 6
    
    U = []
    V = []
    W = []
    
    def add_term(u_list, v_list, w_list):
        U.append(u_list)
        V.append(v_list)
        W.append(w_list)

    # Helper to create zero lists
    def zeros(size):
        return [0.0] * size

    # Indices:
    # U: i*5 + j
    # V: j*6 + k
    # W: k*3 + i

    # Part 1: i=0 (1x5x6)
    # Rank 30
    for j in range(5):
        for k in range(6):
            u = zeros(15)
            v = zeros(30)
            w = zeros(18)
            
            u[0*5 + j] = 1.0
            v[j*6 + k] = 1.0
            w[k*3 + 0] = 1.0
            
            add_term(u, v, w)

    # Part 2: i=1,2 (2x5x6)
    # Split m=5 into m1=2, m2=3.
    # Subpart 2a: i=1,2; j=0,1; k=0..5 (2x2x6)
    # Use Strassen for 3 blocks of 2x2x2.
    # k split: 0-1, 2-3, 4-5.
    
    # Strassen 2x2x2 factors
    # M1: (A11+A22)(B11+B22) -> C11(+1), C22(+1)
    # M2: (A21+A22)B11 -> C21(+1)
    # M3: A11(B21-B22) -> C12(+1), C21(-1)
    # M4: A22(B12-B11) -> C11(+1), C22(-1)
    # M5: (A11+A12)B22 -> C11(-1), C12(+1)
    # M6: (A21-A11)(B11+B12) -> C22(+1)
    # M7: (A12-A22)(B21+B22) -> C11(+1), C22(-1)
    
    # Local indices for 2x2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Global mapping for Subpart 2a (j=0,1; k block start ks)
    # i_local 0 -> i_global 1
    # i_local 1 -> i_global 2
    # j_local 0 -> j_global 0
    # j_local 1 -> j_global 1
    # k_local 0 -> k_global ks
    # k_local 1 -> k_global ks+1
    
    def add_strassen_block(i_base, j_base, k_base):
        # i_base: 1
        # j_base: 0
        # k_base: 0, 2, 4
        
        # M1
        u = zeros(15); v = zeros(30); w = zeros(18)
        u[i_base*5 + j_base] = 1.0      # A11
        u[(i_base+1)*5 + (j_base+1)] = 1.0 # A22
        v[j_base*6 + k_base] = 1.0      # B11
        v[(j_base+1)*6 + (k_base+1)] = 1.0 # B22
        w[k_base*3 + i_base] = 1.0      # C11
        w[(k_base+1)*3 + (i_base+1)] = 1.0 # C22
        add_term(u, v, w)
        
        # M2
        u = zeros(15); v = zeros(30); w = zeros(18)
        u[(i_base+1)*5 + j_base] = 1.0      # A21
        u[(i_base+1)*5 + (j_base+1)] = 1.0 # A22
        v[j_base*6 + k_base] = 1.0      # B11
        w[(k_base)*3 + (i_base+1)] = 1.0 # C21
        add_term(u, v, w)
        
        # M3
        u = zeros(15); v = zeros(30); w = zeros(18)
        u[i_base*5 + j_base] = 1.0      # A11
        v[(j_base+1)*6 + k_base] = 1.0      # B21
        v[(j_base+1)*6 + (k_base+1)] = -1.0 # B22
        w[k_base*3 + i_base] = 1.0      # C12? No, C12 is k_base+1, i_base
        # Wait, C indices: C_{i,k}.
        # C12 is i=0, k=1 in local. Global i=i_base, k=k_base+1.
        # C21 is i=1, k=0 in local. Global i=i_base+1, k=k_base.
        w[(k_base+1)*3 + i_base] = 1.0
        w[k_base*3 + (i_base+1)] = -1.0
        add_term(u, v, w)
        
        # M4
        u = zeros(15); v = zeros(30); w = zeros(18)
        u[(i_base+1)*5 + (j_base+1)] = 1.0 # A22
        v[j_base*6 + (k_base+1)] = 1.0      # B12
        v[j_base*6 + k_base] = -1.0      # B11
        w[k_base*3 + i_base] = 1.0      # C11
        w[(k_base+1)*3 + (i_base+1)] = -1.0 # C22
        add_term(u, v, w)
        
        # M5
        u = zeros(15); v = zeros(30); w = zeros(18)
        u[i_base*5 + j_base] = 1.0      # A11
        u[i_base*5 + (j_base+1)] = 1.0 # A12
        v[(j_base+1)*6 + (k_base+1)] = 1.0 # B22
        w[k_base*3 + i_base] = -1.0      # C11
        w[(k_base+1)*3 + i_base] = 1.0 # C12
        add_term(u, v, w)
        
        # M6
        u = zeros(15); v = zeros(30); w = zeros(18)
        u[(i_base+1)*5 + j_base] = 1.0      # A21
        u[i_base*5 + j_base] = -1.0      # A11
        v[j_base*6 + k_base] = 1.0      # B11
        v[j_base*6 + (k_base+1)] = 1.0 # B12
        w[(k_base+1)*3 + (i_base+1)] = 1.0 # C22
        add_term(u, v, w)
        
        # M7
        u = zeros(15); v = zeros(30); w = zeros(18)
        u[i_base*5 + (j_base+1)] = 1.0 # A12
        u[(i_base+1)*5 + (j_base+1)] = -1.0 # A22
        v[(j_base+1)*6 + k_base] = 1.0      # B21
        v[(j_base+1)*6 + (k_base+1)] = 1.0 # B22
        w[k_base*3 + i_base] = 1.0      # C11
        w[(k_base+1)*3 + (i_base+1)] = -1.0 # C22
        add_term(u, v, w)

    # Call for 3 blocks
    add_strassen_block(1, 0, 0)
    add_strassen_block(1, 0, 2)
    add_strassen_block(1, 0, 4)
    
    # Subpart 2b: i=1,2; j=2,3,4; k=0..5 (2x3x6)
    # Split k into two 3s: 0-2, 3-5.
    # For each 2x3x3, split j into 1+2.
    # j=2 (1x3) and j=3,4 (2x3).
    
    def add_2x3x3_block(i_base, j_base, k_base):
        # j_base: 2 or 3? No, j_base is start of the 3 j's.
        # Here we call for j_base=2, k_base=0 and k_base=3.
        
        # Part 1: j=0 local (global j_base)
        # 2x1x3. Rank 3? No, 2x1x3 tensor rank is 1*3=3?
        # T_{j,k} = A_{ij} B_{jk}.
        # For fixed j, rank is 3 (one for each k).
        for k_loc in range(3):
            u = zeros(15); v = zeros(30); w = zeros(18)
            u[i_base*5 + (j_base)] = 1.0 # A11? No, A is 2x1.
            # A_{1, j_base} and A_{2, j_base}
            # We need to separate i.
            # Term for i=1: A_{1, j_base} B_{j_base, k}
            # Term for i=2: A_{2, j_base} B_{j_base, k}
            # So we need 2 terms per k?
            # Yes.
            pass
        
        # Let's do i=1 and i=2 separately for the 1x1x3 part.
        for i_loc in range(2):
            i_glob = i_base + i_loc
            for k_loc in range(3):
                k_glob = k_base + k_loc
                u = zeros(15); v = zeros(30); w = zeros(18)
                u[i_glob*5 + j_base] = 1.0
                v[j_base*6 + k_glob] = 1.0
                w[k_glob*3 + i_glob] = 1.0
                add_term(u, v, w)
                
        # Part 2: j=1,2 local (global j_base+1, j_base+2)
        # 2x2x3.
        # Split k into 1+2.
        # k=0 local (global k_base): 2x2x1. Rank 4.
        # k=1,2 local (global k_base+1, k_base+2): 2x2x2. Rank 7 (Strassen).
        
        # 2x2x1
        for i_loc in range(2):
            i_glob = i_base + i_loc
            for j_loc in range(2):
                j_glob = j_base + 1 + j_loc
                u = zeros(15); v = zeros(30); w = zeros(18)
                u[i_glob*5 + j_glob] = 1.0
                v[j_glob*6 + k_base] = 1.0
                w[k_base*3 + i_glob] = 1.0
                add_term(u, v, w)
                
        # 2x2x2 (Strassen)
        # j_local 0,1 -> global j_base+1, j_base+2
        # k_local 0,1 -> global k_base+1, k_base+2
        add_strassen_block(i_base, j_base+1, k_base+1)

    add_2x3x3_block(1, 2, 0)
    add_2x3x3_block(1, 2, 3)
    
    # Convert to lists
    return {"U": U, "V": V, "W": W}
