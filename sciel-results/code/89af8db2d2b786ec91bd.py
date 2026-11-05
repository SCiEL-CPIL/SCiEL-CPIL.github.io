import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = DIMENSIONS
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_vals, v_vals, w_vals):
        nonlocal rank
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for idx, val in u_vals:
            u_col[idx] = val
        for idx, val in v_vals:
            v_col[idx] = val
        for idx, val in w_vals:
            w_col[idx] = val
            
        U.append(u_col)
        V.append(v_col)
        W.append(w_col)
        rank += 1
        
    # Strassen blocks
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {2,3}, j in {0,1}, k in {0,1}
    # Block 3: i in {0,1}, j in {2,3}, k in {2,3}
    # Block 4: i in {2,3}, j in {2,3}, k in {2,3}
    
    strassen_blocks = [
        (0, 0, 0),
        (2, 0, 0),
        (0, 2, 2),
        (2, 2, 2)
    ]
    
    for i0, j0, k0 in strassen_blocks:
        # 7 terms for Strassen
        # 1
        add_term([(i0*m+j0, 1), ((i0+1)*m+(j0+1), 1)], 
                 [(j0*p+k0, 1), ((j0+1)*p+(k0+1), 1)], 
                 [(k0*n+i0, 1), ((k0+1)*n+(i0+1), 1)])
        # 2
        add_term([( (i0+1)*m+j0, 1), ((i0+1)*m+(j0+1), 1)], 
                 [(j0*p+k0, 1)], 
                 [(k0*n+(i0+1), 1), ((k0+1)*n+(i0+1), -1)])
        # 3
        add_term([(i0*m+j0, 1)], 
                 [((j0+1)*p+k0, 1), ((j0+1)*p+(k0+1), -1)], 
                 [((k0+1)*n+i0, 1), ((k0+1)*n+(i0+1), 1)])
        # 4
        add_term([((i0+1)*m+(j0+1), 1)], 
                 [(j0*p+(k0+1), 1), ((j0+1)*p+(k0+1), 1)], 
                 [(k0*n+i0, 1), (k0*n+(i0+1), 1)])
        # 5
        add_term([(i0*m+j0, 1), ((i0+1)*m+j0, 1)], 
                 [((j0+1)*p+(k0+1), 1)], 
                 [(k0*n+i0, -1), ((k0+1)*n+i0, 1)])
        # 6
        add_term([((i0+1)*m+(j0+1), 1), (i0*m+j0, -1)], 
                 [(j0*p+k0, 1), ((j0+1)*p+k0, 1)], 
                 [((k0+1)*n+(i0+1), 1)])
        # 7
        add_term([(i0*m+(j0+1), 1), ((i0+1)*m+(j0+1), -1)], 
                 [((j0+1)*p+k0, 1), ((j0+1)*p+(k0+1), 1)], 
                 [(k0*n+i0, -1)])
                 
    # Schoolbook part 1: j=4
    # T_{i4k} = 1
    # U_{i*5+4} = 1, V_{4*6+k} = 1, W_{k*4+i} = 1
    u_vals = [(i*m+4, 1) for i in range(n)]
    v_vals = [(4*p+k, 1) for k in range(p)]
    w_vals = [(k*n+i, 1) for k in range(p) for i in range(n)]
    add_term(u_vals, v_vals, w_vals)
    
    # Schoolbook part 2: k in {4,5}, j in {0..3}
    # T_{ijk} = 1
    # U_{i*5+j} = 1, V_{j*6+k} = 1, W_{k*4+i} = 1
    u_vals = [(i*m+j, 1) for i in range(n) for j in range(4)]
    v_vals = [(j*p+k, 1) for j in range(4) for k in range(4, p)]
    w_vals = [(k*n+i, 1) for k in range(4, p) for i in range(n)]
    add_term(u_vals, v_vals, w_vals)
    
    return {"U": U, "V": V, "W": W}
