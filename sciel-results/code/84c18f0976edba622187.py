"""Combined program: block decomposition 2+3 with Strassen for the 2x2 block."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition 2+3 with Strassen for the 2x2 block.
    # This reduces rank from 125 (schoolbook) to 124.
    n, m, p = DIMENSIONS
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: dict {(i,j): val}
        v_vals: dict {(j,k): val}
        w_vals: dict {(k,i): val}
        """
        U = [0] * (n * m)
        V = [0] * (m * p)
        W = [0] * (p * n)
        
        for (i, j), val in u_vals.items():
            U[i * m + j] = val
        for (j, k), val in v_vals.items():
            V[j * p + k] = val
        for (k, i), val in w_vals.items():
            W[k * n + i] = val
            
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    # 1. C_22 = A_22 B_22 + A_23 B_32
    # A_22: i in 0,1; j in 0,1
    # B_22: j in 0,1; k in 0,1
    # C_22: i in 0,1; k in 0,1
    # Use Strassen for A_22 B_22
    
    # Strassen terms for 2x2
    # Indices are local 0,1 -> global 0,1
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1
    u = {(0,0):1, (1,1):1}
    v = {(0,0):1, (1,1):1}
    # M1 contributes +1 to C00 (k=0, i=0) and +1 to C11 (k=1, i=1)
    add_term(u, v, {(0,0):1, (1,1):1})
    
    # M2
    u = {(1,0):1, (1,1):1}
    v = {(0,0):1}
    # M2 contributes +1 to C10 (k=0, i=1) and -1 to C11 (k=1, i=1)
    add_term(u, v, {(0,1):1, (1,1):-1})
    
    # M3
    u = {(0,0):1}
    v = {(0,1):1, (1,1):-1}
    # M3 contributes +1 to C01 (k=1, i=0) and +1 to C11 (k=1, i=1)
    add_term(u, v, {(1,0):1, (1,1):1})
    
    # M4
    u = {(1,1):1}
    v = {(1,0):1, (0,0):-1}
    # M4 contributes +1 to C00 (k=0, i=0) and +1 to C10 (k=0, i=1)
    add_term(u, v, {(0,0):1, (0,1):1})
    
    # M5
    u = {(0,0):1, (0,1):1}
    v = {(1,1):1}
    # M5 contributes -1 to C00 (k=0, i=0) and +1 to C01 (k=1, i=0)
    add_term(u, v, {(0,0):-1, (1,0):1})
    
    # M6
    u = {(1,0):1, (0,0):-1}
    v = {(0,0):1, (0,1):1}
    # M6 contributes +1 to C11 (k=1, i=1)
    add_term(u, v, {(1,1):1})
    
    # M7
    u = {(0,1):1, (1,1):-1}
    v = {(1,0):1, (1,1):1}
    # M7 contributes +1 to C00 (k=0, i=0)
    add_term(u, v, {(0,0):1})
    
    # A_23 B_32: Schoolbook
    # A_23: i in 0,1; j in 2,3,4
    # B_32: j in 2,3,4; k in 0,1
    # C_22: i in 0,1; k in 0,1
    for i in [0, 1]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    # 2. C_23 = A_22 B_23 + A_23 B_33
    # A_22: i in 0,1; j in 0,1
    # B_23: j in 0,1; k in 2,3,4
    # C_23: i in 0,1; k in 2,3,4
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    # A_23: i in 0,1; j in 2,3,4
    # B_33: j in 2,3,4; k in 2,3,4
    # C_23: i in 0,1; k in 2,3,4
    for i in [0, 1]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    # 3. C_32 = A_32 B_22 + A_33 B_32
    # A_32: i in 2,3,4; j in 0,1
    # B_22: j in 0,1; k in 0,1
    # C_32: i in 2,3,4; k in 0,1
    for i in [2, 3, 4]:
        for j in [0, 1]:
            for k in [0, 1]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    # A_33: i in 2,3,4; j in 2,3,4
    # B_32: j in 2,3,4; k in 0,1
    # C_32: i in 2,3,4; k in 0,1
    for i in [2, 3, 4]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    # 4. C_33 = A_32 B_23 + A_33 B_33
    # A_32: i in 2,3,4; j in 0,1
    # B_23: j in 0,1; k in 2,3,4
    # C_33: i in 2,3,4; k in 2,3,4
    for i in [2, 3, 4]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    # A_33: i in 2,3,4; j in 2,3,4
    # B_33: j in 2,3,4; k in 2,3,4
    # C_33: i in 2,3,4; k in 2,3,4
    for i in [2, 3, 4]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    rank = len(U_list)
    U = [row[:] for row in zip(*U_list)]
    V = [row[:] for row in zip(*V_list)]
    W = [row[:] for row in zip(*W_list)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
