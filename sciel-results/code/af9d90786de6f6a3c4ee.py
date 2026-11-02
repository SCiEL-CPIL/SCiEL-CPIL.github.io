import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    # 2 blocks of m (each 2x2), 3 blocks of p (each 2x2)
    # Each 2x2x2 block uses Strassen rank 7
    
    U = np.zeros((n*m, 42))
    V = np.zeros((m*p, 42))
    W = np.zeros((p*n, 42))
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U rows: a, b, c, d (indices 0,1,2,3 in local 2x2)
    # V rows: e, f, g, h (indices 0,1,2,3 in local 2x2)
    # W rows: C11, C12, C21, C22 (indices 0,1,2,3 in local 2x2)
    
    def add_strassen_block(u_offset, v_offset, w_offset, col_offset):
        # u_offset: row in U for 'a'
        # v_offset: row in V for 'e'
        # w_offset: row in W for 'C11'
        
        # M1
        U[u_offset, col_offset] += 1   # a
        U[u_offset+2, col_offset] += 1 # d
        V[v_offset, col_offset] += 1   # e
        V[v_offset+2, col_offset] += 1 # h
        
        # M2
        U[u_offset+2, col_offset+1] += 1 # c
        U[u_offset+3, col_offset+1] += 1 # d
        V[v_offset, col_offset+1] += 1   # e
        
        # M3
        U[u_offset, col_offset+2] += 1   # a
        V[v_offset+1, col_offset+2] += 1 # f
        V[v_offset+3, col_offset+2] -= 1 # h
        
        # M4
        U[u_offset+3, col_offset+3] += 1 # d
        V[v_offset+2, col_offset+3] += 1 # g
        V[v_offset, col_offset+3] -= 1   # e
        
        # M5
        U[u_offset, col_offset+4] += 1   # a
        U[u_offset+1, col_offset+4] += 1 # b
        V[v_offset+3, col_offset+4] += 1 # h
        
        # M6
        U[u_offset+2, col_offset+5] += 1 # c
        U[u_offset, col_offset+5] -= 1   # a
        V[v_offset, col_offset+5] += 1   # e
        V[v_offset+1, col_offset+5] += 1 # f
        
        # M7
        U[u_offset+1, col_offset+6] += 1 # b
        U[u_offset+3, col_offset+6] -= 1 # d
        V[v_offset+2, col_offset+6] += 1 # g
        V[v_offset+3, col_offset+6] += 1 # h
        
        # W contributions
        # C11 = M1 + M4 - M5 + M7
        W[w_offset, col_offset] += 1
        W[w_offset, col_offset+3] += 1
        W[w_offset, col_offset+4] -= 1
        W[w_offset, col_offset+6] += 1
        
        # C12 = M3 + M5
        W[w_offset+1, col_offset+2] += 1
        W[w_offset+1, col_offset+4] += 1
        
        # C21 = M2 + M4
        W[w_offset+2, col_offset+1] += 1
        W[w_offset+2, col_offset+3] += 1
        
        # C22 = M1 - M2 + M3 + M6
        W[w_offset+3, col_offset] += 1
        W[w_offset+3, col_offset+1] -= 1
        W[w_offset+3, col_offset+2] += 1
        W[w_offset+3, col_offset+5] += 1

    col = 0
    # m=4 split into two blocks of 2: J1 (rows 0,1 of B), J2 (rows 2,3 of B)
    # n=2, so A is 2x4. A_J1 is cols 0,1. A_J2 is cols 2,3.
    for j_block in range(2):
        # u_offset for A_J: rows 0,1,2,3 of U?
        # U index: i*m + j.
        # For J1 (j=0,1): U rows 0,1 (i=0) and 2,3 (i=1)?
        # No, U shape is (n*m, rank). Row index is i*m + j.
        # i=0, j=0 -> 0. i=0, j=1 -> 1.
        # i=1, j=0 -> 2. i=1, j=1 -> 3.
        # So for J1, u_offset = j_block * 2?
        # j_block 0: j=0,1. U rows: 0,1 (i=0) and 2,3 (i=1).
        # Local a,b,c,d map to U rows:
        # a: i=0, j=0 -> 0
        # b: i=0, j=1 -> 1
        # c: i=1, j=0 -> 2
        # d: i=1, j=1 -> 3
        # So u_offset = 0 for J1, u_offset = 4 for J2?
        # J2: j=2,3.
        # a: i=0, j=2 -> 2
        # b: i=0, j=3 -> 3
        # c: i=1, j=2 -> 6
        # d: i=1, j=3 -> 7
        # This is not contiguous 0,1,2,3.
        # My add_strassen_block assumes contiguous u_offset, u_offset+1, etc.
        # I need to pass explicit row indices or adjust the function.
        
        # Let's define the mapping explicitly.
        # A_J rows in U:
        # i=0, j=2*jb + 0 -> 2*jb
        # i=0, j=2*jb + 1 -> 2*jb + 1
        # i=1, j=2*jb + 0 -> 2 + 2*jb
        # i=1, j=2*jb + 1 -> 3 + 2*jb
        
        # B_J rows in V:
        # j=2*jb + 0, k=2*kb + 0 -> (2*jb)*p + (2*kb) ? No.
        # V index: j*p + k.
        # j=0, k=0 -> 0. j=0, k=1 -> 1.
        # j=1, k=0 -> 6. j=1, k=1 -> 7.
        # For J1 (j=0,1):
        # e: j=0, k=2*kb -> 6*kb
        # f: j=0, k=2*kb+1 -> 6*kb+1
        # g: j=1, k=2*kb -> 6*kb+6
        # h: j=1, k=2*kb+1 -> 6*kb+7
        
        # C_J cols in W:
        # W index: k*n + i.
        # k=2*kb + 0, i=0 -> 2*kb*2 + 0 = 4*kb
        # k=2*kb + 0, i=1 -> 4*kb + 1
        # k=2*kb + 1, i=0 -> 4*kb + 2
        # k=2*kb + 1, i=1 -> 4*kb + 3
        
        for kb in range(3):
            # u_rows for a,b,c,d
            u_a = 2*jb # i=0, j=2jb
            u_b = 2*jb + 1 # i=0, j=2jb+1
            u_c = 2 + 2*jb # i=1, j=2jb
            u_d = 3 + 2*jb # i=1, j=2jb+1
            
            # v_rows for e,f,g,h
            v_e = (2*jb)*6 + 2*kb # j=2jb, k=2kb
            v_f = (2*jb)*6 + 2*kb + 1 # j=2jb, k=2kb+1
            v_g = (2*jb+1)*6 + 2*kb # j=2jb+1, k=2kb
            v_h = (2*jb+1)*6 + 2*kb + 1 # j=2jb+1, k=2kb+1
            
            # w_rows for C11,C12,C21,C22
            w_11 = (2*kb)*2 + 0 # k=2kb, i=0
            w_12 = (2*kb)*2 + 2 # k=2kb+1, i=0
            w_21 = (2*kb)*2 + 1 # k=2kb, i=1
            w_22 = (2*kb)*2 + 3 # k=2kb+1, i=1
            
            # Call Strassen logic with these specific indices
            # M1
            U[u_a, col] += 1
            U[u_d, col] += 1
            V[v_e, col] += 1
            V[v_h, col] += 1
            W[w_11, col] += 1
            W[w_22, col] += 1
            
            # M2
            U[u_c, col+1] += 1
            U[u_d, col+1] += 1
            V[v_e, col+1] += 1
            W[w_21, col+1] += 1
            W[w_22, col+1] -= 1
            
            # M3
            U[u_a, col+2] += 1
            V[v_f, col+2] += 1
            V[v_h, col+2] -= 1
            W[w_12, col+2] += 1
            W[w_22, col+2] += 1
            
            # M4
            U[u_d, col+3] += 1
            V[v_g, col+3] += 1
            V[v_e, col+3] -= 1
            W[w_11, col+3] += 1
            W[w_21, col+3] += 1
            
            # M5
            U[u_a, col+4] += 1
            U[u_b, col+4] += 1
            V[v_h, col+4] += 1
            W[w_11, col+4] -= 1
            W[w_12, col+4] += 1
            
            # M6
            U[u_c, col+5] += 1
            U[u_a, col+5] -= 1
            V[v_e, col+5] += 1
            V[v_f, col+5] += 1
            W[w_22, col+5] += 1
            
            # M7
            U[u_b, col+6] += 1
            U[u_d, col+6] -= 1
            V[v_g, col+6] += 1
            V[v_h, col+6] += 1
            W[w_11, col+6] += 1
            
            col += 7

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
