import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    nm, mp, pn = n * m, m * p, p * n
    rank = 28  # 4 blocks of 2x2x2, each rank 7
    
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    # Strassen decomposition vectors for 2x2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B local indices: 0:e, 1:f, 2:g, 3:h
    # C local indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    # Note: Task C index is k*n + i.
    # Local k=0,1.
    # k=0: c=0*i? No.
    # In 2x2 case, C is 2x2.
    # C_{0,0} -> k=0, i=0 -> idx 0
    # C_{1,0} -> k=0, i=1 -> idx 1
    # C_{0,1} -> k=1, i=0 -> idx 2
    # C_{1,1} -> k=1, i=1 -> idx 3
    
    # M1: (a+d)(e+f)
    U_1 = [1, 0, 0, 1]
    V_1 = [1, 1, 0, 0]
    W_1 = [1, 0, -1, 1] # C00=1, C10=0, C01=-1, C11=1
    
    # M2: (c+d)e
    U_2 = [0, 0, 1, 1]
    V_2 = [1, 0, 0, 0]
    W_2 = [0, 1, 0, -1] # C00=0, C10=1, C01=0, C11=-1
    
    # M3: a(f-h)
    U_3 = [1, 0, 0, 0]
    V_3 = [0, 1, 0, -1]
    W_3 = [0, 0, 1, 0] # C00=0, C10=0, C01=1, C11=0
    
    # M4: d(g-e)
    U_4 = [0, 0, 0, 1]
    V_4 = [0, 0, 1, -1]
    W_4 = [1, 1, 0, 0] # C00=1, C10=1, C01=0, C11=0
    
    # M5: (a+b)h
    U_5 = [1, 1, 0, 0]
    V_5 = [0, 0, 0, 1]
    W_5 = [-1, 0, 1, 0] # C00=-1, C10=0, C01=1, C11=0
    
    # M6: (c+d)(f-h)
    U_6 = [0, 0, 1, 1]
    V_6 = [0, 1, 0, -1]
    W_6 = [0, -1, 0, 1] # C00=0, C10=-1, C01=0, C11=1
    
    # M7: (a+b)(e-g)
    U_7 = [1, 1, 0, 0]
    V_7 = [1, 0, -1, 0]
    W_7 = [1, -1, 1, 0] # C00=1, C10=-1, C01=1, C11=0
    
    terms = [
        (U_1, V_1, W_1),
        (U_2, V_2, W_2),
        (U_3, V_3, W_3),
        (U_4, V_4, W_4),
        (U_5, V_5, W_5),
        (U_6, V_6, W_6),
        (U_7, V_7, W_7)
    ]
    
    # 4 blocks of columns in B
    num_blocks = 4
    for b in range(num_blocks):
        # Global indices for B in this block
        # e = B[0, 2b] -> V index 2b
        # f = B[0, 2b+1] -> V index 2b+1
        # g = B[1, 2b] -> V index 8 + 2b
        # h = B[1, 2b+1] -> V index 8 + 2b + 1
        v_map = {
            0: 2 * b,       # e
            1: 2 * b + 1,   # f
            2: 8 + 2 * b,   # g
            3: 8 + 2 * b + 1 # h
        }
        
        # Global indices for C in this block
        # C[0, 2b] -> W index (2b)*2 + 0 = 4b
        # C[1, 2b] -> W index (2b)*2 + 1 = 4b+1
        # C[0, 2b+1] -> W index (2b+1)*2 + 0 = 4b+2
        # C[1, 2b+1] -> W index (2b+1)*2 + 1 = 4b+3
        w_map = {
            0: 4 * b,       # C00
            1: 4 * b + 1,   # C10
            2: 4 * b + 2,   # C01
            3: 4 * b + 3    # C11
        }
        
        for l in range(7):
            r = b * 7 + l
            u_l, v_l, w_l = terms[l]
            
            for i in range(4):
                U[i][r] = u_l[i]
                
            for j in range(4):
                if v_l[j] != 0:
                    V[v_map[j]][r] = v_l[j]
                    
            for k in range(4):
                if w_l[k] != 0:
                    W[w_map[k]][r] = w_l[k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
