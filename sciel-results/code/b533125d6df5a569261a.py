import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x2 matrix by a 2x12 matrix.
    
    The decomposition uses Strassen's algorithm for six independent 2x2 blocks.
    Rank = 6 * 7 = 42.
    """
    n, m, p = 2, 2, 12
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2:
    # M1 = (a+d)(e+h) -> U: a,d ; V: e,h ; W: c00, c11
    # M2 = (c+d)(e)   -> U: c,d ; V: e    ; W: c10
    # M3 = (a)(f-h)   -> U: a    ; V: f,h  ; W: c01
    # M4 = (d)(g-e)   -> U: d    ; V: g,e  ; W: c00, c10
    # M5 = (a+b)(h)   -> U: a,b  ; V: h    ; W: c01, c00
    # M6 = (c-a)(e+f) -> U: c,a  ; V: e,f  ; W: c11
    # M7 = (b-d)(g+h) -> U: b,d  ; V: g,h  ; W: c11, c00
    
    # Indices for A (2x2): 0:a, 1:b, 2:c, 3:d
    # Indices for B (2x2 local): 0:e, 1:f, 2:g, 3:h
    # Indices for C (2x2 local): 0:c00, 1:c01, 2:c10, 3:c11
    
    strassen_U = [
        [1, 0, 0, 1], # M1: a, d
        [0, 0, 1, 1], # M2: c, d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a, b
        [-1, 0, 1, 0],# M6: c, a (c-a)
        [0, 1, 0, -1] # M7: b, d (b-d)
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: e, h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f, h (f-h)
        [-1, 0, 1, 0],# M4: g, e (g-e)
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e, f
        [0, 0, 1, 1]  # M7: g, h
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M1: c00, c11
        [0, 0, 1, 0], # M2: c10
        [0, 1, 0, 0], # M3: c01
        [1, 0, 1, 0], # M4: c00, c10
        [-1, 1, 0, 0],# M5: c01, c00
        [0, 0, 0, 1], # M6: c11
        [1, 0, 0, 0]  # M7: c00
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            # U
            for a_idx in range(4):
                U[a_idx][col] = strassen_U[s][a_idx]
            # V
            for local_b in range(4):
                val = strassen_V[s][local_b]
                if val == 0: continue
                # Map local B index (0:e, 1:f, 2:g, 3:h) to global B index
                # B is 2x12. Block t covers cols 2t, 2t+1.
                # Row 0: e,f -> 2t, 2t+1
                # Row 1: g,h -> 12 + 2t, 12 + 2t + 1
                if local_b == 0: gidx = 2 * t
                elif local_b == 1: gidx = 2 * t + 1
                elif local_b == 2: gidx = 12 + 2 * t
                else: gidx = 12 + 2 * t + 1
                V[gidx][col] = val
            # W
            for local_c in range(4):
                val = strassen_W[s][local_c]
                if val == 0: continue
                # Map local C index (0:c00, 1:c01, 2:c10, 3:c11) to global W index
                # W trace convention: W[k*n + i] where k=colC, i=rowC
                # c00: col=2t, row=0 -> 2t*2 + 0
                # c01: col=2t+1, row=0 -> (2t+1)*2 + 0
                # c10: col=2t, row=1 -> 2t*2 + 1
                # c11: col=2t+1, row=1 -> (2t+1)*2 + 1
                if local_c == 0: gidx = (2 * t) * 2 + 0
                elif local_c == 1: gidx = (2 * t + 1) * 2 + 0
                elif local_c == 2: gidx = (2 * t) * 2 + 1
                else: gidx = (2 * t + 1) * 2 + 1
                W[gidx][col] = val
                
    return {"U": U, "V": V, "W": W}
