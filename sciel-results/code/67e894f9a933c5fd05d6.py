import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 6 * 7 = 42
    rank = 42
    
    # Initialize matrices
    # U: (n*m, rank) = (4, 42)
    U = [[0] * rank for _ in range(n * m)]
    # V: (m*p, rank) = (24, 42)
    V = [[0] * rank for _ in range(m * p)]
    # W: (p*n, rank) = (24, 42)
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2
    # u_s: [a, b, c, d]
    # v_s_local: [e, f, g, h]
    # w_s_local: [C00, C01, C10, C11]
    
    # Note: The local indices for B are mapped to [e, f, g, h]
    # e=B00, f=B01, g=B10, h=B11
    # Local B indices in V: 
    # e -> 0, f -> 1, g -> 2, h -> 3 (relative to block start)
    
    # Note: The local indices for C are mapped to [C00, C01, C10, C11]
    # C00, C01, C10, C11
    
    # Define the 7 steps
    # s=0: P1
    u0 = [1, 0, 0, 1]
    v0 = [1, 1, 0, 0]
    w0 = [1, 0, 0, 1]
    
    # s=1: P2
    u1 = [0, 0, 1, 1]
    v1 = [1, 0, 0, 0]
    w1 = [0, 0, 1, -1]
    
    # s=2: P3
    u2 = [1, 0, 0, 0]
    v2 = [0, 1, 0, -1]
    w2 = [0, 1, 0, 1]
    
    # s=3: P4
    u3 = [0, 0, 0, 1]
    v3 = [-1, 0, 1, 0]
    w3 = [1, 0, 1, 0]
    
    # s=4: P5
    u4 = [1, 1, 0, 0]
    v4 = [0, 0, 0, 1]
    w4 = [-1, 1, 0, 0]
    
    # s=5: P6
    u5 = [-1, 0, 1, 0]
    v5 = [1, 1, 0, 0]
    w5 = [0, 0, 0, 1]
    
    # s=6: P7
    u6 = [0, 1, 0, -1]
    v6 = [0, 0, 1, 1]
    w6 = [1, 0, 0, 0]
    
    steps = [
        (u0, v0, w0),
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6)
    ]
    
    # Map local indices to global indices
    # B global index for block b, local idx l (0=e, 1=f, 2=g, 3=h)
    # e: j=0, k=2b -> 0*12 + 2b
    # f: j=0, k=2b+1 -> 0*12 + 2b+1
    # g: j=1, k=2b -> 1*12 + 2b
    # h: j=1, k=2b+1 -> 1*12 + 2b+1
    
    # C global index for block b, local idx l (0=C00, 1=C01, 2=C10, 3=C11)
    # C00: k=2b, i=0 -> 2b*2 + 0
    # C01: k=2b+1, i=0 -> (2b+1)*2 + 0
    # C10: k=2b, i=1 -> 2b*2 + 1
    # C11: k=2b+1, i=1 -> (2b+1)*2 + 1
    
    num_blocks = p // 2 # 6
    
    for b in range(num_blocks):
        for s in range(7):
            r = 7 * b + s
            u_s, v_s, w_s = steps[s]
            
            # Fill U
            for i in range(4):
                U[i][r] = u_s[i]
                
            # Fill V
            # Map local v_s indices to global
            # v_s[0] -> e -> 2b
            # v_s[1] -> f -> 2b+1
            # v_s[2] -> g -> 12+2b
            # v_s[3] -> h -> 12+2b+1
            V[2*b][r] = v_s[0]
            V[2*b+1][r] = v_s[1]
            V[12+2*b][r] = v_s[2]
            V[12+2*b+1][r] = v_s[3]
            
            # Fill W
            # Map local w_s indices to global
            # w_s[0] -> C00 -> 4b
            # w_s[1] -> C01 -> 4b+2
            # w_s[2] -> C10 -> 4b+1
            # w_s[3] -> C11 -> 4b+3
            W[4*b][r] = w_s[0]
            W[4*b+2][r] = w_s[1]
            W[4*b+1][r] = w_s[2]
            W[4*b+3][r] = w_s[3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
