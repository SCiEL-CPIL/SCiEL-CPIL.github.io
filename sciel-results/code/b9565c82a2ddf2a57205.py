"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Schoolbook for i=0 (1x4x8)
    # i=0, j in 0..3, k in 0..7
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: Strassen for i in 1,2 (2x4x8)
    # 8 blocks of 2x2x2
    # Strassen coefficients
    # M1..M7
    # U_local (2x2, 7)
    U_loc = [
        [0, 0, 1, 0, 1, -1, 1], # i=0, j=0
        [1, 1, 0, 0, 0, 1, 0],  # i=0, j=1
        [1, 1, 0, 1, 1, 0, -1]  # i=1, j=0
        # Wait, U is indexed by i, j.
        # M1: (x10+x11)y00 -> U[1,0]+=1, U[1,1]+=1
        # M2: (x10+x11)(y10+y11) -> U[1,0]+=1, U[1,1]+=1
        # M3: x00(y10-y00) -> U[0,0]+=1
        # M4: x11(y01-y11) -> U[1,1]+=1
        # M5: (x00+x10)y11 -> U[0,0]+=1, U[1,0]+=1
        # M6: (x01-x00)(y00+y10) -> U[0,1]+=1, U[0,0]-=1
        # M7: (x01-x11)(y11+y01) -> U[0,1]+=1, U[1,1]-=1
        
        # Let's re-verify U_loc[i, j, r]
        # r=0 (M1): U[1,0]=1, U[1,1]=1
        # r=1 (M2): U[1,0]=1, U[1,1]=1
        # r=2 (M3): U[0,0]=1
        # r=3 (M4): U[1,1]=1
        # r=4 (M5): U[0,0]=1, U[1,0]=1
        # r=5 (M6): U[0,1]=1, U[0,0]=-1
        # r=6 (M7): U[0,1]=1, U[1,1]=-1
        
        # So:
        # i=0, j=0: r2:1, r4:1, r5:-1 => [0,0,1,0,1,-1,0]
        # i=0, j=1: r5:1, r6:1 => [0,0,0,0,0,1,1]
        # i=1, j=0: r0:1, r1:1, r4:1 => [1,1,0,0,1,0,0]
        # i=1, j=1: r0:1, r1:1, r3:1, r6:-1 => [1,1,0,1,0,0,-1]
    ]
    U_loc = [
        [0, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    # V_local (2x2, 7) indexed by j, k
    # M1: y00 -> V[0,0]=1
    # M2: y10+y11 -> V[1,0]=1, V[1,1]=1
    # M3: y10-y00 -> V[1,0]=1, V[0,0]=-1
    # M4: y01-y11 -> V[0,1]=1, V[1,1]=-1
    # M5: y11 -> V[1,1]=1
    # M6: y00+y10 -> V[0,0]=1, V[1,0]=1
    # M7: y11+y01 -> V[1,1]=1, V[0,1]=1
    
    # j=0, k=0: r0:1, r2:-1, r5:1 => [1,-1,1,0,0,0,0]
    # j=0, k=1: r3:1, r6:1 => [0,0,0,1,0,0,1]
    # j=1, k=0: r1:1, r2:1, r5:1 => [0,1,1,0,0,1,0]
    # j=1, k=1: r1:1, r3:-1, r4:1, r6:1 => [0,1,0,-1,1,0,1]
    
    V_loc = [
        [1, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, -1, 1, 0, 1]
    ]
    
    # W_local (2x2, 7) indexed by k, i
    # Z00 = M1 + M4 - M5 + M7 -> W[0,0]
    # Z01 = M3 + M5 -> W[1,0]
    # Z10 = M1 + M2 -> W[0,1]
    # Z11 = M2 - M4 + M6 + M7 -> W[1,1]
    
    # k=0, i=0: r0:1, r3:1, r4:-1, r6:1 => [1,0,0,1,-1,0,1]
    # k=1, i=0: r2:1, r4:1 => [0,0,1,0,1,0,0]
    # k=0, i=1: r0:1, r1:1 => [1,1,0,0,0,0,0]
    # k=1, i=1: r1:1, r3:-1, r5:1, r6:1 => [0,1,0,-1,0,1,1]
    
    W_loc = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, -1, 0, 1, 1]
    ]
    
    # Define the 8 blocks
    # block_id: (I_base, J_base, K_base)
    # I_base is 1 for all (since i in 1,2)
    # J_base: 0 for A1 (j in 0,1), 2 for A2 (j in 2,3)
    # K_base: 0, 2, 4, 6
    
    blocks = []
    for v in range(4):
        blocks.append((1, 0, 2*v)) # A1, B_top, col v
        blocks.append((1, 2, 2*v)) # A2, B_bot, col v
        
    for I_base, J_base, K_base in blocks:
        for r in range(7):
            # U
            for i_loc in range(2):
                for j_loc in range(2):
                    val = U_loc[i_loc * 2 + j_loc][r]
                    if val != 0:
                        I_glob = I_base + i_loc
                        J_glob = J_base + j_loc
                        U[I_glob * m + J_glob][col + r] += val
                        
            # V
            for j_loc in range(2):
                for k_loc in range(2):
                    val = V_loc[j_loc * 2 + k_loc][r]
                    if val != 0:
                        J_glob = J_base + j_loc
                        K_glob = K_base + k_loc
                        V[J_glob * p + K_glob][col + r] += val
                        
            # W
            for k_loc in range(2):
                for i_loc in range(2):
                    val = W_loc[k_loc * 2 + i_loc][r]
                    if val != 0:
                        K_glob = K_base + k_loc
                        I_glob = I_base + i_loc
                        W[K_glob * n + I_glob][col + r] += val
                        
        col += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
