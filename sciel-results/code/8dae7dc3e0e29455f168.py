"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition: 4 independent 2x2x2 blocks, each rank 7
    # Total rank = 4 * 7 = 28
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 4 blocks
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]   # (4, 28)
    V = [[0] * rank for _ in range(m * p)]   # (16, 28)
    W = [[0] * rank for _ in range(p * n)]   # (16, 28)
    
    # Strassen's 7 products for 2x2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0][0] = M1 + M4 - M5 + M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 + M3 + M6
    
    # U coefficients: A elements (0=a, 1=b, 2=c, 3=d)
    strassen_U = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1], # M7: b-d
    ]
    
    # V coefficients: B elements (0=e, 1=f, 2=g, 3=h)
    strassen_V = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [0, 0, 1, -1], # M4: g-e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1],  # M7: g+h
    ]
    
    # W coefficients: C elements (0=C[0][0], 1=C[0][1], 2=C[1][0], 3=C[1][1])
    strassen_W = [
        [1, 0, 0, 1],   # M1: C[0][0]=+1, C[1][1]=+1
        [0, 0, 1, -1],  # M2: C[1][0]=+1, C[1][1]=-1
        [0, 1, 0, 1],   # M3: C[0][1]=+1, C[1][1]=+1
        [1, 0, 1, 0],   # M4: C[0][0]=+1, C[1][0]=+1
        [-1, 1, 0, 0],  # M5: C[0][0]=-1, C[0][1]=+1
        [0, 0, 0, 1],   # M6: C[1][1]=+1
        [1, 0, 0, -1],  # M7: C[0][0]=+1, C[1][1]=-1
    ]
    
    for block in range(num_blocks):
        base_rank = block * 7
        
        # B global indices: V[row*p + col]
        # Row 0: cols 2*block, 2*block+1 → V[0*8 + 2*block] = V[2*block], V[2*block+1]
        # Row 1: cols 2*block, 2*block+1 → V[1*8 + 2*block] = V[8+2*block], V[8+2*block+1]
        b_idx_map = [2*block, 2*block+1, 8+2*block, 8+2*block+1]
        
        # C global indices: W[k*n + i] where k=col, i=row
        # C[0][2*block]: k=2*block, i=0 → W[2*block*2 + 0] = W[4*block]
        # C[0][2*block+1]: k=2*block+1, i=0 → W[(2*block+1)*2 + 0] = W[4*block+2]
        # C[1][2*block]: k=2*block, i=1 → W[2*block*2 + 1] = W[4*block+1]
        # C[1][2*block+1]: k=2*block+1, i=1 → W[(2*block+1)*2 + 1] = W[4*block+3]
        c_idx_map = [4*block, 4*block+2, 4*block+1, 4*block+3]
        
        for s in range(7):
            r = base_rank + s
            # Fill U
            for local_a in range(4):
                if strassen_U[s][local_a] != 0:
                    U[local_a][r] = strassen_U[s][local_a]
            # Fill V
            for local_b in range(4):
                if strassen_V[s][local_b] != 0:
                    V[b_idx_map[local_b]][r] = strassen_V[s][local_b]
            # Fill W
            for local_c in range(4):
                if strassen_W[s][local_c] != 0:
                    W[c_idx_map[local_c]][r] = strassen_W[s][local_c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
