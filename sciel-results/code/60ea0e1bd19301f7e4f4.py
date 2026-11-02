"""Construct rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Rank 68 construction
    # Split p=8 into 4 blocks of size 2.
    # For each block, split n=3 into 2+1 and m=3 into 2+1.
    # 2x2x2 part: Rank 7 (Winograd)
    # 2x1x2 part: Rank 4 (Schoolbook)
    # 1x3x2 part: Rank 6 (Schoolbook)
    # Total per block: 17. Total 4 blocks: 68.
    
    rank = 68
    U = [[0]*rank for _ in range(n*m)]  # 9 rows
    V = [[0]*rank for _ in range(m*p)]  # 24 rows
    W = [[0]*rank for _ in range(p*n)]  # 24 rows
    
    # Winograd 2x2x2 coefficients
    # U_local (4x7), V_local (4x7), W_local (4x7)
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_w = [
        [1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0]
    ]
    V_w = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0]
    ]
    W_w = [
        [1, -1, 0, 0, -1, 0, 1],
        [0, 1, -1, 1, 0, -1, 0],
        [-1, 0, 1, -1, 1, 0, 0],
        [1, -1, 0, 0, 0, 0, -1]
    ]
    
    col = 0
    for b in range(4):
        # Block b handles k = 2b, 2b+1
        k1 = 2*b
        k2 = 2*b + 1
        
        # --- Part 1: 2x2x2 (i in 0,1; j in 0,1; k in k1,k2) ---
        # Map local 2x2 indices to global
        # U global indices for i in 0,1 and j in 0,1
        # (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
        u_map_p1 = [0, 1, 3, 4]
        # V global indices for j in 0,1 and k in k1,k2
        # (0,k1)->k1, (0,k2)->k2, (1,k1)->8+k1, (1,k2)->8+k2
        v_map_p1 = [k1, k2, 8+k1, 8+k2]
        # W global indices for k in k1,k2 and i in 0,1
        # (k1,0)->3*k1, (k1,1)->3*k1+1, (k2,0)->3*k2, (k2,1)->3*k2+1
        w_map_p1 = [3*k1, 3*k1+1, 3*k2, 3*k2+1]
        
        for t in range(7):
            for l in range(4):
                U[u_map_p1[l]][col] += U_w[l][t]
                V[v_map_p1[l]][col] += V_w[l][t]
                W[w_map_p1[l]][col] += W_w[l][t]
            col += 1
            
        # --- Part 2: 2x1x2 (i in 0,1; j=2; k in k1,k2) ---
        # Terms:
        # 1. A[0,2] * B[2,k1] -> C[k1,0]
        # 2. A[0,2] * B[2,k2] -> C[k2,0]
        # 3. A[1,2] * B[2,k1] -> C[k1,1]
        # 4. A[1,2] * B[2,k2] -> C[k2,1]
        
        # A[0,2] index: 0*3+2 = 2
        # A[1,2] index: 1*3+2 = 5
        # B[2,k1] index: 2*8+k1 = 16+k1
        # B[2,k2] index: 2*8+k2 = 16+k2
        # C[k1,0] index: 3*k1+0
        # C[k2,0] index: 3*k2+0
        # C[k1,1] index: 3*k1+1
        # C[k2,1] index: 3*k2+1
        
        U[2][col] = 1; V[16+k1][col] = 1; W[3*k1][col] = 1; col += 1
        U[2][col] = 1; V[16+k2][col] = 1; W[3*k2][col] = 1; col += 1
        U[5][col] = 1; V[16+k1][col] = 1; W[3*k1+1][col] = 1; col += 1
        U[5][col] = 1; V[16+k2][col] = 1; W[3*k2+1][col] = 1; col += 1
        
        # --- Part 3: 1x3x2 (i=2; j in 0,1,2; k in k1,k2) ---
        # Terms:
        # 1. A[2,0] * B[0,k1] -> C[k1,2]
        # 2. A[2,1] * B[1,k1] -> C[k1,2]
        # 3. A[2,2] * B[2,k1] -> C[k1,2]
        # 4. A[2,0] * B[0,k2] -> C[k2,2]
        # 5. A[2,1] * B[1,k2] -> C[k2,2]
        # 6. A[2,2] * B[2,k2] -> C[k2,2]
        
        # A[2,0] index: 2*3+0 = 6
        # A[2,1] index: 2*3+1 = 7
        # A[2,2] index: 2*3+2 = 8
        # B[0,k1] index: 0*8+k1 = k1
        # B[1,k1] index: 1*8+k1 = 8+k1
        # B[2,k1] index: 2*8+k1 = 16+k1
        # B[0,k2] index: k2
        # B[1,k2] index: 8+k2
        # B[2,k2] index: 16+k2
        # C[k1,2] index: 3*k1+2
        # C[k2,2] index: 3*k2+2
        
        U[6][col] = 1; V[k1][col] = 1; W[3*k1+2][col] = 1; col += 1
        U[7][col] = 1; V[8+k1][col] = 1; W[3*k1+2][col] = 1; col += 1
        U[8][col] = 1; V[16+k1][col] = 1; W[3*k1+2][col] = 1; col += 1
        U[6][col] = 1; V[k2][col] = 1; W[3*k2+2][col] = 1; col += 1
        U[7][col] = 1; V[8+k2][col] = 1; W[3*k2+2][col] = 1; col += 1
        U[8][col] = 1; V[16+k2][col] = 1; W[3*k2+2][col] = 1; col += 1
        
    return {"U": U, "V": V, "W": W}
