"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen instances
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2x2 multiplication
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [1, 0, -1, 0], # P6
        [0, 0, -1, 1]  # P7
    ]
    V_strassen = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, -1, 1, 0], # P3
        [-1, 1, 0, 0], # P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 1, 1, 0]  # P7
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, -1], # Z00
        [0, 1, 1, 0, 0, 0, 0],   # Z01
        [0, 0, 0, 0, 0, 1, 1],   # Z10
        [1, 0, -1, 1, 0, 0, 0]   # Z11
    ]
    
    # Define instances: (u_map, v_map, w_map, col_offset)
    # u_map: local_idx -> global_idx for U
    # v_map: local_idx -> global_idx for V
    # w_map: local_idx -> global_idx for W
    
    # Instance 1: A1 * B1 -> C1
    # A1: rows 0,1 of A; cols 0,1 of A. Global U idx: 0,1,4,5
    # B1: rows 0,1 of B; cols 0,1 of B. Global V idx: 0,1,4,5
    # C1: rows 0,1 of C; cols 0,1 of C. Global W idx: 0,2,4,6
    inst1 = {
        'u': [0, 1, 4, 5],
        'v': [0, 1, 4, 5],
        'w': [0, 2, 4, 6],
        'offset': 0
    }
    
    # Instance 2: A2 * B3 -> C1
    # A2: rows 0,1 of A; cols 2,3 of A. Global U idx: 2,3,6,7
    # B3: rows 2,3 of B; cols 0,1 of B. Global V idx: 8,9,12,13
    # C1: same as above. Global W idx: 0,2,4,6
    inst2 = {
        'u': [2, 3, 6, 7],
        'v': [8, 9, 12, 13],
        'w': [0, 2, 4, 6],
        'offset': 7
    }
    
    # Instance 3: A1 * B2 -> C2
    # A1: same as above. Global U idx: 0,1,4,5
    # B2: rows 0,1 of B; cols 2,3 of B. Global V idx: 2,3,6,7
    # C2: rows 0,1 of C; cols 2,3 of C. Global W idx: 1,3,5,7
    inst3 = {
        'u': [0, 1, 4, 5],
        'v': [2, 3, 6, 7],
        'w': [1, 3, 5, 7],
        'offset': 14
    }
    
    # Instance 4: A2 * B4 -> C2
    # A2: same as above. Global U idx: 2,3,6,7
    # B4: rows 2,3 of B; cols 2,3 of B. Global V idx: 10,11,14,15
    # C2: same as above. Global W idx: 1,3,5,7
    inst4 = {
        'u': [2, 3, 6, 7],
        'v': [10, 11, 14, 15],
        'w': [1, 3, 5, 7],
        'offset': 21
    }
    
    instances = [inst1, inst2, inst3, inst4]
    
    for inst in instances:
        offset = inst['offset']
        u_map = inst['u']
        v_map = inst['v']
        w_map = inst['w']
        
        # Fill U
        for local_j in range(7):
            for local_i in range(4):
                val = U_strassen[local_j][local_i]
                if val != 0:
                    global_i = u_map[local_i]
                    global_j = offset + local_j
                    U[global_i][global_j] += val
                    
        # Fill V
        for local_j in range(7):
            for local_i in range(4):
                val = V_strassen[local_j][local_i]
                if val != 0:
                    global_i = v_map[local_i]
                    global_j = offset + local_j
                    V[global_i][global_j] += val
                    
        # Fill W
        for local_i in range(4):
            for local_j in range(7):
                val = W_strassen[local_i][local_j]
                if val != 0:
                    global_i = w_map[local_i]
                    global_j = offset + local_j
                    W[global_i][global_j] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
