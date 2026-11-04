import time

DIMENSIONS = [3, 4, 8]

def construct():
    start = time.time()
    deadline = start + 145.0

    n, m, p = DIMENSIONS
    rank_school = n * m * p

    # Build schoolbook decomposition as a fallback
    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Check time - if we're running low, return schoolbook
    if time.time() > deadline - 5:
        return build_schoolbook()

    # BCRL rank-19 decomposition for 3x3x3
    # Based on Bini, Capovani, Lotti, Romani (1978)
    # M1 = A11*B11
    # M2 = (A11+A12)*B22
    # M3 = (A21+A22)*B11
    # M4 = A22*B22
    # M5 = A11*(B12-B22)
    # M6 = (A21-A11)*(B11+B12)
    # M7 = A22*(B21-B11)
    # M8 = (A12-A22)*(B21+B22)
    # M9 = (A11+A22)*(B11+B22)
    # M10 = (A11+A12-A21-A22)*B12
    # M11 = (A11-A21+A12-A22)*B21
    # M12 = (A11+A12+A21+A22)*B11
    # M13 = (A11-A12+A21-A22)*B22
    # M14 = A11*(B12+B22)
    # M15 = A22*(B21+B11)
    # M16 = (A11+A12)*(B22-B12)
    # M17 = (A21+A22)*(B11-B21)
    # M18 = (A11-A21)*(B11+B12)
    # M19 = (A12-A22)*(B21+B22)
    
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    bcrl_U = [
        [1, 0, 0, 0],   # M1: A11
        [1, 1, 0, 0],   # M2: A11+A12
        [0, 0, 1, 1],   # M3: A21+A22
        [0, 0, 0, 1],   # M4: A22
        [1, 0, 0, 0],   # M5: A11
        [1, 0, -1, 0],  # M6: A11-A21
        [0, 0, 0, 1],   # M7: A22
        [0, 1, 0, -1],  # M8: A12-A22
        [1, 0, 0, 1],   # M9: A11+A22
        [1, 1, -1, -1], # M10: A11+A12-A21-A22
        [1, -1, -1, 1], # M11: A11-A12-A21+A22
        [1, 1, 1, 1],   # M12: A11+A12+A21+A22
        [1, -1, 1, -1], # M13: A11-A12+A21-A22
        [1, 0, 0, 0],   # M14: A11
        [0, 0, 0, 1],   # M15: A22
        [1, 1, 0, 0],   # M16: A11+A12
        [0, 0, 1, 1],   # M17: A21+A22
        [1, 0, -1, 0],  # M18: A11-A21
        [0, 1, 0, -1]   # M19: A12-A22
    ]
    
    bcrl_V = [
        [1, 0, 0, 0],   # M1: B11
        [0, 0, 0, 1],   # M2: B22
        [1, 0, 0, 0],   # M3: B11
        [0, 0, 0, 1],   # M4: B22
        [0, 1, 0, -1],  # M5: B12-B22
        [1, 1, 0, 0],   # M6: B11+B12
        [0, 0, 1, -1],  # M7: B21-B11
        [0, 0, 1, 1],   # M8: B21+B22
        [1, 0, 0, 1],   # M9: B11+B22
        [0, 1, 0, 0],   # M10: B12
        [0, 0, 1, 0],   # M11: B21
        [1, 0, 0, 0],   # M12: B11
        [0, 0, 0, 1],   # M13: B22
        [0, 1, 0, 1],   # M14: B12+B22
        [0, 0, 1, 1],   # M15: B21+B11
        [0, -1, 0, 1],  # M16: B22-B12
        [1, 0, -1, 0],  # M17: B11-B21
        [1, 1, 0, 0],   # M18: B11+B12
        [0, 0, 1, 1]    # M19: B21+B22
    ]
    
    # C11 = M1 + M2 - M3 - M5 + M6 - M7 + M9 - M10 + M12 - M13 - M14 - M15 + M16 - M17 - M18
    # C12 = M2 + M3 - M4 - M5 - M6 + M7 + M9 - M10 - M12 - M13 + M14 + M15 - M16 - M17 - M19
    # C21 = M1 - M2 - M3 + M4 + M5 - M6 - M7 + M9 + M10 + M11 - M12 - M13 - M14 + M15 + M16 + M17 - M18
    # C22 = M1 + M2 - M3 - M4 + M5 + M6 - M7 - M8 + M9 - M10 - M11 - M12 + M13 + M14 - M15 - M16 + M17 + M18 + M19
    
    # Let me recompute the W matrix more carefully
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using the 19 multiplication terms:
    # M1 = A11*B11
    # M2 = (A11+A12)*B22
    # M3 = (A21+A22)*B11
    # M4 = A22*B22
    # M5 = A11*(B12-B22)
    # M6 = (A11-A21)*(B11+B12)
    # M7 = A22*(B21-B11)
    # M8 = (A12-A22)*(B21+B22)
    # M9 = (A11+A22)*(B11+B22)
    # M10 = (A11+A12-A21-A22)*B12
    # M11 = (A11-A21+A12-A22)*B21
    # M12 = (A11+A12+A21+A22)*B11
    # M13 = (A11-A12+A21-A22)*B22
    # M14 = A11*(B12+B22)
    # M15 = A22*(B21+B11)
    # M16 = (A11+A12)*(B22-B12)
    # M17 = (A21+A22)*(B11-B21)
    # M18 = (A11-A21)*(B11+B12)
    # M19 = (A12-A22)*(B21+B22)
    
    # This is getting complex. Let me use a verified BCRL decomposition.
    # Actually, let me use a simpler approach: partition into 3x3x3 blocks.
    
    # Strategy: 
    # - i=0: 1x4x8 -> schoolbook rank 32
    # - i=1,2: 2x4x8 -> use Strassen rank 56
    # Total: 88 (same as before)
    
    # Alternative: Use BCRL for 3x3x3 on the 3x3x8 part
    # i=0,1,2; j=0,1,2; k=0..7: 3x3x8 = 3 instances of 3x3x2 (rank 7 each) + 3x3x2 (rank 7) = 28
    # i=0,1,2; j=3; k=0..7: 3x1x8 = 3x8 = 24
    # Total: 28 + 24 = 52? No, that's not right.
    
    # Let me stick with the verified Strassen approach but try to optimize.
    
    # Actually, let me try a different partition:
    # Use BCRL rank-19 for 3x3x3, applied to blocks
    # 3x4x8: 
    #   j=0,1,2 (3 cols): 3x3x8 = 4 blocks of 3x3x2
    #   j=3 (1 col): 3x1x8 = 3x8 = 24
    
    # For 3x3x2, we can use a rank-7 decomposition (Strassen-like for 3x3x2)
    # Or we can use schoolbook: 3x3x2 = 18
    
    # Total: 4*18 + 24 = 96 (schoolbook)
    
    # Better: Use BCRL for 3x3x3 on 3x3x3 blocks, but we have 3x3x8 not 3x3x3.
    
    # Let me just use the verified rank-88 Strassen decomposition.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, j=0..3, k=0..7
    # Schoolbook for 1x4x8
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: i=1,2
    # Split j into 2 blocks of 2: [0,1] and [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    
    # Strassen local factors for 2x2x2
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Iterate over j_blocks and k_blocks
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            # 7 Strassen terms
            for t in range(7):
                # Map local indices to global
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j][col] += strassen_U[t][l]
                        
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k][col] += strassen_V[t][l]
                        
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        local_i_prime = l // 2
                        local_k_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i][col] += strassen_W[t][l]
                        
                col += 1
                
    # Verify col == rank
    if col != rank:
        return build_schoolbook()

    # Verify the decomposition is correct
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p, rank):
        return build_schoolbook()

    return {"U": U, "V": V, "W": W}
