import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_strassen(a_offset, b_offset, c_offset, start_rank):
        # a_offset: index in A (0..3)
        # b_offset: index in B (0..13)
        # c_offset: index in C (0..13)
        # Adds 7 terms
        r = start_rank
        # M1 = (A11+A22)(B11+B22)
        # A indices: 0 (11), 3 (22)
        # B indices: b_offset (11), b_offset+2 (22)
        # C indices: c_offset (11), c_offset+1 (12), c_offset+2 (21), c_offset+3 (22)
        
        # Term 1
        U[a_offset][r] += 1
        U[a_offset+3][r] += 1
        V[b_offset][r] += 1
        V[b_offset+2][r] += 1
        W[c_offset][r] += 1
        W[c_offset+3][r] += 1
        r += 1
        
        # Term 2: (A21+A22)B11
        U[a_offset+2][r] += 1
        U[a_offset+3][r] += 1
        V[b_offset][r] += 1
        W[c_offset+2][r] += 1
        W[c_offset+3][r] -= 1
        r += 1
        
        # Term 3: A11(B12-B22)
        U[a_offset][r] += 1
        V[b_offset+1][r] += 1
        V[b_offset+3][r] -= 1
        W[c_offset+1][r] += 1
        W[c_offset+3][r] += 1
        r += 1
        
        # Term 4: A22(B21-B11)
        U[a_offset+3][r] += 1
        V[b_offset+2][r] += 1
        V[b_offset][r] -= 1
        W[c_offset][r] += 1
        W[c_offset+2][r] += 1
        r += 1
        
        # Term 5: (A11+A12)B22
        U[a_offset][r] += 1
        U[a_offset+1][r] += 1
        V[b_offset+3][r] += 1
        W[c_offset][r] -= 1
        W[c_offset+1][r] += 1
        r += 1
        
        # Term 6: (A21-A11)(B11+B12)
        U[a_offset+2][r] += 1
        U[a_offset][r] -= 1
        V[b_offset][r] += 1
        V[b_offset+1][r] += 1
        W[c_offset+3][r] += 1
        r += 1
        
        # Term 7: (A12-A22)(B21+B22)
        U[a_offset+1][r] += 1
        U[a_offset+3][r] -= 1
        V[b_offset+2][r] += 1
        V[b_offset+3][r] += 1
        W[c_offset][r] += 1
        r += 1
        
        return r

    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, start_rank):
        # a_offset: index in A (0..3)
        # b_offset: index in B (0..13)
        # c_offset: index in C (0..13)
        # Adds 4 terms
        r = start_rank
        
        # M1 = A11 B11
        U[a_offset][r] += 1
        V[b_offset][r] += 1
        W[c_offset][r] += 1
        r += 1
        
        # M2 = A12 B21
        U[a_offset+1][r] += 1
        V[b_offset+1][r] += 1
        W[c_offset][r] += 1
        r += 1
        
        # M3 = A21 B11
        U[a_offset+2][r] += 1
        V[b_offset][r] += 1
        W[c_offset+1][r] += 1
        r += 1
        
        # M4 = A22 B21
        U[a_offset+3][r] += 1
        V[b_offset+1][r] += 1
        W[c_offset+1][r] += 1
        r += 1
        
        return r

    # 1. C_2 = A * B_2 (cols 5-6 of B, cols 5-6 of C)
    # A offset 0, B offset 10, C offset 10
    current_rank = add_strassen(0, 10, 10, current_rank)
    
    # 2. C_1b = A * B_1b (cols 3-4 of B, cols 3-4 of C)
    # A offset 0, B offset 6, C offset 6
    current_rank = add_strassen(0, 6, 6, current_rank)
    
    # 3. C_1a1 = A * B_1a1 (cols 0-1 of B, cols 0-1 of C)
    # A offset 0, B offset 0, C offset 0
    current_rank = add_strassen(0, 0, 0, current_rank)
    
    # 4. C_1a2 = A * B_1a2 (col 2 of B, col 2 of C)
    # A offset 0, B offset 4, C offset 4
    current_rank = add_schoolbook_2x2x1(0, 4, 4, current_rank)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
