import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank budget
    # Split j into 0,1 and 2,3
    # For each half (2 cols of A, 2 rows of B):
    #   Split i into 0,1 and 2
    #   For i in 0,1 (2x2x5):
    #       Split k into 0,1; 2,3; 4
    #       2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x2x1 (rank 4) = 18
    #   For i in 2 (1x2x5):
    #       Rank 2
    #   Total per half: 20
    # Total rank: 40
    
    rank = 40
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a block decomposition
    # block_n, block_m, block_p
    # offset_i, offset_j, offset_k
    
    def add_block(block_n, block_m, block_p, off_i, off_j, off_k, r_start):
        # Returns new r_start
        if block_n == 2 and block_m == 2 and block_p == 2:
            # Strassen
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B01
            # M3 = A00(B10-B11) -> Actually standard Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B01
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            
            # We need to map these to U, V, W
            # Local indices i in 0..1, j in 0..1, k in 0..1
            
            # r0: M1
            # U: A00, A11
            # V: B00, B11
            # W: C00, C11 (with coefficients)
            
            # This is complex to hardcode. 
            # Let's use the known Strassen coefficients.
            
            # Term 0:
            # U[0, r] = 1, U[3, r] = 1  (A00, A11)
            # V[0, r] = 1, V[1, r] = 1  (B00, B11)
            # W[0, r] = 1, W[4, r] = 1, W[5, r] = -1, W[9, r] = 1 ? 
            # Wait, W index is k*n + i.
            # Local n=2.
            # C00 (k=0, i=0) -> W[0, r]
            # C11 (k=1, i=1) -> W[1*2+1, r] = W[3, r]
            # C01 (k=1, i=0) -> W[1, r]
            # C10 (k=0, i=1) -> W[2, r]
            
            # Standard Strassen:
            # C00 = M1 + M4 - M5 - M7
            # C01 = M2 + M5
            # C10 = M3 + M6
            # C11 = M1 - M2 + M3 + M7
            
            # Let's implement Strassen ranks carefully.
            
            # M1
            U[off_i*2 + 0][r_start] = 1
            U[off_i*2 + 3][r_start] = 1  # A11 is at local j=1, i=1? 
            # Global index for U: i*m + j.
            # Local i, j. Global i = off_i + i. Global j = off_j + j.
            # U index = (off_i+i)*m + (off_j+j).
            
            # Let's rewrite add_block to take global offsets and handle indexing.
            
            # M1: (A00+A11)(B00+B11)
            U[(off_i+0)*m + (off_j+0)][r_start] += 1
            U[(off_i+1)*m + (off_j+1)][r_start] += 1
            V[(off_j+0)*p + (off_k+0)][r_start] += 1
            V[(off_j+1)*p + (off_k+1)][r_start] += 1
            # C00 += M1, C11 += M1
            W[(off_k+0)*n + (off_i+0)][r_start] += 1
            W[(off_k+1)*n + (off_i+1)][r_start] += 1
            r_start += 1
            
            # M2: (A10+A11)B01
            U[(off_i+1)*m + (off_j+0)][r_start] += 1
            U[(off_i+1)*m + (off_j+1)][r_start] += 1
            V[(off_j+0)*p + (off_k+1)][r_start] += 1
            # C01 += M2, C11 -= M2
            W[(off_k+1)*n + (off_i+0)][r_start] += 1
            W[(off_k+1)*n + (off_i+1)][r_start] -= 1
            r_start += 1
            
            # M3: A00(B10-B11)
            U[(off_i+0)*m + (off_j+0)][r_start] += 1
            V[(off_j+1)*p + (off_k+0)][r_start] += 1
            V[(off_j+1)*p + (off_k+1)][r_start] -= 1
            # C10 += M3, C11 += M3
            W[(off_k+0)*n + (off_i+1)][r_start] += 1
            W[(off_k+1)*n + (off_i+1)][r_start] += 1
            r_start += 1
            
            # M4: A11(B01-B00)
            U[(off_i+1)*m + (off_j+1)][r_start] += 1
            V[(off_j+0)*p + (off_k+1)][r_start] += 1
            V[(off_j+0)*p + (off_k+0)][r_start] -= 1
            # C00 += M4, C01 -= M4
            W[(off_k+0)*n + (off_i+0)][r_start] += 1
            W[(off_k+1)*n + (off_i+0)][r_start] -= 1
            r_start += 1
            
            # M5: (A00+A01)B11
            U[(off_i+0)*m + (off_j+0)][r_start] += 1
            U[(off_i+0)*m + (off_j+1)][r_start] += 1
            V[(off_j+1)*p + (off_k+1)][r_start] += 1
            # C00 -= M5, C01 += M5
            W[(off_k+0)*n + (off_i+0)][r_start] -= 1
            W[(off_k+1)*n + (off_i+0)][r_start] += 1
            r_start += 1
            
            # M6: (A10-A00)(B00+B10)
            U[(off_i+1)*m + (off_j+0)][r_start] += 1
            U[(off_i+0)*m + (off_j+0)][r_start] -= 1
            V[(off_j+0)*p + (off_k+0)][r_start] += 1
            V[(off_j+1)*p + (off_k+0)][r_start] += 1
            # C10 += M6
            W[(off_k+0)*n + (off_i+1)][r_start] += 1
            r_start += 1
            
            # M7: (A01-A11)(B01+B11)
            U[(off_i+0)*m + (off_j+1)][r_start] += 1
            U[(off_i+1)*m + (off_j+1)][r_start] -= 1
            V[(off_j+0)*p + (off_k+1)][r_start] += 1
            V[(off_j+1)*p + (off_k+1)][r_start] += 1
            # C00 -= M7, C11 += M7
            W[(off_k+0)*n + (off_i+0)][r_start] -= 1
            W[(off_k+1)*n + (off_i+1)][r_start] += 1
            r_start += 1
            
        elif block_n == 2 and block_m == 2 and block_p == 1:
            # Schoolbook 2x2x1
            # C00 = A00B00
            # C01 = A01B01
            # C10 = A10B10
            # C11 = A11B11
            # Note: k is fixed to 0 (local)
            
            # 0
            U[(off_i+0)*m + (off_j+0)][r_start] = 1
            V[(off_j+0)*p + (off_k+0)][r_start] = 1
            W[(off_k+0)*n + (off_i+0)][r_start] = 1
            r_start += 1
            
            # 1
            U[(off_i+0)*m + (off_j+1)][r_start] = 1
            V[(off_j+1)*p + (off_k+0)][r_start] = 1
            W[(off_k+0)*n + (off_i+0)][r_start] = 1
            r_start += 1
            
            # 2
            U[(off_i+1)*m + (off_j+0)][r_start] = 1
            V[(off_j+0)*p + (off_k+0)][r_start] = 1
            W[(off_k+0)*n + (off_i+1)][r_start] = 1
            r_start += 1
            
            # 3
            U[(off_i+1)*m + (off_j+1)][r_start] = 1
            V[(off_j+1)*p + (off_k+0)][r_start] = 1
            W[(off_k+0)*n + (off_i+1)][r_start] = 1
            r_start += 1

        elif block_n == 1 and block_m == 2 and block_p == 5:
            # Rank 2
            # C_k = A0 B0k + A1 B1k
            # r0: A0 * B0k
            U[(off_i+0)*m + (off_j+0)][r_start] = 1
            # V for B0k: j=0, k=0..4
            for k_local in range(5):
                V[(off_j+0)*p + (off_k+k_local)][r_start] = 1
            # W for C_k: i=0, k=0..4
            for k_local in range(5):
                W[(off_k+k_local)*n + (off_i+0)][r_start] = 1
            r_start += 1
            
            # r1: A1 * B1k
            U[(off_i+0)*m + (off_j+1)][r_start] = 1
            for k_local in range(5):
                V[(off_j+1)*p + (off_k+k_local)][r_start] = 1
            for k_local in range(5):
                W[(off_k+k_local)*n + (off_i+0)][r_start] = 1
            r_start += 1

        else:
            raise ValueError("Unsupported block")
            
        return r_start

    r = 0
    # Block 1: j in 0,1
    # i in 0,1 (2x2x5)
    r = add_block(2, 2, 5, 0, 0, 0, r) # Wait, my add_block doesn't handle 2x2x5 directly.
    # I need to decompose 2x2x5 into 2x2x2, 2x2x2, 2x2x1.
    
    # Let's do it manually in the main loop.
    
    # Clear U,V,W and restart
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    r = 0
    
    # Helper for 2x2x2 Strassen
    def add_strassen(off_i, off_j, off_k, r):
        # Same as before
        # M1
        U[(off_i+0)*m + (off_j+0)][r] += 1
        U[(off_i+1)*m + (off_j+1)][r] += 1
        V[(off_j+0)*p + (off_k+0)][r] += 1
        V[(off_j+1)*p + (off_k+1)][r] += 1
        W[(off_k+0)*n + (off_i+0)][r] += 1
        W[(off_k+1)*n + (off_i+1)][r] += 1
        r += 1
        # M2
        U[(off_i+1)*m + (off_j+0)][r] += 1
        U[(off_i+1)*m + (off_j+1)][r] += 1
        V[(off_j+0)*p + (off_k+1)][r] += 1
        W[(off_k+1)*n + (off_i+0)][r] += 1
        W[(off_k+1)*n + (off_i+1)][r] -= 1
        r += 1
        # M3
        U[(off_i+0)*m + (off_j+0)][r] += 1
        V[(off_j+1)*p + (off_k+0)][r] += 1
        V[(off_j+1)*p + (off_k+1)][r] -= 1
        W[(off_k+0)*n + (off_i+1)][r] += 1
        W[(off_k+1)*n + (off_i+1)][r] += 1
        r += 1
        # M4
        U[(off_i+1)*m + (off_j+1)][r] += 1
        V[(off_j+0)*p + (off_k+1)][r] += 1
        V[(off_j+0)*p + (off_k+0)][r] -= 1
        W[(off_k+0)*n + (off_i+0)][r] += 1
        W[(off_k+1)*n + (off_i+0)][r] -= 1
        r += 1
        # M5
        U[(off_i+0)*m + (off_j+0)][r] += 1
        U[(off_i+0)*m + (off_j+1)][r] += 1
        V[(off_j+1)*p + (off_k+1)][r] += 1
        W[(off_k+0)*n + (off_i+0)][r] -= 1
        W[(off_k+1)*n + (off_i+0)][r] += 1
        r += 1
        # M6
        U[(off_i+1)*m + (off_j+0)][r] += 1
        U[(off_i+0)*m + (off_j+0)][r] -= 1
        V[(off_j+0)*p + (off_k+0)][r] += 1
        V[(off_j+1)*p + (off_k+0)][r] += 1
        W[(off_k+0)*n + (off_i+1)][r] += 1
        r += 1
        # M7
        U[(off_i+0)*m + (off_j+1)][r] += 1
        U[(off_i+1)*m + (off_j+1)][r] -= 1
        V[(off_j+0)*p + (off_k+1)][r] += 1
        V[(off_j+1)*p + (off_k+1)][r] += 1
        W[(off_k+0)*n + (off_i+0)][r] -= 1
        W[(off_k+1)*n + (off_i+1)][r] += 1
        r += 1
        return r

    def add_schoolbook_2x2x1(off_i, off_j, off_k, r):
        U[(off_i+0)*m + (off_j+0)][r] = 1
        V[(off_j+0)*p + (off_k+0)][r] = 1
        W[(off_k+0)*n + (off_i+0)][r] = 1
        r += 1
        
        U[(off_i+0)*m + (off_j+1)][r] = 1
        V[(off_j+1)*p + (off_k+0)][r] = 1
        W[(off_k+0)*n + (off_i+0)][r] = 1
        r += 1
        
        U[(off_i+1)*m + (off_j+0)][r] = 1
        V[(off_j+0)*p + (off_k+0)][r] = 1
        W[(off_k+0)*n + (off_i+1)][r] = 1
        r += 1
        
        U[(off_i+1)*m + (off_j+1)][r] = 1
        V[(off_j+1)*p + (off_k+0)][r] = 1
        W[(off_k+0)*n + (off_i+1)][r] = 1
        r += 1
        return r

    def add_schoolbook_1x2x5(off_i, off_j, off_k, r):
        # A0 B0k
        U[(off_i+0)*m + (off_j+0)][r] = 1
        for k_local in range(5):
            V[(off_j+0)*p + (off_k+k_local)][r] = 1
            W[(off_k+k_local)*n + (off_i+0)][r] = 1
        r += 1
        
        # A1 B1k
        U[(off_i+0)*m + (off_j+1)][r] = 1
        for k_local in range(5):
            V[(off_j+1)*p + (off_k+k_local)][r] = 1
            W[(off_k+k_local)*n + (off_i+0)][r] = 1
        r += 1
        return r

    # Block j=0,1
    # i=0,1 (2x2x5)
    # k=0,1 (2x2x2)
    r = add_strassen(0, 0, 0, r)
    # k=2,3 (2x2x2)
    r = add_strassen(0, 0, 2, r)
    # k=4 (2x2x1)
    r = add_schoolbook_2x2x1(0, 0, 4, r)
    
    # i=2 (1x2x5)
    r = add_schoolbook_1x2x5(2, 0, 0, r)
    
    # Block j=2,3
    # i=0,1 (2x2x5)
    # k=0,1 (2x2x2)
    r = add_strassen(0, 2, 0, r)
    # k=2,3 (2x2x2)
    r = add_strassen(0, 2, 2, r)
    # k=4 (2x2x1)
    r = add_schoolbook_2x2x1(0, 2, 4, r)
    
    # i=2 (1x2x5)
    r = add_schoolbook_1x2x5(2, 2, 0, r)
    
    if r != rank:
        print(f"Warning: rank mismatch {r} vs {rank}")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
