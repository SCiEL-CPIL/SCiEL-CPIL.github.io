"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen decomposition:
    # A is 2x4, B is 4x8.
    # Split B into B1 (4x4) and B2 (4x4).
    # C = A * B = A*B1 + A*B2.
    # For A*B1 (2x4 * 4x4), split A into A_left (2x2) and A_right (2x2), B1 into B1_top (2x4) and B1_bottom (2x4).
    # Wait, standard Strassen splits the middle dimension.
    # A (n x m), B (m x p).
    # m=4. Split m into 2+2.
    # A = [A1 A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x8.
    # Then C = A1 B1 + A2 B2.
    # Now A1 B1 is 2x2 * 2x8.
    # We can split the 8 columns of B1 into four 2x2 blocks? No, that's not standard Strassen.
    # Standard Strassen applies to square matrices or rectangular where dimensions allow halving.
    # For 2x2 * 2x8:
    # We can view 2x8 as 2x(4*2). 
    # Let's stick to the recursive structure if possible, or just use rank 7 for each 2x2x2 block.
    # The 2x2x8 multiplication can be decomposed into four 2x2x2 multiplications if we split the 8 columns into 2x2 blocks?
    # No, the columns of B are 8. B1 is 2x8.
    # Let B1 = [B11 B12 B13 B14] where Bij are 2x2.
    # A1 * B1 = A1 * B11 + A1 * B12 + A1 * B13 + A1 * B14.
    # Each term is 2x2 * 2x2.
    # Each 2x2 * 2x2 multiplication can be done with rank 7 (Strassen).
    # So A1 B1 requires 4 * 7 = 28 rank terms.
    # Similarly A2 B2 requires 4 * 7 = 28 rank terms.
    # Total rank = 56.
    # This is valid and much better than 64.
    
    # Let's implement this.
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (16, rank)
    
    # Strassen coefficients for 2x2 matrix multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the indices for a single 2x2x2 Strassen block.
    # Input U indices (A part):
    # A11 -> u0, A12 -> u1, A21 -> u2, A22 -> u3
    # Input V indices (B part):
    # B11 -> v0, B12 -> v1, B21 -> v2, B22 -> v3
    # Output W indices (C part):
    # C11 -> w0, C12 -> w1, C21 -> w2, C22 -> w3
    
    # We need to add contributions to U, V, W for each of the 7 terms.
    
    def strassen_block(u_base, v_base, w_base, U, V, W, rank_idx):
        # u_base, v_base, w_base are the base indices for U, V, W for this 2x2 block.
        # We will add 7 entries to U, V, W starting at rank_idx.
        
        # Term 1: (A11 + A22) * (B11 + B22)
        U[u_base][rank_idx] += 1   # A11
        U[u_base+3][rank_idx] += 1  # A22
        V[v_base][rank_idx] += 1    # B11
        V[v_base+3][rank_idx] += 1  # B22
        # Output: C11, C22
        W[w_base][rank_idx] += 1    # C11
        W[w_base+3][rank_idx] += 1  # C22
        
        # Term 2: (A21 + A22) * B11
        U[u_base+2][rank_idx+1] += 1 # A21
        U[u_base+3][rank_idx+1] += 1 # A22
        V[v_base][rank_idx+1] += 1   # B11
        # Output: C21, C22
        W[w_base+2][rank_idx+1] += 1 # C21
        W[w_base+3][rank_idx+1] -= 1 # C22 (subtract M2)
        
        # Term 3: A11 * (B12 - B22)
        U[u_base][rank_idx+2] += 1   # A11
        V[v_base+1][rank_idx+2] += 1 # B12
        V[v_base+3][rank_idx+2] -= 1 # B22
        # Output: C12, C22
        W[w_base+1][rank_idx+2] += 1 # C12
        W[w_base+3][rank_idx+2] += 1 # C22 (add M3)
        
        # Term 4: A22 * (B21 - B11)
        U[u_base+3][rank_idx+3] += 1 # A22
        V[v_base+2][rank_idx+3] += 1 # B21
        V[v_base][rank_idx+3] -= 1   # B11
        # Output: C11, C21
        W[w_base][rank_idx+3] += 1   # C11 (add M4)
        W[w_base+2][rank_idx+3] += 1 # C21 (add M4)
        
        # Term 5: (A11 + A12) * B22
        U[u_base][rank_idx+4] += 1   # A11
        U[u_base+1][rank_idx+4] += 1 # A12
        V[v_base+3][rank_idx+4] += 1 # B22
        # Output: C11, C12
        W[w_base][rank_idx+4] -= 1   # C11 (subtract M5)
        W[w_base+1][rank_idx+4] += 1 # C12 (add M5)
        
        # Term 6: (A21 - A11) * (B11 + B12)
        U[u_base+2][rank_idx+5] += 1 # A21
        U[u_base][rank_idx+5] -= 1   # A11
        V[v_base][rank_idx+5] += 1   # B11
        V[v_base+1][rank_idx+5] += 1 # B12
        # Output: C22
        W[w_base+3][rank_idx+5] += 1 # C22 (add M6)
        
        # Term 7: (A12 - A22) * (B21 + B22)
        U[u_base+1][rank_idx+6] += 1 # A12
        U[u_base+3][rank_idx+6] -= 1 # A22
        V[v_base+2][rank_idx+6] += 1 # B21
        V[v_base+3][rank_idx+6] += 1 # B22
        # Output: C11
        W[w_base][rank_idx+6] += 1   # C11 (add M7)
        
        return rank_idx + 7

    # Now construct the full decomposition.
    # A is 2x4. Split into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # B is 4x8. Split into B1 (2x8, rows 0-1) and B2 (2x8, rows 2-3).
    # C = A1*B1 + A2*B2.
    
    # For A1*B1 (2x2 * 2x8):
    # B1 is 2x8. Split into B11, B12, B13, B14 (each 2x2).
    # B11 cols 0-1, B12 cols 2-3, B13 cols 4-5, B14 cols 6-7.
    
    # For A2*B2 (2x2 * 2x8):
    # B2 is 2x8. Split into B21, B22, B23, B24 (each 2x2).
    # B21 cols 0-1, B22 cols 2-3, B23 cols 4-5, B24 cols 6-7.
    
    rank = 56
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    current_rank = 0
    
    # Process A1 * B1
    # A1 indices in U:
    # A11: row 0, col 0 -> U[0]
    # A12: row 0, col 1 -> U[1]
    # A21: row 1, col 0 -> U[2]
    # A22: row 1, col 1 -> U[3]
    # So u_base = 0.
    
    # B1 indices in V:
    # B1 is rows 0-1 of B.
    # B11 (cols 0-1):
    # B11_11: row 0, col 0 -> V[0*8+0] = V[0]
    # B11_12: row 0, col 1 -> V[0*8+1] = V[1]
    # B11_21: row 1, col 0 -> V[1*8+0] = V[8]
    # B11_22: row 1, col 1 -> V[1*8+1] = V[9]
    # So for block j (0..3), v_base = j*2? No.
    # B11 corresponds to B[0:2, 0:2].
    # V indices:
    # B11_11 -> 0
    # B11_12 -> 1
    # B11_21 -> 8
    # B11_22 -> 9
    
    # Wait, my Strassen function expects v_base such that:
    # B11 -> v_base
    # B12 -> v_base+1
    # B21 -> v_base+2
    # B22 -> v_base+3
    # This assumes B is stored in row-major order for the 2x2 block.
    # But V is 32xrank, corresponding to B (4x8) in row-major order.
    # B element (r, c) is at V[r*8+c].
    # For a 2x2 block B[r1:r1+2, c1:c1+2]:
    # B11 (r1, c1) -> V[r1*8 + c1]
    # B12 (r1, c1+1) -> V[r1*8 + c1+1]
    # B21 (r1+1, c1) -> V[(r1+1)*8 + c1]
    # B22 (r1+1, c1+1) -> V[(r1+1)*8 + c1+1]
    # These are NOT contiguous in V unless c1+1 == c1+2? No.
    # My Strassen function assumes contiguous v indices. I need to adapt it or map indices explicitly.
    # Let's modify strassen_block to take explicit indices for U, V, W.
    
    def strassen_block_indices(u11, u12, u21, u22, v11, v12, v21, v22, w11, w12, w21, w22, U, V, W, rank_idx):
        # Term 1
        U[u11][rank_idx] += 1
        U[u22][rank_idx] += 1
        V[v11][rank_idx] += 1
        V[v22][rank_idx] += 1
        W[w11][rank_idx] += 1
        W[w22][rank_idx] += 1
        
        # Term 2
        U[u21][rank_idx+1] += 1
        U[u22][rank_idx+1] += 1
        V[v11][rank_idx+1] += 1
        W[w21][rank_idx+1] += 1
        W[w22][rank_idx+1] -= 1
        
        # Term 3
        U[u11][rank_idx+2] += 1
        V[v12][rank_idx+2] += 1
        V[v22][rank_idx+2] -= 1
        W[w12][rank_idx+2] += 1
        W[w22][rank_idx+2] += 1
        
        # Term 4
        U[u22][rank_idx+3] += 1
        V[v21][rank_idx+3] += 1
        V[v11][rank_idx+3] -= 1
        W[w11][rank_idx+3] += 1
        W[w21][rank_idx+3] += 1
        
        # Term 5
        U[u11][rank_idx+4] += 1
        U[u12][rank_idx+4] += 1
        V[v22][rank_idx+4] += 1
        W[w11][rank_idx+4] -= 1
        W[w12][rank_idx+4] += 1
        
        # Term 6
        U[u21][rank_idx+5] += 1
        U[u11][rank_idx+5] -= 1
        V[v11][rank_idx+5] += 1
        V[v12][rank_idx+5] += 1
        W[w22][rank_idx+5] += 1
        
        # Term 7
        U[u12][rank_idx+6] += 1
        U[u22][rank_idx+6] -= 1
        V[v21][rank_idx+6] += 1
        V[v22][rank_idx+6] += 1
        W[w11][rank_idx+6] += 1
        
        return rank_idx + 7

    # A1 (2x2) is A[0:2, 0:2].
    # u11 = A[0,0] -> U[0]
    # u12 = A[0,1] -> U[1]
    # u21 = A[1,0] -> U[2]
    # u22 = A[1,1] -> U[3]
    
    # A2 (2x2) is A[0:2, 2:4].
    # u11 = A[0,2] -> U[2]
    # u12 = A[0,3] -> U[3]
    # u21 = A[1,2] -> U[6]
    # u22 = A[1,3] -> U[7]
    
    # For each of the 4 blocks in B1 and 4 blocks in B2:
    # Block j (0..3) corresponds to columns 2j, 2j+1.
    # B1 (rows 0-1):
    # B11 (cols 0-1): v11=V[0], v12=V[1], v21=V[8], v22=V[9]
    # B12 (cols 2-3): v11=V[2], v12=V[3], v21=V[10], v22=V[11]
    # ...
    # B2 (rows 2-3):
    # B21 (cols 0-1): v11=V[16], v12=V[17], v21=V[24], v22=V[25]
    # ...
    
    # W indices for C (2x8).
    # C element (i, k) -> W[k*2 + i].
    # For block j (cols 2j, 2j+1):
    # C11 (0, 2j) -> W[2j*2 + 0] = W[4j]
    # C12 (0, 2j+1) -> W[2j*2 + 1] = W[4j+1]
    # C21 (1, 2j) -> W[2j*2 + 2] = W[4j+2]
    # C22 (1, 2j+1) -> W[2j*2 + 3] = W[4j+3]
    
    # Wait, W is p*n = 16 rows.
    # W index is k*n + i = k*2 + i.
    # So for k=2j, i=0 -> 4j.
    # For k=2j, i=1 -> 4j+1.
    # For k=2j+1, i=0 -> 4j+2.
    # For k=2j+1, i=1 -> 4j+3.
    
    # Let's loop j from 0 to 3.
    for j in range(4):
        # W indices for this block
        w11 = 4*j
        w12 = 4*j + 1
        w21 = 4*j + 2
        w22 = 4*j + 3
        
        # A1 * B1_block_j
        # A1 u indices: 0, 1, 2, 3
        # B1 v indices:
        # row 0: 2j, 2j+1
        # row 1: 8+2j, 9+2j
        v11 = 2*j
        v12 = 2*j + 1
        v21 = 8 + 2*j
        v22 = 9 + 2*j
        
        current_rank = strassen_block_indices(0, 1, 2, 3, v11, v12, v21, v22, w11, w12, w21, w22, U, V, W, current_rank)
        
        # A2 * B2_block_j
        # A2 u indices: 2, 3, 6, 7
        # B2 v indices:
        # row 2: 16+2j, 17+2j
        # row 3: 24+2j, 25+2j
        v11 = 16 + 2*j
        v12 = 17 + 2*j
        v21 = 24 + 2*j
        v22 = 25 + 2*j
        
        current_rank = strassen_block_indices(2, 3, 6, 7, v11, v12, v21, v22, w11, w12, w21, w22, U, V, W, current_rank)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
