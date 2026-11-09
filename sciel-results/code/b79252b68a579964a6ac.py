"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block decomposition:
    # A (3x4) -> A_top (1x4), A_bot (2x4)
    # B (4x6) -> B_top (2x6), B_bot (2x6)
    # Split cols of A (and rows of B) into 2+2.
    # Split cols of B into 3+3.
    
    # Blocks:
    # C_00 = A_t L * B_t L + A_t R * B_b L  (M_1,2,3 + M_1,2,3)
    # C_01 = A_t L * B_t R + A_t R * B_b R  (M_1,2,3 + M_1,2,3)
    # C_10 = A_b L * B_t L + A_b R * B_b L  (M_2,2,3 + M_2,2,3)
    # C_11 = A_b L * B_t R + A_b R * B_b R  (M_2,2,3 + M_2,2,3)
    
    # Ranks:
    # M_1,2,3: rank 6
    # M_2,2,3: rank 11 (7 for 2,2,2 + 4 for 2,2,1)
    
    # Total Rank = 4 * 6 + 4 * 11 = 24 + 44 = 68?
    # Wait, C_00 has two M_1,2,3 terms?
    # A_t is 1x4. Split into A_tL (1x2) and A_tR (1x2).
    # B_t is 2x6. Split into B_tL (2x3) and B_tR (2x3).
    # B_b is 2x6. Split into B_bL (2x3) and B_bR (2x3).
    
    # C_top (1x6) = A_tL B_tL + A_tR B_bL + A_tL B_tR + A_tR B_bR
    # Term 1: A_tL (1x2) * B_tL (2x3) -> M_1,2,3. Rank 6.
    # Term 2: A_tR (1x2) * B_bL (2x3) -> M_1,2,3. Rank 6.
    # Term 3: A_tL (1x2) * B_tR (2x3) -> M_1,2,3. Rank 6.
    # Term 4: A_tR (1x2) * B_bR (2x3) -> M_1,2,3. Rank 6.
    # Total for Top: 24.
    
    # C_bot (2x6) = A_bL B_tL + A_bR B_bL + A_bL B_tR + A_bR B_bR
    # A_bL is 2x2. B_tL is 2x3. -> M_2,2,3. Rank 11.
    # A_bR is 2x2. B_bL is 2x3. -> M_2,2,3. Rank 11.
    # A_bL is 2x2. B_tR is 2x3. -> M_2,2,3. Rank 11.
    # A_bR is 2x2. B_bR is 2x3. -> M_2,2,3. Rank 11.
    # Total for Bot: 44.
    
    # Total Rank = 24 + 44 = 68.
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add M_1,2,3
    def add_m123(i_start, j_start, b_j_start, b_k_start, c_i, c_k_start, offset):
        # A: i=i_start, j in j_start..j_start+1
        # B: j in b_j_start..b_j_start+1, k in b_k_start..b_k_start+2
        # C: i=c_i, k in c_k_start..c_k_start+2
        nonlocal col
        for j_idx in range(2):
            for k_idx in range(3):
                # U row: i_start * m + (j_start + j_idx)
                U[i_start * m + j_start + j_idx][col] = 1
                # V row: (b_j_start + j_idx) * p + (b_k_start + k_idx)
                V[(b_j_start + j_idx) * p + b_k_start + k_idx][col] = 1
                # W row: (c_k_start + k_idx) * n + c_i
                W[(c_k_start + k_idx) * n + c_i][col] = 1
                col += 1

    # Helper to add M_2,2,3
    def add_m223(i_start, j_start, b_j_start, b_k_start, c_i_start, c_k_start, offset):
        # A: i in i_start..i_start+1, j in j_start..j_start+1
        # B: j in b_j_start..b_j_start+1, k in b_k_start..b_k_start+2
        # C: i in c_i_start..c_i_start+1, k in c_k_start..c_k_start+2
        nonlocal col
        
        # Part 1: M_2,2,2 for k in b_k_start..b_k_start+1
        # Strassen
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B10-B11)
        # M4: A11(B01-B00)
        # M5: (A00+A10)B11
        # M6: (A10-A00)(B00+B10)
        # M7: (A01-A11)(B11+B01)
        
        # Local indices for A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # Local indices for B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # Local indices for C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        # U rows: i_start+m*2 + j_start + j_loc
        # V rows: (b_j_start + j_loc)*p + b_k_start + k_loc
        # W rows: (c_k_start + k_loc)*n + c_i_start + i_loc
        
        # We need to map the 7 products to columns col..col+6
        # Each product is sum/diff of U,V,W terms.
        
        # M1
        # A00+A11: U[0]+=1, U[3]+=1
        # B00+B11: V[0]+=1, V[3]+=1
        # C: C00+=1, C01+=1, C10+=1, C11+=1? No.
        # M1 contributes to C00, C01, C10, C11.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M1 - M2 + M6
        # C10 = M2 + M4 - M7 + M5
        # C11 = M1 - M3 + M2 - M6
        
        # Let's just store the coefficients for U, V, W for each of the 7 terms.
        
        # Term 1
        # U: A00(0,0)=1, A11(1,1)=1
        # V: B00(0,0)=1, B11(1,1)=1
        # W: C00(0,0)=1, C01(0,1)=1, C10(1,0)=1, C11(1,1)=1
        U[i_start * m + j_start][col] += 1
        U[i_start * m + j_start + 1] += 0 # A01 is not in M1
        U[i_start * m + j_start + 2] += 0 # A10
        U[i_start * m + j_start + 3] += 1 # A11
        
        V[(b_j_start) * p + b_k_start] += 1 # B00
        V[(b_j_start + 1) * p + b_k_start + 1] += 1 # B11
        
        W[(c_k_start) * n + c_i_start] += 1 # C00
        W[(c_k_start + 1) * n + c_i_start] += 1 # C01
        W[(c_k_start) * n + c_i_start + 1] += 1 # C10
        W[(c_k_start + 1) * n + c_i_start + 1] += 1 # C11
        col += 1
        
        # Term 2
        # (A10+A11)B00
        # U: A10(1,0)=1, A11(1,1)=1
        # V: B00(0,0)=1
        # W: C10(1,0)=1, C11(1,1)=1
        U[i_start * m + j_start + 2] += 1
        U[i_start * m + j_start + 3] += 1
        V[b_j_start * p + b_k_start] += 1
        W[c_k_start * n + c_i_start + 1] += 1
        W[(c_k_start + 1) * n + c_i_start + 1] += 1
        col += 1
        
        # Term 3
        # A00(B10-B11)
        # U: A00(0,0)=1
        # V: B10(1,0)=1, B11(1,1)=-1
        # W: C01(0,1)=1, C11(1,1)=1
        U[i_start * m + j_start] += 1
        V[(b_j_start + 1) * p + b_k_start] += 1
        V[(b_j_start + 1) * p + b_k_start + 1] += -1
        W[(c_k_start + 1) * n + c_i_start] += 1
        W[(c_k_start + 1) * n + c_i_start + 1] += 1
        col += 1
        
        # Term 4
        # A11(B01-B00)
        # U: A11(1,1)=1
        # V: B01(0,1)=1, B00(0,0)=-1
        # W: C00(0,0)=1, C10(1,0)=1
        U[i_start * m + j_start + 3] += 1
        V[b_j_start * p + b_k_start + 1] += 1
        V[b_j_start * p + b_k_start] += -1
        W[c_k_start * n + c_i_start] += 1
        W[c_k_start * n + c_i_start + 1] += 1
        col += 1
        
        # Term 5
        # (A00+A10)B11
        # U: A00(0,0)=1, A10(1,0)=1
        # V: B11(1,1)=1
        # W: C00(0,0)=-1, C10(1,0)=1
        U[i_start * m + j_start] += 1
        U[i_start * m + j_start + 2] += 1
        V[(b_j_start + 1) * p + b_k_start + 1] += 1
        W[c_k_start * n + c_i_start] += -1
        W[c_k_start * n + c_i_start + 1] += 1
        col += 1
        
        # Term 6
        # (A10-A00)(B00+B10)
        # U: A10(1,0)=1, A00(0,0)=-1
        # V: B00(0,0)=1, B10(1,0)=1
        # W: C01(0,1)=1, C11(1,1)=-1
        U[i_start * m + j_start + 2] += 1
        U[i_start * m + j_start] += -1
        V[b_j_start * p + b_k_start] += 1
        V[(b_j_start + 1) * p + b_k_start] += 1
        W[(c_k_start + 1) * n + c_i_start] += 1
        W[(c_k_start + 1) * n + c_i_start + 1] += -1
        col += 1
        
        # Term 7
        # (A01-A11)(B11+B01)
        # U: A01(0,1)=1, A11(1,1)=-1
        # V: B11(1,1)=1, B01(0,1)=1
        # W: C00(0,0)=1, C10(1,0)=-1
        U[i_start * m + j_start + 1] += 1
        U[i_start * m + j_start + 3] += -1
        V[(b_j_start + 1) * p + b_k_start + 1] += 1
        V[b_j_start * p + b_k_start + 1] += 1
        W[c_k_start * n + c_i_start] += 1
        W[c_k_start * n + c_i_start + 1] += -1
        col += 1
        
        # Part 2: M_2,2,1 for k = b_k_start + 2
        # C0 = A00 B02 + A01 B12 + A10 B02? No.
        # B has rows j=0,1. Cols k=2.
        # C0 = A00 B02 + A01 B02? No.
        # C_i0 = sum_j A_ij B_j0.
        # Here p=1 locally. B is 2x1.
        # C_00 = A_00 B_00 + A_01 B_10
        # C_10 = A_10 B_00 + A_11 B_10
        # Terms:
        # 1. A00 B02
        # 2. A01 B12
        # 3. A10 B02
        # 4. A11 B12
        
        # Term 8: A00 B02
        U[i_start * m + j_start] += 1
        V[b_j_start * p + b_k_start + 2] += 1
        W[(c_k_start + 2) * n + c_i_start] += 1
        col += 1
        
        # Term 9: A01 B12
        U[i_start * m + j_start + 1] += 1
        V[(b_j_start + 1) * p + b_k_start + 2] += 1
        W[(c_k_start + 2) * n + c_i_start] += 1
        col += 1
        
        # Term 10: A10 B02
        U[i_start * m + j_start + 2] += 1
        V[b_j_start * p + b_k_start + 2] += 1
        W[(c_k_start + 2) * n + c_i_start + 1] += 1
        col += 1
        
        # Term 11: A11 B12
        U[i_start * m + j_start + 3] += 1
        V[(b_j_start + 1) * p + b_k_start + 2] += 1
        W[(c_k_start + 2) * n + c_i_start + 1] += 1
        col += 1

    # Call add_m123 for Top blocks
    # A_tL (1x2): i=0, j=0,1. B_tL (2x3): j=0,1; k=0,1,2. C: i=0, k=0,1,2.
    add_m123(0, 0, 0, 0, 0, 0, 0)
    # A_tR (1x2): i=0, j=2,3. B_bL (2x3): j=2,3; k=0,1,2. C: i=0, k=0,1,2.
    add_m123(0, 2, 2, 0, 0, 0, 0)
    # A_tL (1x2): i=0, j=0,1. B_tR (2x3): j=0,1; k=3,4,5. C: i=0, k=3,4,5.
    add_m123(0, 0, 0, 3, 0, 3, 0)
    # A_tR (1x2): i=0, j=2,3. B_bR (2x3): j=2,3; k=3,4,5. C: i=0, k=3,4,5.
    add_m123(0, 2, 2, 3, 0, 3, 0)
    
    # Call add_m223 for Bot blocks
    # A_bL (2x2): i=1,2; j=0,1. B_tL (2x3): j=0,1; k=0,1,2. C: i=1,2; k=0,1,2.
    add_m223(1, 0, 0, 0, 1, 0, 0)
    # A_bR (2x2): i=1,2; j=2,3. B_bL (2x3): j=2,3; k=0,1,2. C: i=1,2; k=0,1,2.
    add_m223(1, 2, 2, 0, 1, 0, 0)
    # A_bL (2x2): i=1,2; j=0,1. B_tR (2x3): j=0,1; k=3,4,5. C: i=1,2; k=3,4,5.
    add_m223(1, 0, 0, 3, 1, 3, 0)
    # A_bR (2x2): i=1,2; j=2,3. B_bR (2x3): j=2,3; k=3,4,5. C: i=1,2; k=3,4,5.
    add_m223(1, 2, 2, 3, 1, 3, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
