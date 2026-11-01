import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    # Split m=4 into two halves of size 2.
    # Each half corresponds to a 2x2x8 multiplication.
    # Rank of 2x2x8 is 28 (4 blocks of 2x2x2, each rank 7).
    # Total rank 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Helper to add a 2x2x2 Strassen decomposition
    # i_off: offset for i in U/W (0 or 1 for the 2 rows of A/C)
    # j_off: offset for j in U/V (0 or 2 for the 2 cols of A / 2 rows of B)
    # k_off: offset for k in V/W (0, 2, 4, 6 for the 4 cols of B/C)
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Mapping of Strassen terms to U, V, W updates
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        # i_base in {0, 1} (since n=2, we handle i=0,1 separately? No, n=2 is fixed)
        # Actually n=2, so i is 0 or 1.
        # j_base is 0 or 2.
        # k_base is 0, 2, 4, 6.
        
        # The 2x2x2 block is A[i_base, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]
        # Result C[i_base, k_base:k_base+2]
        
        # We need to add 7 terms.
        # Term 1: (A[i_base, j_base] + A[i_base, j_base+1]) * (B[j_base, k_base] + B[j_base, k_base+1])
        # Wait, Strassen uses A11, A10 etc.
        # Let's map local indices 0,1 to global.
        # i_local 0, 1? No, n=2. The block is 2x2 in j, but only 1 row in i?
        # No, A is 2x4. Split j into 2x2 blocks.
        # For a fixed i (0 or 1), A[i, j_base:j_base+2] is a 1x2 vector.
        # B[j_base:j_base+2, k_base:k_base+2] is a 2x2 matrix.
        # This is 1x2 * 2x2 = 1x2.
        # Rank of 1x2x2?
        # C_k = A_0 B_0k + A_1 B_1k.
        # This is just 4 multiplications. Rank 4.
        # So 2x2x8 (which is 4 blocks of 1x2x2) would be rank 16?
        # Wait.
        # Earlier I said Rank(2x2x8) = 28.
        # 2x2x8 is A (2x2) * B (2x8).
        # Split B into four 2x2 blocks.
        # C = A B1 + A B2 + A B3 + A B4.
        # Each A Bi is 2x2 * 2x2. Rank 7.
        # Total 28.
        # My previous "1x2x2" analysis was for splitting A's rows?
        # No, A is 2x2. We don't split A's rows.
        # So for each 2x2 block of B, we do a full 2x2x2 multiplication.
        # So we need 4 Strassen decompositions.
        
        for t in range(7):
            # Determine U, V, W values for term t
            # Strassen terms for A (2x2) and B (2x2)
            # A indices: (0,0), (1,0), (0,1), (1,1) -> i, j_local
            # B indices: (0,0), (1,0), (0,1), (1,1) -> j_local, k_local
            
            # U[i*m + j]
            # V[j*p + k]
            # W[k*n + i]
            
            if t == 0:
                # M1 = (A00+A11)(B00+B11)
                # A00: i=0, j=0. A11: i=1, j=1.
                # B00: j=0, k=0. B11: j=1, k=1.
                # U: A00=1, A11=1.
                # V: B00=1, B11=1.
                # W: C00 += 1, C11 += 1.
                U[i_base * m + j_base][col] += 1
                U[(i_base + 1) * m + (j_base + 1)][col] += 1
                V[j_base * p + k_base][col] += 1
                V[(j_base + 1) * p + (k_base + 1)][col] += 1
                W[k_base * n + i_base][col] += 1
                W[(k_base + 1) * n + (i_base + 1)][col] += 1
            elif t == 1:
                # M2 = (A10+A11)B00
                # A10: i=1, j=0. A11: i=1, j=1.
                # B00: j=0, k=0.
                U[i_base * m + j_base][col] += 1
                U[(i_base + 1) * m + (j_base + 1)][col] += 1
                V[j_base * p + k_base][col] += 1
                W[k_base * n + i_base][col] += 1
                W[k_base * n + (i_base + 1)][col] += 1
            elif t == 2:
                # M3 = A00(B10-B11)
                # A00: i=0, j=0.
                # B10: j=1, k=0. B11: j=1, k=1.
                U[i_base * m + j_base][col] += 1
                V[(j_base + 1) * p + k_base][col] += 1
                V[(j_base + 1) * p + (k_base + 1)][col] -= 1
                W[k_base * n + i_base][col] += 1
                W[(k_base + 1) * n + i_base][col] -= 1
            elif t == 3:
                # M4 = A11(B01-B00)
                # A11: i=1, j=1.
                # B01: j=0, k=1. B00: j=0, k=0.
                U[(i_base + 1) * m + (j_base + 1)][col] += 1
                V[j_base * p + (k_base + 1)][col] += 1
                V[j_base * p + k_base][col] -= 1
                W[(k_base + 1) * n + (i_base + 1)][col] += 1
                W[k_base * n + (i_base + 1)][col] -= 1
            elif t == 4:
                # M5 = (A00+A01)B11
                # A00: i=0, j=0. A01: i=0, j=1.
                # B11: j=1, k=1.
                U[i_base * m + j_base][col] += 1
                U[i_base * m + (j_base + 1)][col] += 1
                V[(j_base + 1) * p + (k_base + 1)][col] += 1
                W[(k_base + 1) * n + i_base][col] += 1
                W[(k_base + 1) * n + (i_base + 1)][col] += 1
            elif t == 5:
                # M6 = (A10-A00)(B00+B10)
                # A10: i=1, j=0. A00: i=0, j=0.
                # B00: j=0, k=0. B10: j=1, k=0.
                U[i_base * m + j_base][col] -= 1
                U[(i_base + 1) * m + j_base][col] += 1
                V[j_base * p + k_base][col] += 1
                V[(j_base + 1) * p + k_base][col] += 1
                W[k_base * n + i_base][col] -= 1
                W[k_base * n + (i_base + 1)][col] += 1
            elif t == 6:
                # M7 = (A01-A11)(B01+B11)
                # A01: i=0, j=1. A11: i=1, j=1.
                # B01: j=0, k=1. B11: j=1, k=1.
                U[i_base * m + (j_base + 1)][col] += 1
                U[(i_base + 1) * m + (j_base + 1)][col] -= 1
                V[j_base * p + (k_base + 1)][col] += 1
                V[(j_base + 1) * p + (k_base + 1)][col] += 1
                W[(k_base + 1) * n + i_base][col] += 1
                W[(k_base + 1) * n + (i_base + 1)][col] -= 1
            
            col += 1

    # We have 4 blocks of B (k=0..1, 2..3, 4..5, 6..7)
    # And 2 blocks of A (j=0..1, 2..3)
    # We need to combine them.
    # For j_block 0 (j=0,1), we use k_blocks 0,1,2,3.
    # For j_block 1 (j=2,3), we use k_blocks 0,1,2,3.
    
    # Wait, A is 2x4. B is 4x8.
    # A = [A_L, A_R] (2x2 each).
    # B = [B_T; B_B] (2x8 each).
    # C = A_L B_T + A_R B_B.
    # B_T is 2x8. Split into 4 2x2 blocks.
    # B_B is 2x8. Split into 4 2x2 blocks.
    
    # So for j_base = 0 (A_L, B_T):
    # k_base = 0, 2, 4, 6.
    for k_base in [0, 2, 4, 6]:
        add_strassen(0, 0, k_base) # i_base=0? No, i_base is row of A.
        # Wait, add_strassen takes i_base.
        # In 2x2x2, i is 0,1.
        # My add_strassen assumes i_base is the row index?
        # No, in 2x2x2, A is 2x2. i is 0,1.
        # So i_base should be 0?
        # But A is 2x4. i is 0,1.
        # So for the whole 2x2x2 block, i covers 0,1.
        # So i_base is not an offset, it's the start of i?
        # Since n=2, i is always 0,1.
        # So i_base should be 0?
        # But U index is i*m + j.
        # If i_base=0, U[0*m+j] and U[1*m+j].
        # This covers both rows of A.
        # So yes, i_base=0.
        
    # Correction: add_strassen(i_base, j_base, k_base)
    # i_base is used in U and W.
    # If n=2, and we are doing a 2x2x2 multiplication, the i indices are 0 and 1.
    # So i_base should be 0.
    # But wait, if I call add_strassen(0, 0, 0), it updates U[0,0], U[1,1] etc.
    # This is correct for the first 2x2x2 block.
    
    # So:
    # Block j=0..1 (A_L), B_T (j=0..1 of B)
    for k_base in [0, 2, 4, 6]:
        add_strassen(0, 0, k_base)
        
    # Block j=2..3 (A_R), B_B (j=2..3 of B)
    for k_base in [0, 2, 4, 6]:
        add_strassen(0, 2, k_base)
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick verification
    res = construct()
    print("Rank:", len(res['U'][0]))
