"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct the decomposition by splitting the 3x4 matrix into
    # a 2x4 top part and a 1x4 bottom part.
    # Top part: 2x4 * 4x8. Split 8 into two 4s.
    #   Each 2x4 * 4x4 is handled by 4 Strassen 2x2x2 blocks. Rank 4*7=28.
    #   Total top rank: 28 * 2 = 56.
    # Bottom part: 1x4 * 4x8. Split 8 into two 4s.
    #   Each 1x4 * 4x4 is handled by 4 schoolbook 1x2x2 blocks. Rank 4*4=16.
    #   Total bottom rank: 16 * 2 = 32.
    # Total rank: 88.
    
    total_rank = 88
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a 2x2x2 Strassen block
    # local i (0..1), j (0..1), k (0..1)
    # global offset for A (i_off, j_off), B (j_off, k_off), C (k_off, i_off)
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal current_rank
        # Strassen factors for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U factors (2x7)
        # V factors (2x7)
        # W factors (2x7)
        
        # Term 0: M1
        U[(i_off+0)*2 + 0][current_rank] = 1
        U[(i_off+1)*2 + 1][current_rank] = 1
        V[(j_off+0)*2 + 0][current_rank] = 1
        V[(j_off+1)*2 + 1][current_rank] = 1
        W[(k_off+0)*2 + 0][current_rank] = 1
        W[(k_off+1)*2 + 1][current_rank] = 1
        current_rank += 1
        
        # Term 1: M2
        U[(i_off+1)*2 + 0][current_rank] = 1
        U[(i_off+1)*2 + 1][current_rank] = 1
        V[(j_off+0)*2 + 0][current_rank] = 1
        W[(k_off+0)*2 + 1][current_rank] = 1
        current_rank += 1
        
        # Term 2: M3
        U[(i_off+0)*2 + 0][current_rank] = 1
        V[(j_off+1)*2 + 0][current_rank] = 1
        V[(j_off+1)*2 + 1][current_rank] = -1
        W[(k_off+0)*2 + 0][current_rank] = 1
        current_rank += 1
        
        # Term 3: M4
        U[(i_off+1)*2 + 1][current_rank] = 1
        V[(j_off+0)*2 + 1][current_rank] = 1
        V[(j_off+0)*2 + 0][current_rank] = -1
        W[(k_off+0)*2 + 1][current_rank] = 1
        current_rank += 1
        
        # Term 4: M5
        U[(i_off+0)*2 + 0][current_rank] = 1
        U[(i_off+0)*2 + 1][current_rank] = 1
        V[(j_off+1)*2 + 1][current_rank] = 1
        W[(k_off+0)*2 + 0][current_rank] = -1
        W[(k_off+1)*2 + 1][current_rank] = 1
        current_rank += 1
        
        # Term 5: M6
        U[(i_off+1)*2 + 0][current_rank] = 1
        U[(i_off+0)*2 + 0][current_rank] = -1
        V[(j_off+0)*2 + 0][current_rank] = 1
        V[(j_off+0)*2 + 1][current_rank] = 1
        W[(k_off+1)*2 + 0][current_rank] = 1
        current_rank += 1
        
        # Term 6: M7
        U[(i_off+0)*2 + 1][current_rank] = 1
        U[(i_off+1)*2 + 1][current_rank] = -1
        V[(j_off+1)*2 + 0][current_rank] = 1
        V[(j_off+1)*2 + 1][current_rank] = 1
        W[(k_off+0)*2 + 0][current_rank] = 1
        current_rank += 1

    # Helper to add a 1x2x2 Schoolbook block
    # local i (0), j (0..1), k (0..1)
    def add_schoolbook_1x2x2(i_off, j_off, k_off):
        nonlocal current_rank
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        # 4 terms
        for j in range(2):
            for k in range(2):
                U[(i_off+0)*2 + j][current_rank] = 1
                V[(j_off+j)*2 + k][current_rank] = 1
                W[(k_off+k)*2 + (i_off+0)][current_rank] = 1
                current_rank += 1

    # Process Top Part (2x4x8)
    # Split into Left (cols 0-3) and Right (cols 4-7)
    for half in range(2):
        k_base = half * 4
        # 2x4x4 block at k_base
        # Split 4x4 B into 4 2x2 blocks
        # Split 2x4 A into 2 2x2 blocks (Left, Right)
        
        # A_L (cols 0-1 of A) with B_TL (rows 0-1, cols 0-1 of B_half)
        add_strassen_2x2x2(i_off=0, j_off=0, k_off=k_base + 0)
        
        # A_R (cols 2-3 of A) with B_BL (rows 2-3, cols 0-1 of B_half)
        # Note: In 2x4x4, A is 2x4. Cols 0,1 are A_L. Cols 2,3 are A_R.
        # B is 4x4. Rows 0,1 are Top. Rows 2,3 are Bottom.
        # C_L = A_L B_TL + A_R B_BL.
        # C_R = A_L B_TR + A_R B_BR.
        
        add_strassen_2x2x2(i_off=0, j_off=2, k_off=k_base + 0) # A_R (j_off=2 in A? No, j is col of A)
        # Wait, j_off in add_strassen refers to the row of B / col of A.
        # A is 2x4. j is 0..3.
        # A_L uses j=0,1. A_R uses j=2,3.
        # So for A_R, j_off should be 2?
        # The function add_strassen_2x2x2(i_off, j_off, k_off) assumes local 2x2.
        # i_off is row of A (0..1).
        # j_off is col of A / row of B (0..3).
        # k_off is col of B (0..7).
        
        # Let's re-verify the mapping.
        # For A_L (j=0,1) and B_TL (j=0,1, k=0,1 relative to half):
        # i_off=0, j_off=0, k_off=k_base.
        
        # For A_R (j=2,3) and B_BL (j=2,3, k=0,1 relative to half):
        # i_off=0, j_off=2, k_off=k_base.
        
        # For A_L (j=0,1) and B_TR (j=0,1, k=2,3 relative to half):
        # i_off=0, j_off=0, k_off=k_base + 2.
        
        # For A_R (j=2,3) and B_BR (j=2,3, k=2,3 relative to half):
        # i_off=0, j_off=2, k_off=k_base + 2.

    # Process Bottom Part (1x4x8)
    for half in range(2):
        k_base = half * 4
        # 1x4x4 block at k_base
        # A is 1x4. j=0..3.
        # B is 4x4.
        # C_L = A_L B_TL + A_R B_BL.
        # C_R = A_L B_TR + A_R B_BR.
        
        # A_L (j=0,1) with B_TL (j=0,1, k=0,1)
        add_schoolbook_1x2x2(i_off=2, j_off=0, k_off=k_base + 0)
        
        # A_R (j=2,3) with B_BL (j=2,3, k=0,1)
        add_schoolbook_1x2x2(i_off=2, j_off=2, k_off=k_base + 0)
        
        # A_L (j=0,1) with B_TR (j=0,1, k=2,3)
        add_schoolbook_1x2x2(i_off=2, j_off=0, k_off=k_base + 2)
        
        # A_R (j=2,3) with B_BR (j=2,3, k=2,3)
        add_schoolbook_1x2x2(i_off=2, j_off=2, k_off=k_base + 2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
