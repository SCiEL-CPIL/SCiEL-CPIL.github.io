"""Program for matrix multiplication 2x4 x 4x6 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Uses Strassen's algorithm on 2x2 blocks to reduce rank from 48 (schoolbook)
    to 42 (6 blocks of rank 7).
    """
    n, m, p = DIMENSIONS
    rank = 42  # 6 blocks * 7 Strassen terms
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 42)
    
    # Strassen coefficients for 2x2 block multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 - M5
    # C11 = M1 - M2 + M3 + M4
    
    # For each 2x2 block (i_block, j_block, k_block) where:
    # i_block in {0, 1} (rows of A, cols of C)
    # j_block in {0, 1} (cols of A, rows of B)  
    # k_block in {0, 1, 2} (cols of B, cols of C)
    
    # We have 2 (i) * 2 (j) * 3 (k) = 12 blocks of size 2x2x2
    # Wait, let me reconsider the block structure.
    
    # A is 2x4, B is 4x6, C is 2x6
    # Let's partition:
    # A: rows 0-1, cols 0-3 -> A = [A00 A01; A10 A11] where each block is 2x2
    # B: rows 0-3, cols 0-5 -> B = [B00 B01 B02; B10 B11 B12] where each block is 2x2
    # C: rows 0-1, cols 0-5 -> C = [C00 C01 C02; C10 C11 C12] where each block is 2x2
    
    # So we have 2 (i) * 2 (j) * 3 (k) = 12 blocks of 2x2x2
    # Each 2x2x2 block uses Strassen with rank 7
    # Total rank = 12 * 7 = 84? That's worse than schoolbook!
    
    # Let me reconsider. The schoolbook for 2x4 * 4x6 is 48.
    # If I use 2x2 blocks: A has 1x2 blocks, B has 2x3 blocks, C has 1x3 blocks
    # That's 1*2*3 = 6 blocks of 2x2x2, each with rank 7
    # Total rank = 6 * 7 = 42. Yes, that's correct.
    
    # Block indices: i in {0}, j in {0,1}, k in {0,1,2}
    # But A is 2x4, so rows 0-1, cols 0-3. If we use 2x2 blocks:
    # A = [A00 A01] where A00 is rows 0-1, cols 0-1; A01 is rows 0-1, cols 2-3
    # B = [B00 B01 B02; B10 B11 B12] where each is 2x2
    # C = [C00 C01 C02] where each is 2x2
    
    # So we have 1 (i) * 2 (j) * 3 (k) = 6 blocks
    # Each block (i=0, j, k) computes C_block[0,j,k] = A_block[0,j] * B_block[j,k]
    
    # For each block (j, k) where j in {0,1}, k in {0,1,2}:
    # A_block is 2x2: rows 0-1, cols 2*j to 2*j+1
    # B_block is 2x2: rows 2*j to 2*j+1, cols 2*k to 2*k+1
    # C_block is 2x2: rows 0-1, cols 2*k to 2*k+1
    
    # Strassen for this block:
    # Let A_b = [[a00, a01], [a10, a11]]
    # Let B_b = [[b00, b01], [b10, b11]]
    # Let C_b = [[c00, c01], [c10, c11]]
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # c00 = M1 - M4 + M6 - M7
    # c01 = M3 + M5
    # c10 = M2 - M5
    # c11 = M1 - M2 + M3 + M4
    
    # Now map to global indices:
    # For block (j, k):
    # A row i (0 or 1), col 2*j + j_local (0 or 1)
    # B row 2*j + j_local (0 or 1), col 2*k + k_local (0 or 1)
    # C row i (0 or 1), col 2*k + k_local (0 or 1)
    
    # U index: i * m + col_A = i * 4 + (2*j + j_local)
    # V index: row_B * p + col_B = (2*j + j_local) * 6 + (2*k + k_local)
    # W index: col_C * n + row_C = (2*k + k_local) * 2 + i
    
    # Strassen coefficients for each M:
    # M1: A: a00+a11, B: b00+b11
    # M2: A: a10+a11, B: b00
    # M3: A: a00, B: b10-b11
    # M4: A: a11, B: b01-b00
    # M5: A: a00+a01, B: b11
    # M6: A: a10-a00, B: b00+b01
    # M7: A: a01-a11, B: b10+b11
    
    # C coefficients:
    # c00: M1: +1, M4: -1, M6: +1, M7: -1
    # c01: M3: +1, M5: +1
    # c10: M2: +1, M5: -1
    # c11: M1: +1, M2: -1, M3: +1, M4: +1
    
    col = 0
    for j in range(2):  # j_block
        for k in range(3):  # k_block
            for m_idx in range(7):  # Strassen term
                # Determine A coefficients
                a_coeffs = {}  # (i, j_local) -> coeff
                # Determine B coefficients  
                b_coeffs = {}  # (j_local, k_local) -> coeff
                
                if m_idx == 0:  # M1
                    a_coeffs = {(0,0): 1, (1,1): 1}
                    b_coeffs = {(0,0): 1, (1,1): 1}
                elif m_idx == 1:  # M2
                    a_coeffs = {(1,0): 1, (1,1): 1}
                    b_coeffs = {(0,0): 1}
                elif m_idx == 2:  # M3
                    a_coeffs = {(0,0): 1}
                    b_coeffs = {(1,0): 1, (1,1): -1}
                elif m_idx == 3:  # M4
                    a_coeffs = {(1,1): 1}
                    b_coeffs = {(0,1): 1, (0,0): -1}
                elif m_idx == 4:  # M5
                    a_coeffs = {(0,0): 1, (0,1): 1}
                    b_coeffs = {(1,1): 1}
                elif m_idx == 5:  # M6
                    a_coeffs = {(1,0): 1, (0,0): -1}
                    b_coeffs = {(0,0): 1, (0,1): 1}
                elif m_idx == 6:  # M7
                    a_coeffs = {(0,1): 1, (1,1): -1}
                    b_coeffs = {(1,0): 1, (1,1): 1}
                
                # Fill U
                for (i, j_local), coeff in a_coeffs.items():
                    u_idx = i * m + (2*j + j_local)
                    U[u_idx][col] += coeff
                
                # Fill V
                for (j_local, k_local), coeff in b_coeffs.items():
                    v_idx = (2*j + j_local) * p + (2*k + k_local)
                    V[v_idx][col] += coeff
                
                # Fill W
                # c00: M1: +1, M4: -1, M6: +1, M7: -1
                # c01: M3: +1, M5: +1
                # c10: M2: +1, M5: -1
                # c11: M1: +1, M2: -1, M3: +1, M4: +1
                c_coeffs = {}  # (i, k_local) -> coeff
                if m_idx == 0:  # M1
                    c_coeffs = {(0,0): 1, (1,1): 1}
                elif m_idx == 1:  # M2
                    c_coeffs = {(1,0): 1, (1,1): -1}
                elif m_idx == 2:  # M3
                    c_coeffs = {(0,1): 1, (1,1): 1}
                elif m_idx == 3:  # M4
                    c_coeffs = {(0,0): -1, (1,1): 1}
                elif m_idx == 4:  # M5
                    c_coeffs = {(0,1): 1, (1,0): -1}
                elif m_idx == 5:  # M6
                    c_coeffs = {(0,0): 1}
                elif m_idx == 6:  # M7
                    c_coeffs = {(0,0): -1}
                
                for (i, k_local), coeff in c_coeffs.items():
                    w_idx = (2*k + k_local) * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
