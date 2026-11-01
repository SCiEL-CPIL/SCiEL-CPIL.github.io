"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2_block(col_offset):
    """
    Return the 7 Strassen terms for multiplying a 2x2 matrix A by a 2x2 block of B.
    A = [[A00, A01], [A10, A11]]  (indices in U: 0,1,2,3)
    B_block = [[B00, B01], [B10, B11]]  (indices in V: col_offset*2 .. col_offset*2+3)
    C_block = [[C00, C01], [C10, C11]]  (indices in W: (col_offset*2)*2+0 .. etc)
    
    W index: W[k*n + i] where k is column (0..11), i is row (0..1)
    So for column k, row i: index = k*2 + i
    """
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B block indices (within V): base = col_offset * 2
    # B00 = base+0, B01 = base+1, B10 = base+2, B11 = base+3
    # C block indices (within W): 
    #   C00: col=col_offset*2, row=0 -> index = (col_offset*2)*2 + 0
    #   C01: col=col_offset*2+1, row=0 -> index = (col_offset*2+1)*2 + 0
    #   C10: col=col_offset*2, row=1 -> index = (col_offset*2)*2 + 1
    #   C11: col=col_offset*2+1, row=1 -> index = (col_offset*2+1)*2 + 1
    
    base = col_offset * 2
    c00 = base * 2 + 0  # col=base, row=0
    c01 = (base + 1) * 2 + 0  # col=base+1, row=0
    c10 = base * 2 + 1  # col=base, row=1
    c11 = (base + 1) * 2 + 1  # col=base+1, row=1
    
    # Each term is a tuple (u_indices, v_indices, w_indices, coefficients)
    # For Strassen, each product involves sums of A and B entries.
    # We need to express each Mj as a sum of products of linear combinations.
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    terms = []
    
    # M1 = (A00 + A11)(B00 + B11)
    terms.append({
        'U': [(0, 1), (3, 1)],  # A00 + A11
        'V': [(base + 0, 1), (base + 3, 1)],  # B00 + B11
        'W': [(c00, 1), (c01, 1), (c10, 1), (c11, 1)],  # C00 + C01 - C10 + C11... wait
    })
    
    # This approach is getting complex. Let me use a different formulation.
    # Each of the 7 products is a single multiplication of two linear combinations.
    # I'll represent each term as a triple of (U_vec, V_vec, W_vec) where each is a dict.
    
    return terms

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition: 6 blocks of 2 columns, each with rank 7
    # Total rank = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(6):
        # B block columns: block*2, block*2+1
        # V indices for this block: (block*2)*2 .. (block*2+1)*2+1 = block*4 .. block*4+3
        # Wait, V is indexed by (j*p + k) where j is row (0,1) and k is column (0..11)
        # So V[j*12 + k]
        # For block b, columns are 2b and 2b+1
        # V[0*12 + 2b] = B0, V[0*12 + 2b+1] = B1, V[1*12 + 2b] = B2, V[1*12 + 2b+1] = B3
        
        v_b00 = 0 * 12 + 2 * block      # row 0, col 2b
        v_b01 = 0 * 12 + 2 * block + 1  # row 0, col 2b+1
        v_b10 = 1 * 12 + 2 * block      # row 1, col 2b
        v_b11 = 1 * 12 + 2 * block + 1  # row 1, col 2b+1
        
        # W indices: W[k*n + i] where k is output column, i is output row
        # Output columns for this block: 2b, 2b+1
        w_c00 = 2 * block * 2 + 0      # col 2b, row 0
        w_c01 = (2 * block + 1) * 2 + 0  # col 2b+1, row 0
        w_c10 = 2 * block * 2 + 1      # col 2b, row 1
        w_c11 = (2 * block + 1) * 2 + 1  # col 2b+1, row 1
        
        # Strassen's 7 products:
        # M1 = (A00 + A11)(B00 + B11) -> contributes to C00, C01, C10, C11
        # M2 = (A10 + A11)B00 -> contributes to C10, C11
        # M3 = A00(B01 - B11) -> contributes to C01, C11
        # M4 = A11(B10 - B00) -> contributes to C00, C10
        # M5 = (A00 + A01)B11 -> contributes to C00, C01
        # M6 = (A10 - A00)(B00 + B01) -> contributes to C11
        # M7 = (A01 - A11)(B10 + B11) -> contributes to C00
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0][col] += 1  # A00
        U[3][col] += 1  # A11
        V[v_b00][col] += 1  # B00
        V[v_b11][col] += 1  # B11
        W[w_c00][col] += 1
        W[w_c01][col] += 1
        W[w_c10][col] += 1
        W[w_c11][col] -= 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[2][col] += 1  # A10
        U[3][col] += 1  # A11
        V[v_b00][col] += 1  # B00
        W[w_c10][col] += 1
        W[w_c11][col] -= 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[0][col] += 1  # A00
        V[v_b01][col] += 1  # B01
        V[v_b11][col] -= 1  # -B11
        W[w_c01][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[3][col] += 1  # A11
        V[v_b10][col] += 1  # B10
        V[v_b00][col] -= 1  # -B00
        W[w_c00][col] += 1
        W[w_c10][col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[0][col] += 1  # A00
        U[1][col] += 1  # A01
        V[v_b11][col] += 1  # B11
        W[w_c00][col] -= 1
        W[w_c01][col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2][col] += 1  # A10
        U[0][col] -= 1  # -A00
        V[v_b00][col] += 1  # B00
        V[v_b01][col] += 1  # B01
        W[w_c11][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1][col] += 1  # A01
        U[3][col] -= 1  # -A11
        V[v_b10][col] += 1  # B10
        V[v_b11][col] += 1  # B11
        W[w_c00][col] += 1
        col += 1
    
    # Verify: col should be 42
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Check deadline
    if time.time() > deadline:
        # Return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
