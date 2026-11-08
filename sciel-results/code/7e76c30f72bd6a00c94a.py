"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val, col):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # B indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # C indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    add_term(0, 0, 0, 1, 0) # A00 B00 C00
    add_term(0, 4, 0, 1, 0) # A00 B11 C00
    add_term(4, 0, 3, 1, 0) # A11 B00 C11
    add_term(4, 4, 3, 1, 0) # A11 B11 C11
    
    # M2 = (A10+A11)B00 -> C10, C11(-)
    add_term(3, 0, 2, 1, 1) # A10 B00 C10
    add_term(4, 0, 2, 1, 1) # A11 B00 C10
    add_term(3, 0, 3, -1, 1) # A10 B00 C11
    add_term(4, 0, 3, -1, 1) # A11 B00 C11
    
    # M3 = A00(B01-B11) -> C01, C11
    add_term(0, 1, 1, 1, 2) # A00 B01 C01
    add_term(0, 4, 1, -1, 2) # A00 B11 C01
    add_term(0, 1, 3, 1, 2) # A00 B01 C11
    add_term(0, 4, 3, -1, 2) # A00 B11 C11
    
    # M4 = A11(B10-B00) -> C00, C10
    add_term(4, 3, 0, 1, 3) # A11 B10 C00
    add_term(4, 0, 0, -1, 3) # A11 B00 C00
    add_term(4, 3, 2, 1, 3) # A11 B10 C10
    add_term(4, 0, 2, -1, 3) # A11 B00 C10
    
    # M5 = (A00+A01)B11 -> C00(-), C01
    add_term(0, 4, 0, -1, 4) # A00 B11 C00
    add_term(1, 4, 0, -1, 4) # A01 B11 C00
    add_term(0, 4, 1, 1, 4) # A00 B11 C01
    add_term(1, 4, 1, 1, 4) # A01 B11 C01
    
    # M6 = (A10-A00)(B00+B01) -> C11
    add_term(3, 0, 3, 1, 5) # A10 B00 C11
    add_term(0, 0, 3, -1, 5) # A00 B00 C11
    add_term(3, 1, 3, 1, 5) # A10 B01 C11
    add_term(0, 1, 3, -1, 5) # A00 B01 C11
    
    # M7 = (A01-A11)(B10+B11) -> C00
    add_term(1, 3, 0, 1, 6) # A01 B10 C00
    add_term(4, 3, 0, -1, 6) # A11 B10 C00
    add_term(1, 4, 0, 1, 6) # A01 B11 C00
    add_term(4, 4, 0, -1, 6) # A11 B11 C00
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # A indices: 0, 1, 3, 4
    # B indices: 2(0,2), 5(1,2)
    # C indices: 4(0,2), 5(1,2)
    
    # N1 = A00 B02 -> C02
    add_term(0, 2, 4, 1, 7)
    
    # N2 = A01 B12 -> C02
    add_term(1, 5, 4, 1, 8)
    
    # N3 = A10 B02 -> C12
    add_term(3, 2, 5, 1, 9)
    
    # N4 = A11 B12 -> C12
    add_term(4, 5, 5, 1, 10)
    
    # Block 3: A[0:2, 2] * B[2, 0:3] -> C[0:2, 0:3]
    # A indices: 2(0,2), 5(1,2)
    # B indices: 6(2,0), 7(2,1), 8(2,2)
    # C indices: 0, 1, 4 (for i=0), 2, 3, 5 (for i=1)
    
    # P1 = A02 B20 -> C00
    add_term(2, 6, 0, 1, 11)
    
    # P2 = A02 B21 -> C01
    add_term(2, 7, 1, 1, 12)
    
    # P3 = A02 B22 -> C02
    add_term(2, 8, 4, 1, 13)
    
    # P4 = A12 B20 -> C10
    add_term(5, 6, 2, 1, 14)
    
    # P5 = A12 B21 -> C11
    add_term(5, 7, 3, 1, 15)
    
    # P6 = A12 B22 -> C12
    add_term(5, 8, 5, 1, 16)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
