"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # Rank 36 decomposition
    
    # U shape: (n*m, rank) = (8, 36)
    # V shape: (m*p, rank) = (20, 36)
    # W shape: (p*n, rank) = (10, 36)
    
    rank = 36
    U = [[Fraction(0) for _ in range(rank)] for _ in range(n*m)]
    V = [[Fraction(0) for _ in range(rank)] for _ in range(m*p)]
    W = [[Fraction(0) for _ in range(rank)] for _ in range(p*n)]
    
    def add_strassen_block(a_idx, b_idx, c_idx, start_rank):
        # a_idx: list of 4 indices into U (0,1,4,5 style)
        # b_idx: list of 4 indices into V
        # c_idx: list of 4 indices into W
        # Adds 7 rank-1 terms
        
        # Strassen M1..M7
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # a_idx = [a00, a01, a10, a11]
        # b_idx = [b00, b01, b10, b11]
        # c_idx = [c00, c01, c10, c11]
        
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        
        # M1
        U[a00][start_rank] += Fraction(1)
        U[a11][start_rank] += Fraction(1)
        V[b00][start_rank] += Fraction(1)
        V[b11][start_rank] += Fraction(1)
        W[c00][start_rank] += Fraction(1)
        W[c11][start_rank] += Fraction(1)
        
        # M2
        U[a10][start_rank+1] += Fraction(1)
        U[a11][start_rank+1] += Fraction(1)
        V[b00][start_rank+1] += Fraction(1)
        W[c10][start_rank+1] += Fraction(1)
        
        # M3
        U[a00][start_rank+2] += Fraction(1)
        V[b01][start_rank+2] += Fraction(1)
        V[b11][start_rank+2] -= Fraction(1)
        W[c01][start_rank+2] += Fraction(1)
        
        # M4
        U[a11][start_rank+3] += Fraction(1)
        V[b10][start_rank+3] += Fraction(1)
        V[b00][start_rank+3] -= Fraction(1)
        W[c00][start_rank+3] += Fraction(1)
        W[c10][start_rank+3] += Fraction(1)
        
        # M5
        U[a00][start_rank+4] += Fraction(1)
        U[a01][start_rank+4] += Fraction(1)
        V[b11][start_rank+4] += Fraction(1)
        W[c00][start_rank+4] -= Fraction(1)
        W[c01][start_rank+4] += Fraction(1)
        
        # M6
        U[a10][start_rank+5] += Fraction(1)
        U[a00][start_rank+5] -= Fraction(1)
        V[b00][start_rank+5] += Fraction(1)
        V[b01][start_rank+5] += Fraction(1)
        W[c11][start_rank+5] += Fraction(1)
        
        # M7
        U[a01][start_rank+6] += Fraction(1)
        U[a11][start_rank+6] -= Fraction(1)
        V[b10][start_rank+6] += Fraction(1)
        V[b11][start_rank+6] += Fraction(1)
        W[c00][start_rank+6] += Fraction(1)
        W[c11][start_rank+6] -= Fraction(1)
        
        return start_rank + 7

    def add_standard_block(a_idx, b_idx, c_idx, start_rank):
        # a_idx: [a00, a01, a10, a11] (2x2 A)
        # b_idx: [b00, b10] (2x1 B) -> actually B is 2x1, so entries are b00, b10? 
        # Wait, B is 2x1. Indices in B matrix:
        # B[0,0], B[1,0].
        # In our global V, B is 4x5.
        # For a 2x2 A and 2x1 B block:
        # A is 2x2. B is 2x1.
        # C is 2x1.
        # C0 = a00*b00 + a01*b10
        # C1 = a10*b00 + a11*b10
        # b_idx should be [b00, b10] corresponding to the two rows of B for the single column.
        # c_idx should be [c00, c10] corresponding to the two rows of C for the single column.
        
        a00, a01, a10, a11 = a_idx
        b00, b10 = b_idx
        c00, c10 = c_idx
        
        # Term 1: a00 * b00 -> c00
        U[a00][start_rank] += Fraction(1)
        V[b00][start_rank] += Fraction(1)
        W[c00][start_rank] += Fraction(1)
        
        # Term 2: a01 * b10 -> c00
        U[a01][start_rank+1] += Fraction(1)
        V[b10][start_rank+1] += Fraction(1)
        W[c00][start_rank+1] += Fraction(1)
        
        # Term 3: a10 * b00 -> c10
        U[a10][start_rank+2] += Fraction(1)
        V[b00][start_rank+2] += Fraction(1)
        W[c10][start_rank+2] += Fraction(1)
        
        # Term 4: a11 * b10 -> c10
        U[a11][start_rank+3] += Fraction(1)
        V[b10][start_rank+3] += Fraction(1)
        W[c10][start_rank+3] += Fraction(1)
        
        return start_rank + 4

    # Indices for A (2x4)
    # A[i, j] -> U index i*4 + j
    # A1: cols 0,1. A2: cols 2,3.
    # A1 indices:
    # a00: 0*4+0=0
    # a01: 0*4+1=1
    # a10: 1*4+0=4
    # a11: 1*4+1=5
    a1_idx = [0, 1, 4, 5]
    # A2 indices:
    # a00: 0*4+2=2
    # a01: 0*4+3=3
    # a10: 1*4+2=6
    # a11: 1*4+3=7
    a2_idx = [2, 3, 6, 7]
    
    # Indices for B (4x5)
    # B[j, k] -> V index j*5 + k
    # B1: rows 0,1. B2: rows 2,3.
    # B1, col 0,1 (2x2):
    # b00: 0*5+0=0
    # b01: 0*5+1=1
    # b10: 1*5+0=5
    # b11: 1*5+1=6
    b1a_idx = [0, 1, 5, 6]
    
    # B1, col 2,3 (2x2):
    # b00: 0*5+2=2
    # b01: 0*5+3=3
    # b10: 1*5+2=7
    # b11: 1*5+3=8
    b1b_idx = [2, 3, 7, 8]
    
    # B1, col 4 (2x1):
    # b00: 0*5+4=4
    # b10: 1*5+4=9
    b1c_idx = [4, 9]
    
    # B2, col 0,1 (2x2):
    # b00: 2*5+0=10
    # b01: 2*5+1=11
    # b10: 3*5+0=15
    # b11: 3*5+1=16
    b2a_idx = [10, 11, 15, 16]
    
    # B2, col 2,3 (2x2):
    # b00: 2*5+2=12
    # b01: 2*5+3=13
    # b10: 3*5+2=17
    # b11: 3*5+3=18
    b2b_idx = [12, 13, 17, 18]
    
    # B2, col 4 (2x1):
    # b00: 2*5+4=14
    # b10: 3*5+4=19
    b2c_idx = [14, 19]
    
    # Indices for C (2x5)
    # C[k, i] -> W index k*2 + i
    # Note: Output factor follows W[k*n+i, r]. n=2.
    # C is 2x5. Rows i=0,1. Cols k=0..4.
    # Wait, the tensor is T[a, b, c] where c is the output index.
    # Output matrix C is 2x5.
    # Standard matrix multiplication C[i, k] = sum A[i, j] B[j, k].
    # The task says: "output factor follows the official trace convention W[k*n+i, r]".
    # This implies the output index c corresponds to `k*n + i`.
    # Here n=2 (rows of A, cols of C? No, A is n x m, B is m x p, C is n x p).
    # So C has n rows and p cols.
    # Index c = k*n + i.
    # So for C[i, k], the index is k*2 + i.
    
    # C1a (cols 0,1 of C, from A1 B1a):
    # C[0,0] -> 0*2+0=0
    # C[0,1] -> 1*2+0=2  (Wait, k=1, i=0 -> 1*2+0=2)
    # C[1,0] -> 0*2+1=1
    # C[1,1] -> 1*2+1=3
    # So c_idx for cols 0,1:
    # c00 (i=0,k=0): 0
    # c01 (i=0,k=1): 2
    # c10 (i=1,k=0): 1
    # c11 (i=1,k=1): 3
    c1a_idx = [0, 2, 1, 3]
    
    # C1b (cols 2,3 of C, from A1 B1b):
    # c00 (i=0,k=2): 2*2+0=4
    # c01 (i=0,k=3): 3*2+0=6
    # c10 (i=1,k=2): 2*2+1=5
    # c11 (i=1,k=3): 3*2+1=7
    c1b_idx = [4, 6, 5, 7]
    
    # C1c (col 4 of C, from A1 B1c):
    # c00 (i=0,k=4): 4*2+0=8
    # c10 (i=1,k=4): 4*2+1=9
    c1c_idx = [8, 9]
    
    # C2a (cols 0,1 of C, from A2 B2a):
    # Same C indices as C1a.
    c2a_idx = [0, 2, 1, 3]
    
    # C2b (cols 2,3 of C, from A2 B2b):
    # Same C indices as C1b.
    c2b_idx = [4, 6, 5, 7]
    
    # C2c (col 4 of C, from A2 B2c):
    # Same C indices as C1c.
    c2c_idx = [8, 9]
    
    # Construct
    r = 0
    r = add_strassen_block(a1_idx, b1a_idx, c1a_idx, r)
    r = add_strassen_block(a1_idx, b1b_idx, c1b_idx, r)
    r = add_standard_block(a1_idx, b1c_idx, c1c_idx, r)
    
    r = add_strassen_block(a2_idx, b2a_idx, c2a_idx, r)
    r = add_strassen_block(a2_idx, b2b_idx, c2b_idx, r)
    r = add_standard_block(a2_idx, b2c_idx, c2c_idx, r)
    
    # r should be 36
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
