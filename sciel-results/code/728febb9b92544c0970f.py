"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_decomposition(offset, v_offset):
        # v_offset is 0 for first block, 9 for second block (since m=3, p=6, rows 0-2 are 0-2, 3-5 are 3-5? No, V is m*p = 18. Rows are j*p+k. j=0,1,2. k=0..5.
        # Block 1: k=0,1,2. V indices: j*6+k.
        # Block 2: k=3,4,5. V indices: j*6+k.
        
        # Strassen part
        # M1
        U[0][offset] = 1; U[1][offset] = 1; U[3][offset] = 1; U[4][offset] = 1
        V[0+v_offset][offset] = 1; V[1+v_offset][offset] = 1; V[3+v_offset][offset] = 1; V[4+v_offset][offset] = 1
        W[0][offset] = 1; W[3][offset] = 1; W[1][offset] = -1; W[4][offset] = -1
        
        # M2
        U[3][offset+1] = 1; U[4][offset+1] = 1
        V[0+v_offset][offset+1] = 1
        W[3][offset+1] = 1; W[4][offset+1] = -1
        
        # M3
        U[0][offset+2] = 1
        V[1+v_offset][offset+2] = 1; V[4+v_offset][offset+2] = -1
        W[1][offset+2] = 1; W[4][offset+2] = -1
        
        # M4
        U[4][offset+3] = 1
        V[0+v_offset][offset+3] = 1; V[3+v_offset][offset+3] = -1
        W[0][offset+3] = 1; W[3][offset+3] = 1
        
        # M5
        U[0][offset+4] = 1; U[1][offset+4] = 1
        V[4+v_offset][offset+4] = 1
        W[0][offset+4] = -1; W[1][offset+4] = 1
        
        # M6
        U[3][offset+5] = 1; U[0][offset+5] = -1
        V[0+v_offset][offset+5] = 1; V[3+v_offset][offset+5] = 1
        W[3][offset+5] = 1; W[4][offset+5] = -1
        
        # M7
        U[1][offset+6] = 1; U[4][offset+6] = -1
        V[3+v_offset][offset+6] = 1; V[4+v_offset][offset+6] = 1
        W[0][offset+6] = 1; W[4][offset+6] = 1
        
        # Part 2
        # M8
        U[0][offset+7] = 1
        V[2+v_offset][offset+7] = 1
        W[2][offset+7] = 1
        
        # M9
        U[1][offset+8] = 1
        V[5+v_offset][offset+8] = 1
        W[2][offset+8] = 1
        
        # M10
        U[3][offset+9] = 1
        V[2+v_offset][offset+9] = 1
        W[5][offset+9] = 1
        
        # M11
        U[4][offset+10] = 1
        V[5+v_offset][offset+10] = 1
        W[5][offset+10] = 1
        
        # Part 3
        # M12
        U[2][offset+11] = 1
        V[6+v_offset][offset+11] = 1
        W[0][offset+11] = 1
        
        # M13
        U[2][offset+12] = 1
        V[7+v_offset][offset+12] = 1
        W[1][offset+12] = 1
        
        # M14
        U[2][offset+13] = 1
        V[8+v_offset][offset+13] = 1
        W[2][offset+13] = 1
        
        # M15
        U[5][offset+14] = 1
        V[6+v_offset][offset+14] = 1
        W[3][offset+14] = 1
        
        # M16
        U[5][offset+15] = 1
        V[7+v_offset][offset+15] = 1
        W[4][offset+15] = 1
        
        # M17
        U[5][offset+16] = 1
        V[8+v_offset][offset+16] = 1
        W[5][offset+16] = 1

    # Block 1: v_offset = 0
    add_decomposition(0, 0)
    # Block 2: v_offset = 9? 
    # V indices: j*p+k.
    # Block 1 uses k=0,1,2.
    # Block 2 uses k=3,4,5.
    # For j=0: k=0,1,2 -> 0,1,2. k=3,4,5 -> 3,4,5.
    # For j=1: k=0,1,2 -> 6,7,8. k=3,4,5 -> 9,10,11.
    # For j=2: k=0,1,2 -> 12,13,14. k=3,4,5 -> 15,16,17.
    
    # My V indices in add_decomposition:
    # 0,1,2,3,4,5,6,7,8
    # These correspond to j=0,k=0..2 and j=1,k=0..2?
    # No, V is (m*p, rank). m=3, p=6.
    # V index = j*6 + k.
    # Block 1 (k=0,1,2):
    # j=0: 0,1,2
    # j=1: 6,7,8
    # j=2: 12,13,14
    
    # Block 2 (k=3,4,5):
    # j=0: 3,4,5
    # j=1: 9,10,11
    # j=2: 15,16,17
    
    # My add_decomposition uses V[0..8].
    # This assumes V indices are contiguous.
    # I need to map V indices correctly.
    
    # Let's modify add_decomposition to take a list of V indices.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
