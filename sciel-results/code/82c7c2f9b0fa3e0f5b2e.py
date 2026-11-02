"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition using Strassen for 2x2 blocks and schoolbook for the last column
    rank = 18
    
    # Initialize U, V, W
    # U: (n*m, rank) = (4, 18)
    # V: (m*p, rank) = (10, 18)
    # W: (p*n, rank) = (10, 18)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    def add_strassen(start_col, b_idx_00, b_idx_01, b_idx_10, b_idx_11, c_idx_00, c_idx_01, c_idx_10, c_idx_11):
        # Strassen intermediates:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # U: A indices 0,1,2,3 correspond to A00, A01, A10, A11
        # V: B indices
        # W: C indices
        
        # Col 0: M1
        U[0][start_col] = 1
        U[3][start_col] = 1
        V[b_idx_00][start_col] = 1
        V[b_idx_11][start_col] = 1
        W[c_idx_00][start_col] = 1
        W[c_idx_11][start_col] = 1
        
        # Col 1: M2
        U[2][start_col+1] = 1
        U[3][start_col+1] = 1
        V[b_idx_00][start_col+1] = 1
        W[c_idx_10][start_col+1] = 1
        W[c_idx_11][start_col+1] = -1
        
        # Col 2: M3
        U[0][start_col+2] = 1
        V[b_idx_01][start_col+2] = 1
        V[b_idx_11][start_col+2] = -1
        W[c_idx_01][start_col+2] = 1
        W[c_idx_11][start_col+2] = 1
        
        # Col 3: M4
        U[3][start_col+3] = 1
        V[b_idx_10][start_col+3] = 1
        V[b_idx_00][start_col+3] = -1
        W[c_idx_00][start_col+3] = 1
        W[c_idx_10][start_col+3] = 1
        
        # Col 4: M5
        U[0][start_col+4] = 1
        U[1][start_col+4] = 1
        V[b_idx_11][start_col+4] = 1
        W[c_idx_01][start_col+4] = 1
        W[c_idx_00][start_col+4] = -1
        
        # Col 5: M6
        U[2][start_col+5] = 1
        U[0][start_col+5] = -1
        V[b_idx_00][start_col+5] = 1
        V[b_idx_01][start_col+5] = 1
        W[c_idx_11][start_col+5] = 1
        
        # Col 6: M7
        U[1][start_col+6] = 1
        U[3][start_col+6] = -1
        V[b_idx_10][start_col+6] = 1
        V[b_idx_11][start_col+6] = 1
        W[c_idx_00][start_col+6] = 1
        W[c_idx_01][start_col+6] = -1

    # Helper to add a schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(start_col, b_idx_00, b_idx_10, c_idx_00, c_idx_10):
        # M1 = A00 * B00
        U[0][start_col] = 1
        V[b_idx_00][start_col] = 1
        W[c_idx_00][start_col] = 1
        
        # M2 = A01 * B10
        U[1][start_col+1] = 1
        V[b_idx_10][start_col+1] = 1
        W[c_idx_00][start_col+1] = 1
        
        # M3 = A10 * B00
        U[2][start_col+2] = 1
        V[b_idx_00][start_col+2] = 1
        W[c_idx_10][start_col+2] = 1
        
        # M4 = A11 * B10
        U[3][start_col+3] = 1
        V[b_idx_10][start_col+3] = 1
        W[c_idx_10][start_col+3] = 1

    # Part 1: Columns 0, 1 of B
    # B indices: j*p + k
    # B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
    # C indices: k*n + i
    # C[0,0]=0, C[0,1]=1, C[1,0]=5, C[1,1]=6
    add_strassen(0, 0, 1, 5, 6, 0, 1, 5, 6)
    
    # Part 2: Columns 2, 3 of B
    # B[0,2]=2, B[0,3]=3, B[1,2]=7, B[1,3]=8
    # C[0,2]=2, C[0,3]=3, C[1,2]=7, C[1,3]=8
    add_strassen(7, 2, 3, 7, 8, 2, 3, 7, 8)
    
    # Part 3: Column 4 of B
    # B[0,4]=4, B[1,4]=9
    # C[0,4]=4, C[1,4]=9
    add_schoolbook_2x2x1(14, 4, 9, 4, 9)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
