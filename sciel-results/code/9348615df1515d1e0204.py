"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-44 decomposition for 3x4 * 4x4 matrix multiplication.
    
    Decomposition strategy:
    - Partition rows: top 2 rows (i=0,1), bottom 1 row (i=2)
    - Partition columns: left 2 (k=0,1), right 2 (k=2,3)
    - Partition inner dim: first 2 (j=0,1), last 2 (j=2,3)
    
    For top 2 rows: 4 sub-problems of 2x2 * 2x2, each using Strassen (rank 7)
    For bottom row: 4 sub-problems of 1x2 * 2x2, each trivial (rank 4)
    Total rank: 4*7 + 4*4 = 44
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    # Strassen's algorithm for 2x2 * 2x2 multiplication
    # Given A (2x2), B (2x2), compute C = A*B
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x2 with indices:
    # A[i,j] where i in {0,1}, j in {0,1} (local)
    # B[j,k] where j in {0,1}, k in {0,1} (local)
    # C[i,k] where i in {0,1}, k in {0,1} (local)
    
    def strassen_2x2(U, V, W, row_offset_A, col_offset_B, row_offset_C, start_col):
        """
        Add Strassen rank-7 decomposition for a 2x2*2x2 sub-problem.
        
        row_offset_A: offset for A rows in U (A[i,j] -> U[i*m+j])
        col_offset_B: offset for B cols in V (B[j,k] -> V[j*p+k])  
        row_offset_C: offset for C rows in W (C[k,n+i] -> W[k*n+i])
        start_col: starting column in the factor matrices
        """
        # Local indices for 2x2 A: rows 0,1; cols 0,1
        # Local indices for 2x2 B: rows 0,1; cols 0,1
        # Local indices for 2x2 C: rows 0,1; cols 0,1
        
        # M1 = (a+d)(e+h): A[0,0]+A[1,1] times B[0,0]+B[1,1]
        # U contribution: A[0,0] and A[1,1]
        # V contribution: B[0,0] and B[1,1]
        U[row_offset_A + 0 * 2 + 0][start_col] = 1  # A[0,0]
        U[row_offset_A + 1 * 2 + 1][start_col] = 1  # A[1,1]
        V[col_offset_B + 0 * 2 + 0][start_col] = 1  # B[0,0]
        V[col_offset_B + 1 * 2 + 1][start_col] = 1  # B[1,1]
        # W: C[0,0] += 1, C[1,1] += 1, C[0,1] -= 1, C[1,0] += 1
        # C[i,k] -> W[k*n + i], so C[0,0]->W[0*3+0]=W[0], C[0,1]->W[1*3+0]=W[3]
        # C[1,0]->W[0*3+1]=W[1], C[1,1]->W[1*3+1]=W[4]
        W[row_offset_C + 0 * 3 + 0][start_col] = 1   # C[0,0]
        W[row_offset_C + 1 * 3 + 1][start_col] = 1   # C[1,1]
        W[row_offset_C + 1 * 3 + 0][start_col] = -1  # C[0,1]
        W[row_offset_C + 0 * 3 + 1][start_col] = 1   # C[1,0]
        
        # M2 = (c+d)e: A[1,0]+A[1,1] times B[0,0]
        U[row_offset_A + 1 * 2 + 0][start_col + 1] = 1  # A[1,0]
        U[row_offset_A + 1 * 2 + 1][start_col + 1] = 1  # A[1,1]
        V[col_offset_B + 0 * 2 + 0][start_col + 1] = 1  # B[0,0]
        # W: C[1,0] += 1, C[1,1] -= 1
        W[row_offset_C + 0 * 3 + 1][start_col + 1] = 1   # C[1,0]
        W[row_offset_C + 1 * 3 + 1][start_col + 1] = -1  # C[1,1]
        
        # M3 = a(f-h): A[0,0] times B[0,1]-B[1,1]
        U[row_offset_A + 0 * 2 + 0][start_col + 2] = 1  # A[0,0]
        V[col_offset_B + 0 * 2 + 1][start_col + 2] = 1  # B[0,1]
        V[col_offset_B + 1 * 2 + 1][start_col + 2] = -1 # B[1,1]
        # W: C[0,1] += 1, C[1,1] += 1
        W[row_offset_C + 1 * 3 + 0][start_col + 2] = 1   # C[0,1]
        W[row_offset_C + 1 * 3 + 1][start_col + 2] = 1   # C[1,1]
        
        # M4 = d(g-e): A[1,1] times B[1,0]-B[0,0]
        U[row_offset_A + 1 * 2 + 1][start_col + 3] = 1  # A[1,1]
        V[col_offset_B + 1 * 2 + 0][start_col + 3] = 1  # B[1,0]
        V[col_offset_B + 0 * 2 + 0][start_col + 3] = -1 # B[0,0]
        # W: C[0,0] += 1, C[1,0] += 1
        W[row_offset_C + 0 * 3 + 0][start_col + 3] = 1   # C[0,0]
        W[row_offset_C + 0 * 3 + 1][start_col + 3] = 1   # C[1,0]
        
        # M5 = (a+b)h: A[0,0]+A[0,1] times B[1,1]
        U[row_offset_A + 0 * 2 + 0][start_col + 4] = 1  # A[0,0]
        U[row_offset_A + 0 * 2 + 1][start_col + 4] = 1  # A[0,1]
        V[col_offset_B + 1 * 2 + 1][start_col + 4] = 1  # B[1,1]
        # W: C[0,0] -= 1, C[0,1] += 1
        W[row_offset_C + 0 * 3 + 0][start_col + 4] = -1  # C[0,0]
        W[row_offset_C + 1 * 3 + 0][start_col + 4] = 1   # C[0,1]
        
        # M6 = (c-a)(e+f): A[1,0]-A[0,0] times B[0,0]+B[0,1]
        U[row_offset_A + 1 * 2 + 0][start_col + 5] = 1  # A[1,0]
        U[row_offset_A + 0 * 2 + 0][start_col + 5] = -1 # A[0,0]
        V[col_offset_B + 0 * 2 + 0][start_col + 5] = 1  # B[0,0]
        V[col_offset_B + 0 * 2 + 1][start_col + 5] = 1  # B[0,1]
        # W: C[1,1] += 1
        W[row_offset_C + 1 * 3 + 1][start_col + 5] = 1   # C[1,1]
        
        # M7 = (b-d)(g+h): A[0,1]-A[1,1] times B[1,0]+B[1,1]
        U[row_offset_A + 0 * 2 + 1][start_col + 6] = 1  # A[0,1]
        U[row_offset_A + 1 * 2 + 1][start_col + 6] = -1 # A[1,1]
        V[col_offset_B + 1 * 2 + 0][start_col + 6] = 1  # B[1,0]
        V[col_offset_B + 1 * 2 + 1][start_col + 6] = 1  # B[1,1]
        # W: C[0,0] += 1
        W[row_offset_C + 0 * 3 + 0][start_col + 6] = 1   # C[0,0]
        
        return start_col + 7
    
    # Trivial 1x2 * 2x2 multiplication (rank 4)
    # A has 1 row, 2 cols. B has 2 rows, 2 cols. C has 1 row, 2 cols.
    # C[0] = A[0]*B[0,0] + A[1]*B[1,0]
    # C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    def trivial_1x2x2(U, V, W, row_offset_A, col_offset_B, row_offset_C, start_col):
        """
        Add trivial rank-4 decomposition for a 1x2 * 2x2 sub-problem.
        
        A[i,j] where i=0, j in {0,1}
        B[j,k] where j in {0,1}, k in {0,1}
        C[k,n+i] -> W[k*n + 0] for the single row
        """
        # Multiplication 1: A[0] * B[0,0] -> C[0,0]
        U[row_offset_A + 0 * 2 + 0][start_col] = 1
        V[col_offset_B + 0 * 2 + 0][start_col] = 1
        W[row_offset_C + 0 * 3 + 0][start_col] = 1  # C[0,0]
        
        # Multiplication 2: A[0] * B[0,1] -> C[0,1]
        U[row_offset_A + 0 * 2 + 0][start_col + 1] = 1
        V[col_offset_B + 0 * 2 + 1][start_col + 1] = 1
        W[row_offset_C + 1 * 3 + 0][start_col + 1] = 1  # C[0,1]
        
        # Multiplication 3: A[1] * B[1,0] -> C[0,0]
        U[row_offset_A + 0 * 2 + 1][start_col + 2] = 1
        V[col_offset_B + 1 * 2 + 0][start_col + 2] = 1
        W[row_offset_C + 0 * 3 + 0][start_col + 2] = 1  # C[0,0]
        
        # Multiplication 4: A[1] * B[1,1] -> C[0,1]
        U[row_offset_A + 0 * 2 + 1][start_col + 3] = 1
        V[col_offset_B + 1 * 2 + 1][start_col + 3] = 1
        W[row_offset_C + 1 * 3 + 0][start_col + 3] = 1  # C[0,1]
        
        return start_col + 4
    
    # Now build the full decomposition
    # U has shape (12, 44): U[i*4+j] for i in 0..2, j in 0..3
    # V has shape (16, 44): V[j*4+k] for j in 0..3, k in 0..3
    # W has shape (12, 44): W[k*3+i] for k in 0..3, i in 0..2
    
    col = 0
    
    # Top 2 rows (i=0,1), 4 sub-problems of 2x2 * 2x2
    # Sub-problem (row_group, col_group, inner_group):
    #   row_group: which 2 rows (only one option: rows 0,1)
    #   col_group: which 2 columns: left (k=0,1) or right (k=2,3)
    #   inner_group: which 2 inner indices: first (j=0,1) or last (j=2,3)
    
    for col_half in range(2):  # 0: cols 0,1; 1: cols 2,3
        for inner_half in range(2):  # 0: inner 0,1; 1: inner 2,3
            # A rows: 0,1 (global), cols: inner_half*2, inner_half*2+1
            # B rows: inner_half*2, inner_half*2+1, cols: col_half*2, col_half*2+1
            # C rows: 0,1 (global), cols: col_half*2, col_half*2+1
            
            row_offset_A = 0  # A rows start at 0
            col_offset_B = col_half * 2  # B cols start at col_half*2
            row_offset_C = col_half * 2  # C cols start at col_half*2 (but W is indexed by k*n+i)
            
            # For Strassen, we need local 2x2 indexing
            # A local: row r (0,1), col c (0,1) -> global A[r, inner_half*2+c] -> U[r*4 + inner_half*2+c]
            # B local: row r (0,1), col c (0,1) -> global B[inner_half*2+r, col_half*2+c] -> V[(inner_half*2+r)*4 + col_half*2+c]
            # C local: row r (0,1), col c (0,1) -> global C[r, col_half*2+c] -> W[(col_half*2+c)*3 + r]
            
            # I need to adapt the Strassen function to use proper global indexing
            # Let me inline it with correct offsets
            
            # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
            # A[0,0] -> U[0*4 + inner_half*2 + 0]
            # A[1,1] -> U[1*4 + inner_half*2 + 1]
            # B[0,0] -> V[(inner_half*2+0)*4 + col_half*2 + 0]
            # B[1,1] -> V[(inner_half*2+1)*4 + col_half*2 + 1]
            # C[0,0] -> W[(col_half*2+0)*3 + 0]
            # C[1,1] -> W[(col_half*2+1)*3 + 1]
            # C[0,1] -> W[(col_half*2+1)*3 + 0]
            # C[1,0] -> W[(col_half*2+0)*3 + 1]
            
            U[0 * 4 + inner_half * 2 + 0][col] = 1
            U[1 * 4 + inner_half * 2 + 1][col] = 1
            V[(inner_half * 2 + 0) * 4 + col_half * 2 + 0][col] = 1
            V[(inner_half * 2 + 1) * 4 + col_half * 2 + 1][col] = 1
            W[(col_half * 2 + 0) * 3 + 0][col] = 1
            W[(col_half * 2 + 1) * 3 + 1][col] = 1
            W[(col_half * 2 + 1) * 3 + 0][col] = -1
            W[(col_half * 2 + 0) * 3 + 1][col] = 1
            col += 1
            
            # M2 = (A[1,0]+A[1,1]) * B[0,0]
            U[1 * 4 + inner_half * 2 + 0][col] = 1
            U[1 * 4 + inner_half * 2 + 1][col] = 1
            V[(inner_half * 2 + 0) * 4 + col_half * 2 + 0][col] = 1
            W[(col_half * 2 + 0) * 3 + 1][col] = 1
            W[(col_half * 2 + 1) * 3 + 1][col] = -1
            col += 1
            
            # M3 = A[0,0] * (B[0,1] - B[1,1])
            U[0 * 4 + inner_half * 2 + 0][col] = 1
            V[(inner_half * 2 + 0) * 4 + col_half * 2 + 1][col] = 1
            V[(inner_half * 2 + 1) * 4 + col_half * 2 + 1][col] = -1
            W[(col_half * 2 + 1) * 3 + 0][col] = 1
            W[(col_half * 2 + 1) * 3 + 1][col] = 1
            col += 1
            
            # M4 = A[1,1] * (B[1,0] - B[0,0])
            U[1 * 4 + inner_half * 2 + 1][col] = 1
            V[(inner_half * 2 + 1) * 4 + col_half * 2 + 0][col] = 1
            V[(inner_half * 2 + 0) * 4 + col_half * 2 + 0][col] = -1
            W[(col_half * 2 + 0) * 3 + 0][col] = 1
            W[(col_half * 2 + 0) * 3 + 1][col] = 1
            col += 1
            
            # M5 = (A[0,0]+A[0,1]) * B[1,1]
            U[0 * 4 + inner_half * 2 + 0][col] = 1
            U[0 * 4 + inner_half * 2 + 1][col] = 1
            V[(inner_half * 2 + 1) * 4 + col_half * 2 + 1][col] = 1
            W[(col_half * 2 + 0) * 3 + 0][col] = -1
            W[(col_half * 2 + 1) * 3 + 0][col] = 1
            col += 1
            
            # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
            U[1 * 4 + inner_half * 2 + 0][col] = 1
            U[0 * 4 + inner_half * 2 + 0][col] = -1
            V[(inner_half * 2 + 0) * 4 + col_half * 2 + 0][col] = 1
            V[(inner_half * 2 + 0) * 4 + col_half * 2 + 1][col] = 1
            W[(col_half * 2 + 1) * 3 + 1][col] = 1
            col += 1
            
            # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
            U[0 * 4 + inner_half * 2 + 1][col] = 1
            U[1 * 4 + inner_half * 2 + 1][col] = -1
            V[(inner_half * 2 + 1) * 4 + col_half * 2 + 0][col] = 1
            V[(inner_half * 2 + 1) * 4 + col_half * 2 + 1][col] = 1
            W[(col_half * 2 + 0) * 3 + 0][col] = 1
            col += 1
    
    # Bottom row (i=2), 4 sub-problems of 1x2 * 2x2
    for col_half in range(2):  # 0: cols 0,1; 1: cols 2,3
        for inner_half in range(2):  # 0: inner 0,1; 1: inner 2,3
            # A: row 2, cols inner_half*2, inner_half*2+1
            # B: rows inner_half*2, inner_half*2+1, cols col_half*2, col_half*2+1
            # C: row 2, cols col_half*2, col_half*2+1
            
            # C[2, col_half*2+0] = A[2, inner_half*2+0]*B[inner_half*2+0, col_half*2+0] + A[2, inner_half*2+1]*B[inner_half*2+1, col_half*2+0]
            # C[2, col_half*2+1] = A[2, inner_half*2+0]*B[inner_half*2+0, col_half*2+1] + A[2, inner_half*2+1]*B[inner_half*2+1, col_half*2+1]
            
            # Mult 1: A[2, inner_half*2+0] * B[inner_half*2+0, col_half*2+0] -> C[2, col_half*2+0]
            U[2 * 4 + inner_half * 2 + 0][col] = 1
            V[(inner_half * 2 + 0) * 4 + col_half * 2 + 0][col] = 1
            W[(col_half * 2 + 0) * 3 + 2][col] = 1
            col += 1
            
            # Mult 2: A[2, inner_half*2+0] * B[inner_half*2+0, col_half*2+1] -> C[2, col_half*2+1]
            U[2 * 4 + inner_half * 2 + 0][col] = 1
            V[(inner_half * 2 + 0) * 4 + col_half * 2 + 1][col] = 1
            W[(col_half * 2 + 1) * 3 + 2][col] = 1
            col += 1
            
            # Mult 3: A[2, inner_half*2+1] * B[inner_half*2+1, col_half*2+0] -> C[2, col_half*2+0]
            U[2 * 4 + inner_half * 2 + 1][col] = 1
            V[(inner_half * 2 + 1) * 4 + col_half * 2 + 0][col] = 1
            W[(col_half * 2 + 0) * 3 + 2][col] = 1
            col += 1
            
            # Mult 4: A[2, inner_half*2+1] * B[inner_half*2+1, col_half*2+1] -> C[2, col_half*2+1]
            U[2 * 4 + inner_half * 2 + 1][col] = 1
            V[(inner_half * 2 + 1) * 4 + col_half * 2 + 1][col] = 1
            W[(col_half * 2 + 1) * 3 + 2][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
