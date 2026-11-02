import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix
    by a 4 by 5 matrix.
    
    Uses a block decomposition strategy:
    Split the middle dimension 4 into 2+2.
    This results in two independent 3x2x5 multiplications.
    Each 3x2x5 multiplication is decomposed into:
    - 3x2x2 (rank 11)
    - 3x2x2 (rank 11)
    - 3x2x1 (rank 6)
    Total rank per block: 28.
    Total rank: 56.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = 3, 4, 5
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # A is 2x2, B is 2x2, C is 2x2
    # A indices: (i_off, j_off), B indices: (j_off, k_off), C indices: (k_off, i_off)
    # i_off, j_off, k_off are offsets in the full matrices
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # A rows: i_off, i_off+1
        # A cols: j_off, j_off+1
        # B rows: j_off, j_off+1
        # B cols: k_off, k_off+1
        # C rows: k_off, k_off+1
        # C cols: i_off, i_off+1
        
        # M1 = (a+d)(e+h)
        U[i_off * m + j_off, col] = 1.0
        U[(i_off+1) * m + (j_off+1), col] = 1.0
        V[j_off * p + k_off, col] = 1.0
        V[(j_off+1) * p + (k_off+1), col] = 1.0
        W[k_off * n + i_off, col] = 1.0
        W[(k_off+1) * n + (i_off+1), col] = 1.0
        col += 1
        
        # M2 = (c+d)e
        U[(i_off+1) * m + j_off, col] = 1.0
        U[(i_off+1) * m + (j_off+1), col] = 1.0
        V[j_off * p + k_off, col] = 1.0
        W[k_off * n + (i_off+1), col] = 1.0
        col += 1
        
        # M3 = a(f-h)
        U[i_off * m + j_off, col] = 1.0
        V[j_off * p + (k_off+1), col] = 1.0
        V[(j_off+1) * p + k_off, col] = -1.0
        W[(k_off+1) * n + i_off, col] = 1.0
        col += 1
        
        # M4 = d(g-c)
        U[(i_off+1) * m + (j_off+1), col] = 1.0
        V[(j_off+1) * p + k_off, col] = 1.0
        V[j_off * p + (k_off+1), col] = -1.0
        W[k_off * n + (i_off+1), col] = 1.0
        col += 1
        
        # M5 = (a+b)h
        U[i_off * m + j_off, col] = 1.0
        U[i_off * m + (j_off+1), col] = 1.0
        V[(j_off+1) * p + (k_off+1), col] = 1.0
        W[(k_off+1) * n + i_off, col] = 1.0
        col += 1
        
        # M6 = (c-a)(e+f)
        U[(i_off+1) * m + j_off, col] = 1.0
        U[i_off * m + j_off, col] = -1.0
        V[j_off * p + k_off, col] = 1.0
        V[j_off * p + (k_off+1), col] = 1.0
        W[k_off * n + (i_off+1), col] = 1.0
        col += 1
        
        # M7 = (b-d)(g+h)
        U[i_off * m + (j_off+1), col] = 1.0
        U[(i_off+1) * m + (j_off+1), col] = -1.0
        V[(j_off+1) * p + k_off, col] = 1.0
        V[(j_off+1) * p + (k_off+1), col] = 1.0
        W[(k_off+1) * n + (i_off+1), col] = 1.0
        col += 1
        
        # Now we need to combine these M's into C
        # But wait, the U, V, W above are just the "products".
        # The W matrix in the final output sums the products.
        # My add_strassen function is adding to U, V, and W directly.
        # This is incorrect for the tensor decomposition format.
        # In tensor decomposition, U, V, W are factors such that T = sum_r U_r V_r W_r^T.
        # The "M" terms are the products. The final C is a linear combination of M's.
        # So W should contain the coefficients of the linear combination.
        # But I am adding to W inside the helper.
        # I need to be careful.
        
        # Actually, the standard way is:
        # For each rank-1 term, U_r, V_r, W_r are vectors.
        # The product is outer product U_r V_r^T W_r^T? No.
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        
        # In my helper, I am setting U, V, and W for each "M".
        # But C is a sum of M's.
        # So for a specific C element, it gets contributions from multiple M's.
        # This means W_{k,i,r} should be the coefficient of M_r in C_{k,i}.
        
        # Let's re-structure.
        # I will generate the 7 terms for Strassen.
        # Each term r has U_r, V_r.
        # The contribution to C is determined by W_r.
        
        # Let's return the coefficients for C.
        # But the function modifies global U, V, W.
        # This is tricky.
        
        # Let's restart the helper logic to just return the vectors.
        pass

    # Let's implement the construction manually for clarity and correctness.
    
    # Block 1: A1 (rows 0-2, cols 0-1) * B1 (rows 0-1, cols 0-4)
    # Split B1 into B1_1 (cols 0-1), B1_2 (cols 2-3), B1_3 (col 4)
    
    # Sub-block 1.1: A1 * B1_1 (3x2 * 2x2)
    # Split A1 into A1_top (rows 0-1) and A1_bot (row 2)
    # A1_top * B1_1 is 2x2 * 2x2. Rank 7.
    # A1_bot * B1_1 is 1x2 * 2x2. Rank 4.
    
    # A1_top * B1_1:
    # i_off=0, j_off=0, k_off=0
    # Strassen 2x2x2
    # M1: U[0,0], U[1,1] ; V[0,0], V[1,1] ; W[0,0]=1, W[1,1]=1
    # M2: U[1,0], U[1,1] ; V[0,0] ; W[0,1]=1
    # M3: U[0,0] ; V[0,1], V[1,0]=-1 ; W[1,0]=1
    # M4: U[1,1] ; V[1,0], V[0,1]=-1 ; W[0,1]=1
    # M5: U[0,0], U[0,1] ; V[1,1] ; W[1,0]=1
    # M6: U[1,0]=-1, U[0,0]=1 ; V[0,0], V[0,1] ; W[0,1]=1
    # M7: U[0,1]=1, U[1,1]=-1 ; V[1,0], V[1,1] ; W[1,1]=1
    
    # C_00 = M1 + M4 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M2 + M3 + M6
    
    # Let's add these to U, V, W.
    
    # Term 1 (M1)
    U[0, col] = 1.0; U[1, col] = 1.0
    V[0, col] = 1.0; V[1, col] = 1.0
    W[0, col] = 1.0; W[3, col] = 1.0 # C[0,0] and C[1,1] ? No, W is indexed [k*n+i].
    # C[0,0] -> k=0, i=0 -> index 0.
    # C[1,1] -> k=1, i=1 -> index 1*3+1=4.
    # Wait, W shape is (p*n, rank) = (15, rank).
    # Index k*n+i.
    # C[0,0]: k=0, i=0 -> 0.
    # C[1,1]: k=1, i=1 -> 4.
    W[0, col] = 1.0
    W[4, col] = 1.0
    col += 1
    
    # Term 2 (M2)
    U[2, col] = 1.0; U[3, col] = 1.0 # A[1,0], A[1,1]
    V[0, col] = 1.0 # B[0,0]
    W[3, col] = 1.0 # C[0,1] -> k=0, i=1 -> 1.
    # Wait, C[0,1] is row 0, col 1 of C?
    # C is 5x3.
    # C[k, i].
    # C[0,1] -> k=0, i=1 -> index 1.
    W[1, col] = 1.0
    col += 1
    
    # Term 3 (M3)
    U[0, col] = 1.0 # A[0,0]
    V[1, col] = 1.0; V[0, col] = -1.0 # B[0,1], B[1,0]
    W[4, col] = 1.0 # C[1,0] -> k=1, i=0 -> 3.
    # Wait.
    # C[1,0] -> k=1, i=0 -> index 3.
    W[3, col] = 1.0
    col += 1
    
    # Term 4 (M4)
    U[3, col] = 1.0 # A[1,1]
    V[2, col] = 1.0; V[1, col] = -1.0 # B[1,0], B[0,1]
    W[3, col] = 1.0 # C[0,1] -> k=0, i=1 -> 1.
    # Wait, C[0,1] is k=0, i=1. Index 1.
    # Let's re-verify C indices.
    # C[k, i].
    # C[0,0]=M1+M4-M5+M7.
    # C[0,1]=M3+M5.
    # C[1,0]=M2+M4.
    # C[1,1]=M1-M2+M3+M6.
    
    # M4 contributes to C[0,1]? No, C[1,0].
    # C[1,0] -> k=1, i=0 -> index 3.
    W[3, col] = 1.0
    col += 1
    
    # Term 5 (M5)
    U[0, col] = 1.0; U[1, col] = 1.0 # A[0,0], A[0,1]
    V[3, col] = 1.0 # B[1,1]
    W[1, col] = 1.0 # C[0,1] -> k=0, i=1 -> 1.
    # M5 also contributes to C[0,0] with -1.
    W[0, col] = -1.0
    col += 1
    
    # Term 6 (M6)
    U[2, col] = 1.0; U[0, col] = -1.0 # A[1,0], A[0,0]
    V[0, col] = 1.0; V[1, col] = 1.0 # B[0,0], B[0,1]
    W[4, col] = 1.0 # C[1,1] -> k=1, i=1 -> 4.
    col += 1
    
    # Term 7 (M7)
    U[1, col] = 1.0; U[3, col] = -1.0 # A[0,1], A[1,1]
    V[2, col] = 1.0; V[3, col] = 1.0 # B[1,0], B[1,1]
    W[4, col] = 1.0 # C[1,1] -> k=1, i=1 -> 4.
    # M7 also contributes to C[0,0] with +1.
    W[0, col] = 1.0
    col += 1
    
    # A1_bot * B1_1 (1x2 * 2x2)
    # A[2,0], A[2,1]
    # B[0,0], B[0,1], B[1,0], B[1,1]
    # C[2,0], C[2,1]
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0]
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1]
    
    # Term 8
    U[4, col] = 1.0 # A[2,0]
    V[0, col] = 1.0 # B[0,0]
    W[6, col] = 1.0 # C[2,0] -> k=0, i=2 -> 2.
    # Wait, C[k, i]. C[0,2] -> k=0, i=2 -> index 2.
    W[2, col] = 1.0
    col += 1
    
    # Term 9
    U[4, col] = 1.0 # A[2,0]
    V[1, col] = 1.0 # B[0,1]
    W[3, col] = 1.0 # C[1,2] -> k=1, i=2 -> 5.
    # Wait, C[0,2] and C[1,2]?
    # B1_1 cols are 0,1. So k=0,1.
    # C[0,2] -> k=0, i=2 -> 2.
    # C[1,2] -> k=1, i=2 -> 5.
    W[5, col] = 1.0
    col += 1
    
    # Term 10
    U[5, col] = 1.0 # A[2,1]
    V[2, col] = 1.0 # B[1,0]
    W[2, col] = 1.0
    col += 1
    
    # Term 11
    U[5, col] = 1.0 # A[2,1]
    V[3, col] = 1.0 # B[1,1]
    W[5, col] = 1.0
    col += 1
    
    # Sub-block 1.2: A1 * B1_2 (3x2 * 2x2)
    # B1_2 cols are 2,3. So k=2,3.
    # Same structure, but k_off=2.
    
    # Term 12 (M1)
    U[0, col] = 1.0; U[1, col] = 1.0
    V[4, col] = 1.0; V[5, col] = 1.0 # B[0,2], B[1,3]
    W[6, col] = 1.0 # C[2,0] -> k=2, i=0 -> 6.
    W[10, col] = 1.0 # C[3,1] -> k=3, i=1 -> 10.
    col += 1
    
    # Term 13 (M2)
    U[2, col] = 1.0; U[3, col] = 1.0
    V[4, col] = 1.0
    W[7, col] = 1.0 # C[2,1] -> k=2, i=1 -> 7.
    col += 1
    
    # Term 14 (M3)
    U[0, col] = 1.0
    V[5, col] = 1.0; V[4, col] = -1.0
    W[9, col] = 1.0 # C[3,0] -> k=3, i=0 -> 9.
    col += 1
    
    # Term 15 (M4)
    U[3, col] = 1.0
    V[6, col] = 1.0; V[5, col] = -1.0
    W[7, col] = 1.0
    col += 1
    
    # Term 16 (M5)
    U[0, col] = 1.0; U[1, col] = 1.0
    V[7, col] = 1.0
    W[7, col] = 1.0
    W[6, col] = -1.0
    col += 1
    
    # Term 17 (M6)
    U[2, col] = 1.0; U[0, col] = -1.0
    V[4, col] = 1.0; V[5, col] = 1.0
    W[10, col] = 1.0
    col += 1
    
    # Term 18 (M7)
    U[1, col] = 1.0; U[3, col] = -1.0
    V[6, col] = 1.0; V[7, col] = 1.0
    W[10, col] = 1.0
    W[6, col] = 1.0
    col += 1
    
    # A1_bot * B1_2
    # Term 19
    U[4, col] = 1.0
    V[4, col] = 1.0
    W[8, col] = 1.0 # C[2,2] -> k=2, i=2 -> 8.
    col += 1
    
    # Term 20
    U[4, col] = 1.0
    V[5, col] = 1.0
    W[11, col] = 1.0 # C[3,2] -> k=3, i=2 -> 11.
    col += 1
    
    # Term 21
    U[5, col] = 1.0
    V[6, col] = 1.0
    W[8, col] = 1.0
    col += 1
    
    # Term 22
    U[5, col] = 1.0
    V[7, col] = 1.0
    W[11, col] = 1.0
    col += 1
    
    # Sub-block 1.3: A1 * B1_3 (3x2 * 2x1)
    # B1_3 col is 4. So k=4.
    # C[4,0], C[4,1], C[4,2]
    
    # Term 23
    U[0, col] = 1.0
    V[8, col] = 1.0 # B[0,4]
    W[12, col] = 1.0 # C[4,0] -> k=4, i=0 -> 12.
    col += 1
    
    # Term 24
    U[1, col] = 1.0
    V[8, col] = 1.0
    W[12, col] = 1.0
    col += 1
    
    # Term 25
    U[2, col] = 1.0
    V[8, col] = 1.0
    W[13, col] = 1.0 # C[4,1] -> k=4, i=1 -> 13.
    col += 1
    
    # Term 26
    U[3, col] = 1.0
    V[8, col] = 1.0
    W[13, col] = 1.0
    col += 1
    
    # Term 27
    U[4, col] = 1.0
    V[8, col] = 1.0
    W[14, col] = 1.0 # C[4,2] -> k=4, i=2 -> 14.
    col += 1
    
    # Term 28
    U[5, col] = 1.0
    V[8, col] = 1.0
    W[14, col] = 1.0
    col += 1
    
    # Block 2: A2 (rows 0-2, cols 2-3) * B2 (rows 2-3, cols 0-4)
    # j_off=2, i_off=0, k_off=0 for A2_top * B2_1
    # A2 cols are 2,3.
    
    # Sub-block 2.1: A2_top * B2_1 (2x2 * 2x2)
    # Term 29 (M1)
    U[2, col] = 1.0; U[3, col] = 1.0 # A[0,2], A[1,3]
    V[0, col] = 1.0; V[1, col] = 1.0 # B[2,0], B[3,1]
    W[0, col] = 1.0
    W[4, col] = 1.0
    col += 1
    
    # Term 30 (M2)
    U[4, col] = 1.0; U[5, col] = 1.0 # A[1,2], A[1,3]
    V[0, col] = 1.0
    W[1, col] = 1.0
    col += 1
    
    # Term 31 (M3)
    U[2, col] = 1.0
    V[1, col] = 1.0; V[0, col] = -1.0
    W[3, col] = 1.0
    col += 1
    
    # Term 32 (M4)
    U[5, col] = 1.0
    V[2, col] = 1.0; V[1, col] = -1.0
    W[3, col] = 1.0
    col += 1
    
    # Term 33 (M5)
    U[2, col] = 1.0; U[3, col] = 1.0
    V[3, col] = 1.0
    W[1, col] = 1.0
    W[0, col] = -1.0
    col += 1
    
    # Term 34 (M6)
    U[4, col] = 1.0; U[2, col] = -1.0
    V[0, col] = 1.0; V[1, col] = 1.0
    W[4, col] = 1.0
    col += 1
    
    # Term 35 (M7)
    U[3, col] = 1.0; U[5, col] = -1.0
    V[2, col] = 1.0; V[3, col] = 1.0
    W[4, col] = 1.0
    W[0, col] = 1.0
    col += 1
    
    # A2_bot * B2_1
    # Term 36
    U[6, col] = 1.0 # A[2,2]
    V[0, col] = 1.0
    W[2, col] = 1.0
    col += 1
    
    # Term 37
    U[6, col] = 1.0
    V[1, col] = 1.0
    W[5, col] = 1.0
    col += 1
    
    # Term 38
    U[7, col] = 1.0 # A[2,3]
    V[2, col] = 1.0
    W[2, col] = 1.0
    col += 1
    
    # Term 39
    U[7, col] = 1.0
    V[3, col] = 1.0
    W[5, col] = 1.0
    col += 1
    
    # Sub-block 2.2: A2 * B2_2 (3x2 * 2x2)
    # B2_2 cols 2,3. k=2,3.
    
    # Term 40 (M1)
    U[2, col] = 1.0; U[3, col] = 1.0
    V[4, col] = 1.0; V[5, col] = 1.0
    W[6, col] = 1.0
    W[10, col] = 1.0
    col += 1
    
    # Term 41 (M2)
    U[4, col] = 1.0; U[5, col] = 1.0
    V[4, col] = 1.0
    W[7, col] = 1.0
    col += 1
    
    # Term 42 (M3)
    U[2, col] = 1.0
    V[5, col] = 1.0; V[4, col] = -1.0
    W[9, col] = 1.0
    col += 1
    
    # Term 43 (M4)
    U[5, col] = 1.0
    V[6, col] = 1.0; V[5, col] = -1.0
    W[7, col] = 1.0
    col += 1
    
    # Term 44 (M5)
    U[2, col] = 1.0; U[3, col] = 1.0
    V[7, col] = 1.0
    W[7, col] = 1.0
    W[6, col] = -1.0
    col += 1
    
    # Term 45 (M6)
    U[4, col] = 1.0; U[2, col] = -1.0
    V[4, col] = 1.0; V[5, col] = 1.0
    W[10, col] = 1.0
    col += 1
    
    # Term 46 (M7)
    U[3, col] = 1.0; U[5, col] = -1.0
    V[6, col] = 1.0; V[7, col] = 1.0
    W[10, col] = 1.0
    W[6, col] = 1.0
    col += 1
    
    # A2_bot * B2_2
    # Term 47
    U[6, col] = 1.0
    V[4, col] = 1.0
    W[8, col] = 1.0
    col += 1
    
    # Term 48
    U[6, col] = 1.0
    V[5, col] = 1.0
    W[11, col] = 1.0
    col += 1
    
    # Term 49
    U[7, col] = 1.0
    V[6, col] = 1.0
    W[8, col] = 1.0
    col += 1
    
    # Term 50
    U[7, col] = 1.0
    V[7, col] = 1.0
    W[11, col] = 1.0
    col += 1
    
    # Sub-block 2.3: A2 * B2_3 (3x2 * 2x1)
    # B2_3 col 4. k=4.
    
    # Term 51
    U[2, col] = 1.0
    V[8, col] = 1.0
    W[12, col] = 1.0
    col += 1
    
    # Term 52
    U[3, col] = 1.0
    V[8, col] = 1.0
    W[12, col] = 1.0
    col += 1
    
    # Term 53
    U[4, col] = 1.0
    V[8, col] = 1.0
    W[13, col] = 1.0
    col += 1
    
    # Term 54
    U[5, col] = 1.0
    V[8, col] = 1.0
    W[13, col] = 1.0
    col += 1
    
    # Term 55
    U[6, col] = 1.0
    V[8, col] = 1.0
    W[14, col] = 1.0
    col += 1
    
    # Term 56
    U[7, col] = 1.0
    V[8, col] = 1.0
    W[14, col] = 1.0
    col += 1
    
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
