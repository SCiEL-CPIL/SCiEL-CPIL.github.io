"""
Strategy: Structural Block Decomposition.
Instead of stochastic HOOI, we use a deterministic algebraic construction.
We decompose the 2x3 x 3x8 multiplication into four independent 2x3 x 3x2 sub-problems.
Each 2x3 x 3x2 multiplication is decomposed into:
1. A 2x2 x 2x2 multiplication (using Strassen's algorithm, rank 7).
2. A 2x1 x 1x2 multiplication (using schoolbook, rank 4).
Total rank per block: 7 + 4 = 11.
Total rank: 4 * 11 = 44.
This is an improvement over the schoolbook rank of 48.
We attempt to verify this construction exactly.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def _strassen_2x2(A, B):
    """
    Returns 7 products for 2x2 matrix multiplication using Strassen's algorithm.
    A, B are 2x2 matrices (flattened or structured).
    Returns a list of 7 linear forms (tuples of (rowA, colA, rowB, colB, coeff)).
    Actually, we need the factor matrices U, V, W for the tensor.
    For Strassen 2x2x2:
    U (4, 7), V (4, 7), W (4, 7)
    Indices:
    A: 00, 01, 10, 11 -> 0, 1, 2, 3
    B: 00, 01, 10, 11 -> 0, 1, 2, 3
    C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    Strassen products:
    P1 = (A00 + A11)(B00 + B11)
    P2 = (A10 + A11)B00
    P3 = A00(B01 - B11)
    P4 = A11(B10 - B00)
    P5 = (A00 + A01)B11
    P6 = (A10 - A00)(B00 + B01)
    P7 = (A01 - A11)(B10 + B11)
    
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1 # Wait, C10 gets P2, C11 gets -P2.
    # C10 is index 2. C11 is index 3.
    W[2, 1] = 1
    W[3, 1] = -1
    
    # P3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1 # C00 gets P4-P5+P7+P1. C01 gets P3+P5.
    # C00 (0): P1+P4-P5+P7
    # C01 (1): P3+P5
    # C10 (2): P2+P4
    # C11 (3): P1-P2+P3+P6
    
    # Let's re-map W carefully.
    # W[c, d] is coeff of P_d in C_c.
    
    # P1 (d=0): C00(+1), C11(+1)
    W[0, 0] = 1
    W[3, 0] = 1
    
    # P2 (d=1): C10(+1), C11(-1)
    W[2, 1] = 1
    W[3, 1] = -1
    
    # P3 (d=2): C01(+1), C11(+1)
    W[1, 2] = 1
    W[3, 2] = 1
    
    # P4 (d=3): C00(+1), C10(+1)
    W[0, 3] = 1
    W[2, 3] = 1
    
    # P5 (d=4): C00(-1), C01(+1)
    W[0, 4] = -1
    W[1, 4] = 1
    
    # P6 (d=5): C11(+1)
    W[3, 5] = 1
    
    # P7 (d=6): C00(+1)
    W[0, 6] = 1
    
    # Now U and V
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # P2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # P3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # P4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # P5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # P6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # P7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    return U, V, W

def _schoolbook_2x1x2():
    """
    Returns factor matrices for 2x1 x 1x2 multiplication.
    A is 2x1, B is 1x2, C is 2x2.
    C00 = A00 * B00
    C01 = A00 * B01
    C10 = A10 * B00
    C11 = A10 * B01
    Rank 4.
    U (2, 4), V (1, 4), W (4, 4)
    """
    U = np.zeros((2, 4))
    V = np.zeros((1, 4))
    W = np.zeros((4, 4))
    
    # d=0: A00 * B00 -> C00
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # d=1: A00 * B01 -> C01
    U[0, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # d=2: A10 * B00 -> C10
    U[1, 2] = 1
    V[0, 2] = 1
    W[2, 2] = 1
    
    # d=3: A10 * B01 -> C11
    U[1, 3] = 1
    V[0, 3] = 1
    W[3, 3] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS # 2, 3, 8
    
    # We decompose into 4 blocks of 2x3 x 3x2.
    # Block j handles columns 2j and 2j+1 of the 3x8 matrix B.
    # The result C is 2x8. Block j contributes to columns 2j and 2j+1 of C.
    
    # Total rank = 4 * (7 + 4) = 44.
    rank = 44
    
    # Initialize factor matrices
    # U: (n*m, rank) = (6, 44)
    # V: (m*p, rank) = (24, 44)
    # W: (p*n, rank) = (16, 44)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen part for 2x2x2 (uses first 2 cols of A, first 2 rows of B)
    # Schoolbook part for 2x1x2 (uses 3rd col of A, 3rd row of B)
    
    U_strassen, V_strassen, W_strassen = _strassen_2x2()
    U_sch, V_sch, W_sch = _schoolbook_2x1x2()
    
    # Mapping for a single 2x3 x 3x2 block:
    # A indices (2x3): 
    #   0: 00, 1: 01, 2: 02
    #   3: 10, 4: 11, 5: 12
    # B indices (3x2):
    #   0: 00, 1: 01
    #   2: 10, 3: 11
    #   4: 20, 5: 21
    # C indices (2x2):
    #   0: 00, 1: 01
    #   2: 10, 3: 11
    
    # Strassen operates on A[0:2, 0:2] (indices 0,1,3,4) and B[0:2, :] (indices 0,1,2,3).
    # Wait, Strassen 2x2x2 expects A 2x2, B 2x2.
    # Our A is 2x3. We take the first 2 columns.
    # Our B is 3x2. We take the first 2 rows.
    # The product covers C[0:2, 0:2] partially?
    # No, Strassen computes A_sub B_sub.
    # A_sub is A[:, 0:2] (2x2). B_sub is B[0:2, :] (2x2).
    # This product contributes to the full C (2x2).
    # The remaining term is A[:, 2] (2x1) * B[2, :] (1x2).
    
    for block_idx in range(4):
        # Offset for this block in the global matrices
        # U (A) is shared across blocks? No, A is the same for all blocks.
        # But in the tensor decomposition, each term is a rank-1 tensor.
        # The "rank" dimension is the sum of ranks of all blocks.
        # For block b, we add 11 new columns to U, V, W.
        
        base_col = block_idx * 11
        
        # --- Strassen Part (7 terms) ---
        # U_strassen is (4, 7). We need to map to A's 6 elements.
        # A indices in U_strassen: 0,1,2,3 correspond to A00, A01, A10, A11.
        # In our A (2x3), these are:
        # A00 -> index 0
        # A01 -> index 1
        # A10 -> index 3
        # A11 -> index 4
        # A02 -> index 2 (not used in Strassen)
        # A12 -> index 5 (not used in Strassen)
        
        a_map = [0, 1, 3, 4] # Strassen index -> Global A index
        
        for d in range(7):
            col = base_col + d
            # U
            for s_idx in range(4):
                if U_strassen[s_idx, d] != 0:
                    U[a_map[s_idx], col] = U_strassen[s_idx, d]
            
            # V_strassen is (4, 7). B indices 0,1,2,3 correspond to B00, B01, B10, B11.
            # In our B (3x2) for this block:
            # B is 3x2.
            # Global B indices for this block:
            # Block 0: B[0:3, 0:2] -> Global B indices 0..5
            # Block 1: B[0:3, 2:4] -> Global B indices 6..11
            # Block b: B[0:3, 2b:2b+2]
            
            # B indices in V_strassen:
            # 0: B00 -> Global B[0, 2b]
            # 1: B01 -> Global B[0, 2b+1]
            # 2: B10 -> Global B[1, 2b]
            # 3: B11 -> Global B[1, 2b+1]
            
            b_row_offset = block_idx * 2 # Column offset in B
            # B global index = row * p + col
            # p = 8.
            
            b_map = [
                0 * 8 + b_row_offset,     # B00
                0 * 8 + b_row_offset + 1, # B01
                1 * 8 + b_row_offset,     # B10
                1 * 8 + b_row_offset + 1  # B11
            ]
            
            for s_idx in range(4):
                if V_strassen[s_idx, d] != 0:
                    V[b_map[s_idx], col] = V_strassen[s_idx, d]
            
            # W_strassen is (4, 7). C indices 0,1,2,3.
            # C global indices for this block:
            # C is 2x8.
            # Block b contributes to C[:, 2b] and C[:, 2b+1].
            # C indices:
            # 0: C00 -> Global C[0, 2b]
            # 1: C01 -> Global C[0, 2b+1]
            # 2: C10 -> Global C[1, 2b]
            # 3: C11 -> Global C[1, 2b+1]
            
            c_col_offset = block_idx * 2
            c_map = [
                0 * 8 + c_col_offset,     # C00 (n=2, p=8, so C[i, j] is i*8+j? No, W is p*n. W[k*n+i, r].
                # Wait, the interface says W[k*n+i, r].
                # C is 2x8. i is row (0,1), k is col (0..7).
                # Index in W = k * n + i = k * 2 + i.
                
                # Let's re-read the convention: W[k*n+i, r].
                # n=2, p=8.
                # C[i, k] where i in 0..1, k in 0..7.
                # W index = k * 2 + i.
                
                # C00 (i=0, k=2b) -> W[2b * 2 + 0] = 4b
                # C01 (i=0, k=2b+1) -> W[(2b+1) * 2 + 0] = 4b + 2
                # C10 (i=1, k=2b) -> W[2b * 2 + 1] = 4b + 1
                # C11 (i=1, k=2b+1) -> W[(2b+1) * 2 + 1] = 4b + 3
            ]
            
            c_map_corrected = [
                4 * block_idx,
                4 * block_idx + 2,
                4 * block_idx + 1,
                4 * block_idx + 3
            ]
            
            for s_idx in range(4):
                if W_strassen[s_idx, d] != 0:
                    W[c_map_corrected[s_idx], col] = W_strassen[s_idx, d]
        
        # --- Schoolbook Part (4 terms) ---
        # A[:, 2] (2x1) * B[2, :] (1x2)
        # A indices: 2 (A02), 5 (A12)
        # B indices: 4 (B20), 5 (B21) relative to block.
        # Global B indices:
        # B20 -> Global B[2, 2b] -> 2*8 + 2b = 16 + 2b
        # B21 -> Global B[2, 2b+1] -> 2*8 + 2b+1 = 17 + 2b
        
        # U_sch (2, 4):
        # U_sch[0, d] is coeff for A00 (here A02).
        # U_sch[1, d] is coeff for A10 (here A12).
        
        # V_sch (1, 4):
        # V_sch[0, d] is coeff for B00 (here B20 or B21 depending on d).
        # Wait, V_sch in _schoolbook_2x1x2 was for B 1x2.
        # V_sch[0, d] applies to the single row of B.
        # In our case, B[2, :] has 2 elements.
        # The schoolbook function returned V of shape (1, 4).
        # This implies V[0, d] is the coeff for the single "column" of B?
        # No, B is 1x2.
        # My _schoolbook_2x1x2 function:
        # V is (1, 4).
        # d=0: B00. d=1: B01. d=2: B00. d=3: B01.
        # So V[0, d] is 1 for both B00 and B01 cases?
        # Let's check _schoolbook_2x1x2:
        # d=0: V[0,0]=1 (B00)
        # d=1: V[0,1]=1 (B01)
        # d=2: V[0,2]=1 (B00)
        # d=3: V[0,3]=1 (B01)
        # This is correct for a 1x2 matrix B where we pick one element at a time?
        # No, the rank-1 term is A_i * B_j.
        # If B is 1x2, B has 2 elements.
        # The term A00 * B00 uses B[0,0].
        # The term A00 * B01 uses B[0,1].
        # So in the factor V, for a given term d, we need to specify which element of B is used.
        # My V_sch is (1, 4). This is insufficient if B has 2 columns.
        # V should be (m*p, rank). For the sub-problem, B is 1x2.
        # So V_local should be (2, 4).
        
        # Let's fix _schoolbook_2x1x2 to return V of shape (2, 4).
        
    # Re-implementing schoolbook part correctly inside the loop for clarity
    
    for block_idx in range(4):
        base_col = block_idx * 11
        sch_base = base_col + 7
        
        # A indices: 2 (A02), 5 (A12)
        # B indices: 16+2b (B20), 17+2b (B21)
        # C indices: 4b (C00), 4b+2 (C01), 4b+1 (C10), 4b+3 (C11)
        
        # Term 0: A02 * B20 -> C00
        col = sch_base + 0
        U[2, col] = 1
        V[16 + 2*block_idx, col] = 1
        W[4*block_idx, col] = 1
        
        # Term 1: A02 * B21 -> C01
        col = sch_base + 1
        U[2, col] = 1
        V[17 + 2*block_idx, col] = 1
        W[4*block_idx + 2, col] = 1
        
        # Term 2: A12 * B20 -> C10
        col = sch_base + 2
        U[5, col] = 1
        V[16 + 2*block_idx, col] = 1
        W[4*block_idx + 1, col] = 1
        
        # Term 3: A12 * B21 -> C11
        col = sch_base + 3
        U[5, col] = 1
        V[17 + 2*block_idx, col] = 1
        W[4*block_idx + 3, col] = 1

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Clean up floats
    def clean(val):
        if abs(val) < 1e-10: return 0
        if abs(val * 2 - round(val * 2)) < 1e-10: return round(val * 2) / 2
        return val
        
    U_list = [[clean(v) for v in row] for row in U_list]
    V_list = [[clean(v) for v in row] for row in V_list]
    W_list = [[clean(v) for v in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
