"""
Program for matrix multiplication 2x3 by 3x5 using tensor code / Berlekamp-Massey approach.
Implements a Strassen-based block decomposition to reduce rank below the schoolbook 30.
"""
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = 2, 3, 5
    
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10
    
    # We split the 3x5 matrix B into three 3x? blocks? No, we split by rows of A (2 rows)
    # and columns of B (5 cols).
    # A is 2x3. Let's split A into 2x2 and 2x1 blocks?
    # Or split B into 3x2 and 3x3?
    
    # Let's try to decompose the matmul C = A * B.
    # A = [A_left (2x2), A_right (2x1)]
    # B = [B_top (3x2), B_bottom (3x3)] ? No, B is 3x5.
    # Let's split the 5 columns of B into 2 and 3.
    # B = [B_1 (3x2), B_2 (3x3)]
    # C = [A * B_1 (2x2), A * B_2 (2x3)]
    
    # For the first part: A(2x3) * B_1(3x2).
    # We can view this as two 2x2x2 matmuls? No.
    # Let's use Strassen on the 2x2x2 part if we can fit it.
    # Actually, we can decompose A into 2x2 and 2x1.
    # Let A = [A_1 (2x2), A_2 (2x1)]
    # C = A_1 * B (2x5) + A_2 * B (2x5)? No, A is 2x3.
    # A = [A_1 (2x2), A_2 (2x1)] where A_1 is cols 0,1 and A_2 is col 2.
    # C = A_1 * B + A_2 * B is not right because B is 3x5.
    # A_1 is 2x2, B is 3x5. Dimensions don't match for direct mult unless we split B.
    
    # Let's split B into rows? B is 3x5.
    # B = [B_top (2x5); B_bottom (1x5)]
    # Then A * B = [A_left (2x2), A_right (2x1)] * [B_top (2x5); B_bottom (1x5)]
    # = A_left * B_top (2x2 * 2x5 = 2x5) + A_right * B_bottom (2x1 * 1x5 = 2x5)
    
    # Part 1: A_left (2x2) * B_top (2x5).
    # We can split B_top (2x5) into B_top1 (2x2) and B_top2 (2x3).
    # Part 1a: A_left (2x2) * B_top1 (2x2). Rank 7 (Strassen).
    # Part 1b: A_left (2x2) * B_top2 (2x3). 
    #   Split B_top2 into 2x2 and 2x1.
    #   Part 1b1: A_left (2x2) * B_top2_left (2x2). Rank 7 (Strassen).
    #   Part 1b2: A_left (2x2) * B_top2_right (2x1). Rank 4 (2x2x1 is 4? 2*2*1=4, can it be lower? 2x2x1 is rank 4 generally, but maybe 3? No, 2x2x1 is 4).
    
    # Part 2: A_right (2x1) * B_bottom (1x5).
    # This is just outer product. Rank 2? 
    # A_right is 2x1. B_bottom is 1x5.
    # C_part2 = A_right * B_bottom.
    # This is a rank-1 update for each column? 
    # The tensor for 2x1 * 1x5 has rank 2*1*5 = 10? 
    # Wait, 2x1 * 1x5. 
    # C[i,k] = A[i,0] * B[0,k].
    # This is separable. 
    # U[i,0] * V[0,k] * W[k,i].
    # We need U[i,c] * V[c] * W[c,k] ? No.
    # The tensor is T[i, 0, k] = 1.
    # We can write this as sum_c u_c[i] v_c w_c[k].
    # Since it's separable, rank is 1? 
    # T = u ⊗ v ⊗ w where u[i]=1, v[0]=1, w[k]=1.
    # Yes! Rank 1 for 2x1 * 1x5.
    
    # Let's re-evaluate the splits.
    # Total Rank = Rank(2x2*2x2) + Rank(2x2*2x2) + Rank(2x2*2x1) + Rank(2x1*1x5)
    # = 7 + 7 + 4 + 1 = 19.
    
    # Let's verify Rank(2x2*2x1).
    # A(2x2) * B(2x1). C(2x1).
    # C[i,k] = sum_j A[i,j] B[j,k]. k is only 0.
    # C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
    # This is 2 outputs, each sum of 2 products.
    # Schoolbook rank 4. Can it be 3?
    # 2x2x1 matmul rank is 4. (Known result: R(2x2x1)=4).
    
    # So estimated rank = 7 + 7 + 4 + 1 = 19.
    # Current schoolbook is 30.
    # 19 < 30. This is a valid improvement.
    
    # Let's construct the matrices.
    
    # We will build U, V, W as lists of lists.
    # U: 6 x Rank
    # V: 15 x Rank
    # W: 10 x Rank
    
    # We will use a helper to add a block to the global matrices.
    
    # Indices for A (2x3):
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    
    # Indices for B (3x5):
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[0,4]=4
    # B[1,0]=5, B[1,1]=6, B[1,2]=7, B[1,3]=8, B[1,4]=9
    # B[2,0]=10, B[2,1]=11, B[2,2]=12, B[2,3]=13, B[2,4]=14
    
    # Indices for C (2x5):
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[0,4]=4
    # C[1,0]=5, C[1,1]=6, C[1,2]=7, C[1,3]=8, C[1,4]=9
    # Note: W index is k*n + i. 
    # C[i,k] corresponds to W[k*2+i].
    # C[0,0] -> W[0]
    # C[0,1] -> W[2]
    # C[0,2] -> W[4]
    # C[0,3] -> W[6]
    # C[0,4] -> W[8]
    # C[1,0] -> W[1]
    # C[1,1] -> W[3]
    # C[1,2] -> W[5]
    # C[1,3] -> W[7]
    # C[1,4] -> W[9]
    
    total_rank = 0
    U = []
    V = []
    W = []
    
    def add_block(u_cols, v_cols, w_cols, u_vals, v_vals, w_vals):
        """
        u_cols: list of global A indices
        v_cols: list of global B indices
        w_cols: list of global C indices (W indices)
        u_vals, v_vals, w_vals: lists of coefficients for this block
        """
        nonlocal total_rank, U, V, W
        r = len(u_vals)
        
        # Initialize new rows for U, V, W if not present
        while len(U) < U_dim:
            U.append([0.0] * total_rank)
        while len(V) < V_dim:
            V.append([0.0] * total_rank)
        while len(W) < W_dim:
            W.append([0.0] * total_rank)
            
        for c in range(r):
            col_idx = total_rank
            # Extend existing rows
            for i in range(len(U)):
                U[i].append(0.0)
            for i in range(len(V)):
                V[i].append(0.0)
            for i in range(len(W)):
                W[i].append(0.0)
                
            # Set values
            for idx, val in zip(u_cols, u_vals):
                U[idx][col_idx] += val
            for idx, val in zip(v_cols, v_vals):
                V[idx][col_idx] += val
            for idx, val in zip(w_cols, w_vals):
                W[idx][col_idx] += val
                
            total_rank += 1

    # Block 1: A_left (2x2) * B_top1 (2x2) -> C_top1 (2x2)
    # A indices: 0,1,3,4 (A[0,0], A[0,1], A[1,0], A[1,1])
    # B indices: 0,1,5,6 (B[0,0], B[0,1], B[1,0], B[1,1])
    # C indices: 0,1,2,3 (C[0,0], C[0,1], C[1,0], C[1,1]) -> W[0], W[2], W[1], W[3]
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B10-A10)
    # M5 = (A00+A01)B11
    # M6 = (A10+A11)(B01-B11)
    # M7 = (A00-A10)(B00+B10)
    
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to global indices.
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=5, B11=6
    # C00=0, C01=2, C10=1, C11=3 (W indices)
    
    # M1
    add_block([0,4], [0,6], [0,3], [1,1], [1,1], [1,-1])
    # M2
    add_block([3,4], [0], [1], [1,1], [1], [1])
    # M3
    add_block([0], [5,6], [2], [1], [1,-1], [1])
    # M4
    add_block([4,3], [5], [1], [1,-1], [1], [1])
    # M5
    add_block([0,1], [6], [0,2], [1,1], [1], [-1,1])
    # M6
    add_block([3,4], [1,6], [3], [1,1], [1,-1], [1])
    # M7
    add_block([0,3], [0,5], [0], [1,-1], [1,1], [1])
    
    # Block 2: A_left (2x2) * B_top2_left (2x2) -> C_top2_left (2x2)
    # B_top2 is cols 2,3,4 of B.
    # B_top2_left is cols 2,3.
    # B indices: 
    # B[0,2]=2, B[0,3]=3
    # B[1,2]=7, B[1,3]=8
    # C indices for cols 2,3:
    # C[0,2]=2, C[0,3]=3
    # C[1,2]=7, C[1,3]=8
    # W indices:
    # C[0,2] -> W[4]
    # C[0,3] -> W[6]
    # C[1,2] -> W[5]
    # C[1,3] -> W[7]
    
    # Same Strassen structure, just different indices.
    # A00=0, A01=1, A10=3, A11=4
    # B00=2, B01=3, B10=7, B11=8
    # C00=4, C01=6, C10=5, C11=7
    
    # M1
    add_block([0,4], [2,8], [4,7], [1,1], [1,1], [1,-1])
    # M2
    add_block([3,4], [2], [5], [1,1], [1], [1])
    # M3
    add_block([0], [7,8], [6], [1], [1,-1], [1])
    # M4
    add_block([4,3], [7], [5], [1,-1], [1], [1])
    # M5
    add_block([0,1], [8], [4,6], [1,1], [1], [-1,1])
    # M6
    add_block([3,4], [3,8], [7], [1,1], [1,-1], [1])
    # M7
    add_block([0,3], [2,7], [4], [1,-1], [1,1], [1])
    
    # Block 3: A_left (2x2) * B_top2_right (2x1) -> C_top2_right (2x1)
    # B_top2_right is col 4.
    # B indices: B[0,4]=4, B[1,4]=9
    # C indices: C[0,4]=4, C[1,4]=9 -> W[8], W[9]
    
    # A00=0, A01=1, A10=3, A11=4
    # B0=4, B1=9
    # C0=8, C1=9
    
    # Schoolbook for 2x2x1:
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Rank 4.
    
    # C0 term 1
    add_block([0], [4], [8], [1], [1], [1])
    # C0 term 2
    add_block([1], [9], [8], [1], [1], [1])
    # C1 term 1
    add_block([3], [4], [9], [1], [1], [1])
    # C1 term 2
    add_block([4], [9], [9], [1], [1], [1])
    
    # Block 4: A_right (2x1) * B_bottom (1x5) -> C_bottom (2x5)
    # A_right is col 2. A[0,2]=2, A[1,2]=5.
    # B_bottom is row 2. B[2,0]=10, ..., B[2,4]=14.
    # C is all 2x5.
    
    # C[i,k] = A[i,2] * B[2,k]
    # This is rank 1.
    # U[i] * V[k] * W[i,k]
    # U[0]=1, U[1]=1? No.
    # T[i, 2, k] = 1.
    # We can factor this as:
    # Sum_c U[i,c] V[2*k, c] W[i,k,c]
    # If we set U[i,0]=1, V[2*k,0]=1, W[i,k,0]=1, then sum is 1.
    # But V index is j*p+k. For j=2, indices are 10,11,12,13,14.
    # W index is k*n+i.
    
    # We need one column.
    # U[2]=1, U[5]=1.
    # V[10]=1, V[11]=1, V[12]=1, V[13]=1, V[14]=1.
    # W[0]=1 (i=0,k=0), W[2]=1 (i=0,k=1), W[4]=1 (i=0,k=2), W[6]=1 (i=0,k=3), W[8]=1 (i=0,k=4)
    # W[1]=1 (i=1,k=0), W[3]=1 (i=1,k=1), W[5]=1 (i=1,k=2), W[7]=1 (i=1,k=3), W[9]=1 (i=1,k=4)
    
    # Wait, does this conflict with previous W assignments?
    # Yes, we are adding to the same W matrix.
    # The previous blocks contributed to W[0..9].
    # This block will add 1 to all W entries?
    # Let's check.
    # Block 4 contributes to all C[i,k].
    # So we add 1 to W[k*2+i] for all i,k.
    
    add_block(
        [2, 5], 
        [10, 11, 12, 13, 14], 
        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9], 
        [1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    
    # Trim U, V, W to exact dimensions
    U = U[:U_dim]
    V = V[:V_dim]
    W = W[:W_dim]
    
    # Convert to lists of lists of ints if possible
    U = [[int(x) if x == int(x) else x for x in row] for row in U]
    V = [[int(x) if x == int(x) else x for x in row] for row in V]
    W = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
