import numpy as np

def construct():
    # Dimensions
    n, m, p = 4, 5, 6
    
    # We will build the decomposition by combining smaller blocks.
    # Rank of 2x2x3 is 11
    # Rank of 2x3x3 is 13
    # Rank of 2x5x3 is 11 + 13 = 24
    # Rank of 2x5x6 is 24 + 24 = 48
    # Rank of 4x5x6 is 48 + 48 = 96
    
    # Helper to create identity-like blocks for specific indices
    # This is complex to do manually, so I will implement the specific decompositions.
    
    # Decomposition for 2x2x3 (Rank 11)
    # A(2x2) B(2x3) = C(2x3)
    # We can view this as A * [B1(2x2) B2(2x1)]
    # C1 = A * B1 (2x2x2, Rank 7)
    # C2 = A * B2 (2x2x1, Rank 4)
    # Total Rank 11
    
    # Decomposition for 2x3x3 (Rank 13)
    # A(2x3) B(3x3) = C(2x3)
    # A = [A1(2x2) A2(2x1)]
    # B = [[B11(2x2) B12(2x1)], [B21(1x2) B22(1x1)]]
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    # Term 1: A1 B11 (2x2x2, Rank 7)
    # Term 2: A2 B21 (2x1x2, Rank 2)
    # Term 3: A1 B12 (2x2x1, Rank 2)
    # Term 4: A2 B22 (2x1x1, Rank 2)
    # Total Rank 13

    # I will implement a generic combiner that takes decompositions of sub-problems
    # and merges them into the full problem.
    
    # However, writing a full symbolic algebra system is too long.
    # I will hardcode the rank 96 decomposition by constructing the U, V, W matrices
    # based on the block structure.
    
    rank = 96
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # We need to fill U, V, W such that:
    # sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
    
    # This is equivalent to providing the 96 rank-1 tensors.
    # I will generate them by iterating through the block structure.
    
    # Block 1: A_top (2x5) * B_left (5x3) -> C_TL (2x3)
    # Block 2: A_top (2x5) * B_right (5x3) -> C_TR (2x3)
    # Block 3: A_bot (2x5) * B_left (5x3) -> C_BL (2x3)
    # Block 4: A_bot (2x5) * B_right (5x3) -> C_BR (2x3)
    
    # Each block is 2x5x3 with rank 24.
    # Total 4 * 24 = 96.
    
    # I need the explicit rank-24 decomposition for 2x5x3.
    # 2x5x3 = (2x2x3) + (2x3x3)
    
    # Let's define the sub-decompositions.
    
    # Sub-decomp 1: 2x2x3 (Rank 11)
    # U1 (4x11), V1 (6x11), W1 (6x11)
    # Sub-decomp 2: 2x3x3 (Rank 13)
    # U2 (6x13), V2 (9x13), W2 (6x13)
    
    # Combining 2x2x3 and 2x3x3 to 2x5x3:
    # A (2x5) = [A1(2x2) A2(2x3)]
    # B (5x3) = [B1(2x3); B2(3x3)]
    # C (2x3) = C1 + C2
    # C1 = A1 B1
    # C2 = A2 B2
    
    # For C1 (2x2x3):
    # U1 indices: 0,1,2,3 (A1)
    # V1 indices: 0,1,2,3,4,5 (B1)
    # W1 indices: 0..5 (C1)
    
    # For C2 (2x3x3):
    # U2 indices: 0,1,2,3,4,5 (A2) -> mapped to 4,5,6,7,8,9 in A
    # V2 indices: 0..8 (B2) -> mapped to 6..14 in B
    # W2 indices: 0..5 (C2) -> mapped to 0..5 in C (same as C1)
    
    # So for 2x5x3:
    # U is 10x24.
    # V is 15x24.
    # W is 6x24.
    
    # I will construct U, V, W for 2x5x3 first.
    
    # To save space and time, I will implement the 2x2x2 (Rank 7), 2x2x1 (Rank 4), etc.
    
    # Actually, I can just return the schoolbook decomposition for the sub-blocks if I can't easily get the low-rank ones.
    # But 11 and 13 are known.
    
    # Let's implement the Rank 7 (Strassen) for 2x2x2.
    # U, V, W for 2x2x2.
    
    # This is getting long. I will write the code to construct the Rank 96 solution.
    
    # Since I cannot easily verify the exact coefficients for Strassen in this short format, 
    # and the risk of error is high, I might fall back to a slightly higher rank 
    # that is easier to construct, or just use the schoolbook for sub-blocks 
    # which gives Rank 120.
    
    # Wait, if I use schoolbook for sub-blocks:
    # 2x2x3 -> 12
    # 2x3x3 -> 18
    # 2x5x3 -> 30
    # 4x5x6 -> 120.
    
    # So I MUST use the low-rank decompositions to improve the score.
    
    # I will hardcode the Strassen decomposition for 2x2x2.
    
    # Strassen 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # I will create a function to generate the U, V, W for a given rank-1 term.
    
    # Given the complexity, I will output the Rank 96 solution.
    
    # I will implement the construction.
    
    pass # Placeholder for actual implementation in the edit block
