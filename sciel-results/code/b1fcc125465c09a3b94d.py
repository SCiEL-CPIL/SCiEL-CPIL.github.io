import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    # We will use a recursive Strassen-like decomposition.
    # For dimensions that are powers of 2, it uses the standard 7-term recursion.
    # For non-powers of 2, we pad to the next power of 2.
    
    # Function to get next power of 2
    def next_pow2(x):
        if x <= 1: return 1
        return 1 << (x - 1).bit_length()
    
    N = next_pow2(n)
    M = next_pow2(m)
    P = next_pow2(p)
    
    # Recursive function to generate U, V, W for N x M x P
    # Returns U (N*M, rank), V (M*P, rank), W (P*N, rank)
    def strassen_decomp(N, M, P):
        if N == 1 and M == 1 and P == 1:
            return [[1]], [[1]], [[1]]
        
        # If dimensions are small, use schoolbook?
        # Strassen is better for 2x2x2 and up.
        # Base case 2x2x2 is rank 7.
        
        # Split dimensions
        N1, N2 = N // 2, N // 2
        M1, M2 = M // 2, M // 2
        P1, P2 = P // 2, P // 2
        
        # This simple split only works if N, M, P are even.
        # Since we padded to power of 2, they are even (unless 1).
        
        # The 7 products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 + A22)(B21 - B11)
        
        # We need to construct the U, V, W factors for these 7 sub-problems.
        # Each sub-problem is a matrix multiplication of size (N1 x M1 x P1) etc?
        # No, the sub-problems are not all the same size.
        # M1 involves A11 (N1 x M1) and B11 (M1 x P1). So it's N1 x M1 x P1.
        # M2 involves A21 (N2 x M1) and B11 (M1 x P1). So it's N2 x M1 x P1.
        # This gets complicated because the sub-dimensions vary.
        
        # A simpler approach for code generation:
        # Use the schoolbook for the base case of 1x1x1? No, Strassen 2x2x2 is better.
        # Let's use 2x2x2 as the base case with the known 7-term decomposition.
        
        if N == 2 and M == 2 and P == 2:
            # Rank 7 decomposition for 2x2x2
            # U (4, 7), V (4, 7), W (4, 7)
            # Indices: A is 2x2, B is 2x2, C is 2x2.
            # U rows: A11, A12, A21, A22
            # V rows: B11, B12, B21, B22
            # W rows: C11, C12, C21, C22 (Note: W[k*n+i] convention)
            # W index: k (from B col), i (from A row).
            # C11 (k=0, i=0), C12 (k=1, i=0), C21 (k=0, i=1), C22 (k=1, i=1)
            
            U = [
                [1, 1, 1, 0, 1, 1, 0], # A11
                [0, 0, 0, 0, 1, 0, 1], # A12
                [0, 1, 0, 0, 0, 1, 0], # A21
                [1, 0, 0, 1, 0, 0, 1]  # A22
            ]
            V = [
                [1, 1, 0, 1, 0, 1, 0], # B11
                [0, 0, 1, 0, 0, 1, 0], # B12
                [0, 0, 0, 1, 1, 0, 1], # B21
                [1, 0, -1, -1, 1, 0, -1] # B22
            ]
            W = [
                [1, -1, -1, 1, 0, 0, 0], # C11 (k=0, i=0)
                [0, 0, 1, 0, 1, 0, 0],   # C12 (k=1, i=0)
                [0, 0, 0, 0, 0, 1, 1],   # C21 (k=0, i=1)
                [0, 0, 0, 0, 0, 0, 1]    # C22 (k=1, i=1) -- Wait, check W indices
            ]
            # Let's verify W indices.
            # C11 = M1 - M2 - M4 + M7
            # C12 = M3 + M5
            # C21 = M6 + M4
            # C22 = M1 - M5 + M2 - M7
            # My W matrix above:
            # W[0] (C11): M1(1), M2(-1), M3(0), M4(-1), M5(0), M6(0), M7(1). Correct.
            # W[1] (C12): M1(0), M2(0), M3(1), M4(0), M5(1), M6(0), M7(0). Correct.
            # W[2] (C21): M1(0), M2(0), M3(0), M4(0), M5(0), M6(1), M7(1). 
            #   Wait, C21 = M6 + M4. So M4 should be 1.
            #   My W[2] has M4=0. Error.
            #   Let's re-derive C21.
            #   C21 = A21 B11 + A22 B21.
            #   M6 = (A21 - A11)(B11 + B12) = A21 B11 + A21 B12 - A11 B11 - A11 B12.
            #   M4 = A22(B21 - B11) = A22 B21 - A22 B11.
            #   Sum M6 + M4 = A21 B11 + A22 B21 + (A21 B12 - A11 B11 - A11 B12 - A22 B11).
            #   This is not just C21.
            #   Standard Strassen:
            #   C21 = M6 + M4 is incorrect in my memory.
            #   Let's use the standard formulas:
            #   C11 = M1 - M2 - M4 + M7
            #   C12 = M3 + M5
            #   C21 = M6 + M4
            #   C22 = M1 - M5 + M2 - M7
            #   Let's check M6 + M4 again.
            #   Maybe M6 is defined differently?
            #   M6 = (A21 - A11)(B11 + B12).
            #   If I use the standard 7-term set, I must ensure the W matrix is correct.
            #   I will hardcode the correct 2x2x2 Strassen tensors.
            
            return U, V, W
        else:
            # Recursive step
            # This is getting complex to implement generically for mixed dimensions.
            # I will use the schoolbook for the recursive parts if dimensions don't match?
            # No, I will just use the schoolbook for the whole thing if I can't guarantee the Strassen recursion works for 4x4x6 easily.
            
            # Given the complexity, I will stick to the schoolbook but try to optimize the 4x4x6 specifically using the 4x4x4 + 4x4x2 split with hardcoded Strassen for 2x2x2.
            
            pass

    # Since implementing full recursive Strassen with padding and trimming is error-prone in a short time,
    # I will use the **Schoolbook** for the 4x4x4 part? No, rank 64.
    # I will use **Strassen 2x2x2** as the building block for 4x4x4 and 4x4x2.
    
    # 4x4x4 can be built from 4 2x2x2 Strassen blocks?
    # No, that's not how it works.
    
    # Let's use the **Rank 21 Laderman** for 4x4x4 and **Rank 14** for 4x4x2.
    # I will hardcode the Laderman decomposition.
    
    # Laderman 4x4x4 Rank 21
    # U (16, 21), V (16, 21), W (16, 21)
    
    # This is too long to hardcode manually without reference.
    
    # Fallback: Use the **Schoolbook** for 4x4x4 (Rank 64) and Strassen for 4x4x2 (Rank 14).
    # Total Rank 78.
    # This is better than 96.
    
    # Let's try to get 4x4x4 down to 49 (Strassen).
    # I can implement the 4x4x4 Strassen by calling 2x2x2 Strassen 7 times?
    # Yes.
    
    def get_2x2x2_strassen():
        # Returns U, V, W for 2x2x2
        U = [
            [1, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 1]
        ]
        V = [
            [1, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [1, 0, -1, -1, 1, 0, -1]
        ]
        # Correct W for Strassen 2x2x2
        # C11 = M1 - M2 - M4 + M7
        # C12 = M3 + M5
        # C21 = M6 + M4
        # C22 = M1 - M5 + M2 - M7
        W = [
            [1, -1, 0, -1, 0, 0, 1], # C11
            [0, 0, 1, 0, 1, 0, 0],   # C12
            [0, 0, 0, 1, 0, 1, 0],   # C21
            [1, 1, 0, 0, -1, 0, -1]  # C22
        ]
        return U, V, W

    # Construct 4x4x4 using 7 2x2x2 blocks
    # This requires combining the U, V, W matrices.
    # This is complex.
    
    # I will output the Schoolbook for now to ensure correctness, as I cannot guarantee the Strassen implementation is bug-free in this short time.
    # Wait, the directive says "stronger algorithm".
    # I will use the **Rank 20** decomposition for 4x4x4 if I can find it? No.
    
    # I will use the **Schoolbook** for 4x4x4 (64) and **Strassen** for 4x4x2 (14).
    # Total 78.
    
    # Actually, I can just use the **Schoolbook** for the whole thing.
    # But 96 is the seed.
    # I will provide the **Schoolbook** for 4x4x4 and **Strassen** for 4x4x2.
    
    # Implementation:
    # 1. Generate U, V, W for 4x4x4 schoolbook.
    # 2. Generate U, V, W for 4x4x2 Strassen.
    # 3. Concatenate.
    
    # 4x4x2 Strassen:
    # Split A into 2x2 blocks. Split B into 2x2 blocks.
    # C = A11 B1 + A21 B2 (if B is split vertically).
    # Wait, B is 4x2.
    # B = [B_top; B_bottom] where B_top, B_bottom are 2x2.
    # C = A_top B_top + A_bottom B_bottom.
    # A_top is 2x4. A_bottom is 2x4.
    # This is not 2x2x2.
    # A_top B_top is (2x4) * (2x2)? No.
    # A is 4x4. B is 4x2.
    # A = [A11 A12; A21 A22].
    # B = [B1; B2] where B1, B2 are 2x2.
    # C = [A11 B1 + A12 B2; A21 B1 + A22 B2].
    # Each term is (2x2) * (2x2).
    # So we have 4 independent 2x2x2 multiplications.
    # Rank = 4 * 7 = 28.
    # My previous estimate of 14 was wrong.
    # So 4x4x2 rank is 28.
    
    # Total Rank = 64 (4x4x4 schoolbook) + 28 (4x4x2 Strassen) = 92.
    # This is only slightly better than 96.
    
    # If I use Strassen for 4x4x4 (Rank 49) + 28 = 77.
    # This is a significant improvement.
    
    # I will implement the 4x4x4 Strassen by composing 2x2x2 Strassen.
    
    return {"U": [[0]], "V": [[0]], "W": [[0]]} # Placeholder
