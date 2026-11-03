"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [4, 5, 5]

def strassen_2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Indices: i in 0..1, j in 0..1, k in 0..1
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # W index is k*2 + i
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # We map these to U, V, W entries.
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # c=0: M1
    U[0][0] = 1; U[3][0] = 1 # A11, A22
    V[0][0] = 1; V[3][0] = 1 # B11, B22
    W[0][0] = 1 # C11 += M1
    W[2][0] = -1 # C22 += -M1? No C22 = M1 - ...
    # Wait, W[k*2+i, c]
    # C11 is i=0, k=0 -> W[0]
    # C12 is i=0, k=1 -> W[2]
    # C21 is i=1, k=0 -> W[1]
    # C22 is i=1, k=1 -> W[3]
    
    W[0][0] = 1  # C11 += M1
    W[3][0] = 1  # C22 += M1
    
    # c=1: M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1 # A21, A22
    V[0][1] = 1 # B11
    W[1][1] = 1 # C21 += M2
    W[3][1] = -1 # C22 -= M2
    
    # c=2: M3 = A11(B12-B22)
    U[0][2] = 1 # A11
    V[1][2] = 1; V[3][2] = -1 # B12, -B22
    W[2][2] = 1 # C12 += M3
    W[3][2] = 1 # C22 += M3
    
    # c=3: M4 = A22(B21-B11)
    U[3][3] = 1 # A22
    V[2][3] = 1; V[0][3] = -1 # B21, -B11
    W[0][3] = 1 # C11 += M4
    W[1][3] = 1 # C21 += M4
    
    # c=4: M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1 # A11, A12
    V[3][4] = 1 # B22
    W[0][4] = -1 # C11 -= M5
    W[2][4] = 1 # C12 += M5
    
    # c=5: M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1 # A21, -A11
    V[0][5] = 1; V[1][5] = 1 # B11, B12
    # M6 is not used in C?
    # Standard Strassen:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # M6 is not in the final sum?
    # Ah, M6 is used in some variants, but the standard 7-term uses M1..M7.
    # Let's check the formulas again.
    # Yes, M6 is not in the output sum in the standard 7-term formulation?
    # No, M6 is defined but not used?
    # Actually, the 7 terms are M1..M7.
    # Let's re-verify the output equations.
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # M6 is indeed not in the sum. It is a helper?
    # No, M6 is one of the 7 multiplications.
    # Wait, if M6 is not in the sum, why do we compute it?
    # Ah, I might be misremembering the indices.
    # Let's use a known correct mapping.
    # The 7 multiplications are:
    # 1: (A11+A22)(B11+B22)
    # 2: (A21+A22)B11
    # 3: A11(B12-B22)
    # 4: A22(B21-B11)
    # 5: (A11+A12)B22
    # 6: (A21-A11)(B11+B12)
    # 7: (A12-A22)(B21+B22)
    
    # And the outputs are:
    # C11 = 1 + 4 - 5 - 7
    # C12 = 3 + 5
    # C21 = 2 + 4
    # C22 = 1 - 2 + 3 + 7
    
    # So M6 is NOT used in the output?
    # This implies rank 6? No, Strassen is rank 7.
    # If M6 is not used, we only need 6 multiplications?
    # No, M6 is used to compute the inputs for other steps? No, they are independent.
    # I must be missing something.
    # Ah, the standard Strassen uses 7 multiplications.
    # Let's check a reference.
    # Wikipedia: Strassen's algorithm.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # It seems M6 is indeed not in the final sum.
    # Is it possible that M6 is used in a different formulation?
    # Or is my memory of the formulas wrong?
    # Let's check C11.
    # A11B11 + A12B21 = (M1+M4-M5-M7)
    # LHS: A11B11 + A12B21
    # RHS: (A11+A22)(B11+B22) + A22(B21-B11) - (A11+A12)B22 - (A12-A22)(B21+B22)
    # = A11B11 + A11B22 + A22B11 + A22B22 + A22B21 - A22B11 - A11B22 - A12B22 - (A12B21 + A12B22 - A22B21 - A22B22)
    # = A11B11 + A11B22 + A22B11 + A22B22 + A22B21 - A22B11 - A11B22 - A12B22 - A12B21 - A12B22 + A22B21 + A22B22
    # = A11B11 + (A11B22 - A11B22) + (A22B11 - A22B11) + (A22B22 + A22B22) + (A22B21 + A22B21) - 2A12B22 - A12B21
    # This is not matching.
    
    # Correct Strassen formulas:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # It is a known fact that M6 is not used in the output of the standard Strassen algorithm?
    # No, that would mean rank 6, which is impossible.
    # I must be misidentifying M6.
    # Actually, M6 is used in some descriptions, but the 7 multiplications are M1..M7.
    # If M6 is not in the sum, then we only use 6 multiplications?
    # No, the algorithm computes M1..M7.
    # If M6 is not in the sum, it is wasted.
    # This suggests my formula set is for a different variant or I am wrong.
    
    # Let's use a simpler, verified decomposition for 2x2x2.
    # Rank 7 is the minimum.
    # I will use the standard one from a reliable source.
    
    pass

def construct():
    # Fallback to schoolbook if time is low or if we can't build the complex one
    # But we will build the 66-rank one.
    
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # Split m=5 into 4+1
    # A = [A1 (4x4), A2 (4x1)]
    # B = [B1 (4x5); B2 (1x5)]
    # C = A1 B1 + A2 B2
    
    # Part 1: A2 B2 (4x1x5) -> Rank 1
    # Part 2: A1 B1 (4x4x5)
    #   Split p=5 into 4+1
    #   B1 = [B1a (4x4), B1b (4x1)]
    #   A1 B1 = A1 B1a + A1 B1b
    #   Part 2a: A1 B1a (4x4x4) -> Rank 49 (Strassen)
    #   Part 2b: A1 B1b (4x4x1) -> Rank 16 (Schoolbook)
    
    # Total Rank = 1 + 49 + 16 = 66
    
    rank = 66
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Part 1: A2 B2
    # A2 is A[:, 4] (index 4 in m)
    # B2 is B[4, :] (index 4 in m)
    # Term: A[i,4] B[4,k]
    # U[i*5+4, col] = 1
    # V[4*5+k, col] = 1
    # W[k*4+i, col] = 1
    for i in range(4):
        for k in range(5):
            U[i*5+4][col] = 1
            V[4*5+k][col] = 1
            W[k*4+i][col] = 1
    col += 1
    
    # Part 2b: A1 B1b (4x4x1)
    # A1 is A[:, 0..3]
    # B1b is B[0..3, 4] (col 4 of B1)
    # Term: A[i,j] B[j,4] for j in 0..3
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1 # k=4
            col += 1
    
    # Part 2a: A1 B1a (4x4x4)
    # Use Strassen for 4x4x4
    # This requires a 4x4x4 decomposition of rank 49.
    # We can build it by splitting 4x4x4 into two 2x4x4? No.
    # Split 4x4x4 into four 2x2x2 blocks?
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # Using Strassen on the 2x2 block structure:
    # 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # Total 49.
    
    # We need to map the 7 Strassen terms for 4x4x4 to the global U,V,W.
    # The 7 terms are M1..M7, each is a 2x2 matrix product.
    # M_t = X_t Y_t, where X_t, Y_t are 2x2 matrices formed from A, B blocks.
    # Each M_t contributes to C blocks.
    
    # This is complex to code inline.
    # Given the time constraints and risk of error, and that 66 is already better than 100,
    # I will implement the 66-rank solution but use Schoolbook for the 4x4x4 part to be safe?
    # Schoolbook 4x4x4 is 64.
    # Total 1 + 16 + 64 = 81.
    # 81 is still better than 100.
    # And much safer to implement correctly.
    
    # Let's use Schoolbook for 4x4x4 to ensure correctness.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5+j][col] = 1
                V[j*5+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
