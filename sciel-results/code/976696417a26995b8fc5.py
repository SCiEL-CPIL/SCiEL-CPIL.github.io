import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    rank = 67
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Use Kronecker product of two Strassen 2x2 tensors
    # Strassen 2x2 factors
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map 2x2 Strassen to 4x4.
    # 4x4 is (2x2) x (2x2).
    # Let A4 = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # Let B4 = [[B00, B01], [B10, B11]] where Bij are 2x2.
    # C4 = A4 B4.
    # We can apply Strassen to the 2x2 block matrix, where each entry is a 2x2 matrix multiplication.
    # This requires 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 4 (schoolbook) or 7 (Strassen).
    # If we use Strassen for the inner 2x2, total rank 7*7=49.
    
    # Let's just use schoolbook for the inner 2x2 to keep it simple?
    # Rank 7 (outer Strassen) * 4 (inner Schoolbook) = 28.
    # This is much better than 49!
    # Let's verify:
    # Outer Strassen computes 7 terms, each is a 2x2 matrix product.
    # If we compute each 2x2 product with rank 4, total rank 28.
    # Is this valid? Yes, tensor rank is multiplicative under Kronecker product.
    # Rank(T1 x T2) = Rank(T1) * Rank(T2).
    # So Rank(T_Strassen_2x2 x T_Schoolbook_2x2) = 7 * 4 = 28.
    
    # So Part 1 Rank = 28.
    
    # Let's implement the 28 rank decomposition for 4x4x4.
    # Outer Strassen indices: 0..6
    # Inner Schoolbook indices: 0..3
    # Total 28 columns.
    
    # Strassen factors for 2x2:
    # U_strassen[i, k, c], V_strassen[k, j, c], W_strassen[j, i, c]
    # c in 0..6
    # i, k in 0..1 (for 2x2)
    
    # We map 4x4 indices (I, K, J) to (i, k_inner, j)
    # I = i * 2 + i_inner
    # K = k * 2 + k_inner
    # J = j * 2 + j_inner
    
    # U_4x4[I, K, c_outer * 4 + c_inner]
    
    # Strassen 2x2 definitions (U, V, W)
    # c=0: (A00+A11)(B00+B11) -> U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1?
    # Actually, let's just use the known rank-7 factors.
    
    # It's easier to just use the rank-49 Kronecker of two Strassen if we don't want to derive the mixed factors.
    # But 28 is better.
    
    # Let's stick to the 49 (Strassen x Strassen) for simplicity and correctness, as 28 requires careful index mapping.
    # Actually, 28 is easy.
    
    # Let's use Rank 49 for now to ensure correctness, then optimize.
    # 49 + 1 + 16 + 1 = 67.
    
    # If we use 28 for Part 1:
    # 28 + 1 + 16 + 1 = 46.
    
    # Let's implement Rank 46.
    
    # Part 1: 4x4x4 with Rank 28 (Strassen x Schoolbook)
    # Strassen 2x2 factors (U, V, W) for c in 0..6
    # We need the explicit 0/1 patterns.
    
    # U_strassen (2x2x7)
    # V_strassen (2x2x7)
    # W_strassen (2x2x7)
    
    # c=0: M1 = (A00+A11)(B00+B11)
    # A00, A11. B00, B11.
    # U[0,0]=1, U[1,1]=1. V[0,0]=1, V[1,1]=1.
    # W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1.
    
    # c=1: M2 = (A10+A11)B00
    # U[1,0]=1, U[1,1]=1. V[0,0]=1.
    # W[0,0]=1, W[1,0]=1.
    
    # c=2: M3 = A00(B01-B11)
    # U[0,0]=1. V[0,1]=1, V[1,1]=-1.
    # W[0,1]=1, W[1,1]=1.
    
    # c=3: M4 = A11(B10-B00)
    # U[1,1]=1. V[1,0]=1, V[0,0]=-1.
    # W[0,0]=1, W[1,0]=1.
    
    # c=4: M5 = (A00+A01)B11
    # U[0,0]=1, U[0,1]=1. V[1,1]=1.
    # W[0,1]=1, W[1,1]=1.
    
    # c=5: M6 = (A10-A00)(B00+B01)
    # U[1,0]=1, U[0,0]=-1. V[0,0]=1, V[0,1]=1.
    # W[0,0]=1, W[1,0]=1.
    
    # c=6: M7 = (A01-A11)(B10+B11)
    # U[0,1]=1, U[1,1]=-1. V[1,0]=1, V[1,1]=1.
    # W[0,1]=1, W[1,1]=1.
    
    # Inner Schoolbook 2x2 (Rank 4)
    # c_inner in 0..3
    # U_in[i, k, c_in] = 1 if i==c_in and k==c_in? No.
    # Schoolbook: U[i, k, c] = 1 if i==c and k==c?
    # No, schoolbook for 2x2:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Factors:
    # c=0: A00 B00. U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # c=1: A01 B10. U[0,1]=1, V[1,0]=1, W[0,1]=1.
    # c=2: A10 B00. U[1,0]=1, V[0,0]=1, W[1,0]=1.
    # c=3: A11 B10. U[1,1]=1, V[1,0]=1, W[1,1]=1.
    
    # Combined Rank 28 for 4x4x4.
    
    rank_part1 = 28
    for c_out in range(7):
        for c_in in range(4):
            r = col + c_out * 4 + c_in
            # Get Strassen factors
            # We need to define the Strassen U, V, W arrays explicitly.
            
            # This is getting long to hardcode.
            # Let's just use Rank 49 (Strassen x Strassen) to be safe and quick.
            # 49 + 1 + 16 + 1 = 67.
            
    # Let's use Rank 67.
    
    # Part 1: 4x4x4 Rank 49
    for c_out in range(7):
        for c_in in range(7):
            r = col + c_out * 7 + c_in
            # Map indices
            # I = i_out * 2 + i_in
            # K = k_out * 2 + k_in
            # J = j_out * 2 + j_in
            
            # We need Strassen factors for 2x2.
            # Let's define them as lists of tuples (i, k, val), etc.
            
            # This is too complex to generate dynamically in the loop without pre-defining.
            # Let's just use Schoolbook for 4x4x4? Rank 64.
            # 64 + 1 + 16 + 1 = 82.
            # 82 < 100.
            
    # Let's use Rank 82 (Schoolbook 4x4 + extras).
    
    col = 0
    # Part 1: A[0:4, 0:4] * B[0:4, 0:4]
    for i in range(4):
        for k in range(4):
            for j in range(4):
                U[i*5+k][col] = 1
                V[k*5+j][col] = 1
                W[j*4+i][col] = 1
                col += 1
    # col is now 64
    
    # Part 2: A[0:4, 4] * B[4, 0:4] (Rank 1)
    for i in range(4):
        for j in range(4):
            U[i*5+4][col] = 1
            V[4*5+j][col] = 1
            W[j*4+i][col] = 1
    col += 1
    # col is now 65
    
    # Part 3: A[0:4, 0:4] * B[0:4, 4] (Rank 16)
    for i in range(4):
        for k in range(4):
            U[i*5+k][col] = 1
            V[k*5+4][col] = 1
            W[4*4+i][col] = 1
            col += 1
    # col is now 65 + 16 = 81
    
    # Part 4: A[0:4, 4] * B[4, 4] (Rank 1)
    for i in range(4):
        U[i*5+4][col] = 1
        V[4*5+4][col] = 1
        W[4*4+i][col] = 1
    col += 1
    # col is now 82
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
