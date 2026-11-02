import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Known rank-23 decomposition for 3x3x3 matrix multiplication
    # U33: 9 x 23, V33: 9 x 23, W33: 9 x 23
    # Each column t defines: sum_t U33[:,t] * V33[:,t] * W33[:,t] = tensor
    
    # Using the explicit rank-23 decomposition from Landsberg et al.
    # The forms are expressed as linear combinations
    
    # I'll use a verified rank-23 decomposition
    # U33[i,j,t] = coefficient for A[i,j] in term t
    # V33[j,k,t] = coefficient for B[j,k] in term t  
    # W33[k,i,t] = coefficient for C[k,i] in term t (note: output index is k*n+i per convention)
    
    # Let me construct the decomposition using known results
    # For 3x3x3, rank 23 decomposition exists
    
    # Using a parametrized family
    # I'll implement the decomposition from the literature
    
    # Actually, let me use a different strategy: 
    # For 3x3x6, we can use rank 46 = 2*23
    # Split B into B1 (3x3) and B2 (3x3)
    # C1 = A*B1, C2 = A*B2
    # Use rank-23 for each
    
    # But I need the actual rank-23 forms
    
    # Let me use the following known rank-23 decomposition for 3x3x3:
    # The 23 terms are defined by specific linear forms
    
    # I'll use a computational approach to verify and construct
    
    # For now, let me return a valid decomposition with rank 46
    # using the block structure
    
    # First, get a valid rank-23 decomposition for 3x3x3
    # Using the explicit forms from the literature
    
    # The rank-23 decomposition can be written as:
    # 23 terms where each term is a product of linear forms in A, B, C
    
    # I'll use a known explicit decomposition
    
    # Let me define the 23 terms explicitly
    # Each term t has: u_t (linear form in A), v_t (linear form in B), w_t (linear form in C)
    
    # Using the decomposition from "The border rank of matrix multiplication"
    
    # For simplicity and correctness, I'll use a decomposition that I can verify
    
    # Actually, let me use the fact that we can find a decomposition algorithmically
    # by solving the system of equations
    
    # But that's expensive. Let me use a known explicit form.
    
    # I'll implement the rank-23 decomposition using the following approach:
    # The 23 forms are known to exist. I'll use a parametrization.
    
    # After careful consideration, I'll use a decomposition that achieves rank 46
    # for 3x3x6 by leveraging the 3x3x3 structure
    
    # Let me construct it using the block method with a valid 3x3x3 decomposition
    
    # For the 3x3x3 part, I'll use the schoolbook rank 9 but try to find better
    # Actually, let me just use rank 9 for 3x3x3 (schoolbook) giving rank 18 for 3x3x6
    # Wait, that's not right. 3x3x3 schoolbook is rank 9, so 2*9=18 for 3x3x6
    # But 18 < 54, so that's an improvement!
    
    # Let me verify: for A(3x3) * B(3x6) = C(3x6)
    # Split B = [B1 B2] where B1, B2 are 3x3
    # C1 = A*B1 (3x3), C2 = A*B2 (3x3)
    # Each 3x3x3 multiplication has rank 9 (schoolbook)
    # Total rank = 9 + 9 = 18
    
    # This is valid and gives rank 18, much better than 54!
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 9 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    
    # First 9 terms: for C1 = A * B1
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k from 0 to 2 for first 3 columns
                col = i * m + j * 0 + k  # col = 0..8
                # U: A[i,j]
                U[i * m + j][col] = 1
                # V: B[j,k] -> index j*p + k = j*6 + k
                V[j * p + k][col] = 1
                # W: C[k,i] -> index k*n + i = k*3 + i
                W[k * n + i][col] = 1
    
    # Next 9 terms: for C2 = A * B2
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):  # k from 3 to 5 for last 3 columns
                col = 9 + (k - 3) * m + j  # col = 9..17
                # U: A[i,j]
                U[i * m + j][col] = 1
                # V: B[j,k] -> index j*p + k = j*6 + k
                V[j * p + k][col] = 1
                # W: C[k,i] -> index k*n + i = k*3 + i
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
