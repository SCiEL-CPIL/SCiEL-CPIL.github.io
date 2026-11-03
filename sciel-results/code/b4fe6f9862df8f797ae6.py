import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: Split B into 5 blocks of size 3x2.
    # Each block multiplication 2x3 * 3x2 has rank 9 (2 for 2x1*1x2 + 7 for 2x2*2x2 Strassen).
    # Total rank = 5 * 9 = 45.
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 * 2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for Strassen terms (relative to 2x2 block)
    # A indices: 0,1 ; B indices: 0,1
    # U terms (i,j)
    # V terms (j,k)
    # W terms (k,i)
    
    strassen_terms = [
        # (U_i, U_j, V_j, V_k, W_k, W_i, coeff)
        # M1
        (0, 0, 0, 0, 0, 0, 1), (1, 1, 0, 0, 0, 0, 1),
        (0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1),
        # M2
        (1, 0, 0, 0, 0, 1, 1), (1, 1, 0, 0, 0, 1, 1),
        # M3
        (0, 0, 0, 1, 1, 0, 1), (0, 0, 1, 1, 1, 0, -1),
        # M4
        (1, 1, 1, 0, 0, 1, 1), (1, 1, 0, 0, 0, 1, -1),
        # M5
        (0, 0, 1, 1, 1, 0, 1), (0, 1, 1, 1, 1, 0, 1),
        # M6
        (1, 0, 0, 0, 0, 1, 1), (1, 0, 0, 1, 0, 1, 1),
        (0, 0, 0, 0, 0, 0, -1), (0, 0, 0, 1, 0, 0, -1),
        # M7
        (0, 1, 1, 0, 1, 0, 1), (0, 1, 1, 1, 1, 0, 1),
        (1, 1, 1, 0, 1, 1, -1), (1, 1, 1, 1, 1, 1, -1)
    ]
    
    # Note: The above Strassen terms are for 2x2.
    # We need to map these to the global indices.
    
    # For each block b (0 to 4):
    # k indices are 2b, 2b+1.
    # j indices are 0, 1, 2.
    # i indices are 0, 1.
    
    # Part 1: j=0 part (2x1 * 1x2)
    # A[:, 0] * B[0, :]
    # Rank 2.
    # Term 1: i=0, j=0, k=2b
    # Term 2: i=1, j=0, k=2b
    # Term 3: i=0, j=0, k=2b+1
    # Term 4: i=1, j=0, k=2b+1
    # Actually, this is rank 2 for the whole 2x2 block?
    # A[:,0] is 2x1. B[0,:] is 1x2.
    # Product is 2x2.
    # Rank of outer product is 1.
    # So 1 term?
    # U[i,0,c] = 1, V[0,k,c] = 1, W[k,i,c] = 1.
    # This gives 1 for all i,k.
    # We need A[i,0] B[0,k].
    # So U[i,0,c] = A[i,0]? No, U is constant.
    # We can't use A[i,0] in U.
    # So we must use 2 terms?
    # Term 1: U[0,0,c]=1, V[0,2b,c]=1, W[2b,0,c]=1. (Contributes to C[0,2b])
    # Term 2: U[1,0,c]=1, V[0,2b,c]=1, W[2b,1,c]=1. (Contributes to C[1,2b])
    # This works for k=2b.
    # Similarly for k=2b+1.
    # Total 4 terms for the j=0 part?
    # No, the rank of the 2x1 * 1x2 multiplication is 2.
    # C[0,2b] = A[0,0] B[0,2b]
    # C[1,2b] = A[1,0] B[0,2b]
    # C[0,2b+1] = A[0,0] B[0,2b+1]
    # C[1,2b+1] = A[1,0] B[0,2b+1]
    # This is 4 independent products.
    # So rank 4?
    # No, the rank of the tensor for 2x1 * 1x2 is 2.
    # T[i,j,k] = 1 if j=0.
    # T[i,0,k] = 1.
    # Rank of this tensor is 2?
    # T[i,0,k] = sum_r U[i,0,r] V[0,k,r] W[k,i,r].
    # If R=2, can we do it?
    # Yes.
    
    # Let's just use the schoolbook for the j=0 part to be safe.
    # Rank 4 for j=0 part (2 i's * 2 k's).
    # Rank 7 for j=1,2 part (Strassen).
    # Total 11 per block.
    # 5 blocks * 11 = 55.
    # This matches the known rank 55.
    
    # Let's implement rank 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Part 1: j=0 (Schoolbook for 2x1 * 1x2)
        # 4 terms
        for i in range(2):
            for k in [k0, k1]:
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        # Part 2: j=1,2 (Strassen for 2x2 * 2x2)
        # Map local indices (i_loc, j_loc, k_loc) to global (i, j, k)
        # i_loc in {0,1} -> i
        # j_loc in {0,1} -> j = 1 + j_loc
        # k_loc in {0,1} -> k = k0 + k_loc
        
        # Strassen terms for 2x2
        # We need to add 7 terms.
        # Each term is a sum of products.
        # We will expand the Strassen terms into rank-1 terms.
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00: i=0, j=1. A11: i=1, j=2.
        # B00: j=1, k=k0. B11: j=2, k=k1.
        
        # Term M1
        # U: (0,1), (1,2). V: (1,k0), (2,k1). W: (k0,0), (k1,1).
        # This is 4 rank-1 terms?
        # No, Strassen uses 7 multiplications.
        # Each multiplication is a rank-1 term in the decomposition of the 2x2x2 tensor.
        
        # Let's list the 7 Strassen multiplications explicitly as rank-1 terms.
        # M1 = (A00+A11)(B00+B11)
        # This is not a single rank-1 term. It is a product of sums.
        # In the tensor decomposition, each rank-1 term is a single product U*V*W.
        # Strassen reduces the number of such terms from 8 to 7.
        
        # The 7 terms are:
        # 1. (A00+A11)(B00+B11)
        # 2. (A10+A11)B00
        # 3. A00(B01-B11)
        # 4. A11(B10-B00)
        # 5. (A00+A01)B11
        # 6. (A10-A00)(B00+B01)
        # 7. (A01-A11)(B10+B11)
        
        # Each of these is a product of a linear form in A and a linear form in B.
        # So each is a rank-1 term in the A-B tensor.
        
        # Let's define the 7 terms.
        
        # Term 1: (A00+A11)(B00+B11)
        # A part: A00+A11. B part: B00+B11.
        # This is NOT a single rank-1 term in the i,j,k space.
        # It is a sum of 4 rank-1 terms?
        # No, the rank of the Strassen decomposition is 7.
        # This means there are 7 rank-1 terms in total for the 2x2x2 tensor.
        
        # Let's use the standard Strassen rank-1 decomposition.
        # It is known that the 2x2x2 tensor has rank 7.
        # The 7 terms are:
        # 1. U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1) ? No.
        
        # I will use the schoolbook for the 2x2 part as well to be safe, then optimize if needed.
        # Schoolbook for 2x2 is rank 8.
        # Total rank = 5 * (4 + 8) = 60.
        # This is the same as the original.
        
        # To get 55, we need rank 7 for the 2x2 part.
        # I will hardcode the 7 Strassen terms.
        
        # Strassen Terms (i,j,k) indices for 2x2 block (local)
        # 1. (0,0,0), (0,0,1), (1,1,0), (1,1,1) with coeffs 1?
        # No, the Strassen terms are:
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # This is 4 terms.
        
        # I will stick to rank 60 (schoolbook) if I can't easily construct rank 55.
        # But the knowledge says rank 55 is possible.
        
        # Let's just return the schoolbook rank 60 for now to ensure correctness.
        pass

    # Re-implement schoolbook correctly
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
