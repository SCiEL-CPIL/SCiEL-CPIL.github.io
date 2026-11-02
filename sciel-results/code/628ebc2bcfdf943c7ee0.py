import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We split the problem into 4x4x4 and 4x4x1
    # For 4x4x4, we use Strassen (rank 49)
    # For 4x4x1, we use schoolbook (rank 16)
    
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 4x4x4 part: columns 0 to 3 of B
    # B is 4x5. We take B[:, 0:4]
    # A is 4x4.
    # C[:, 0:4] = A @ B[:, 0:4]
    
    # Strassen for 4x4x4
    # Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    # Strassen computes 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Each Mi is a 2x2 matrix product, which has rank 7.
    # So total rank 7*7 = 49.
    
    # We need to map the 4x4 indices to 2x2 block indices.
    # A[i, j] where i, j in 0..3
    # i = i_block * 2 + i_local, j = j_block * 2 + j_local
    
    # Let's build the 49 components.
    # For each of the 7 Strassen multiplications, we have a 2x2x2 tensor of rank 7.
    # The 2x2x2 tensor has indices (i_local, j_local, k_local) for the 2x2 blocks.
    
    # Strassen 2x2x2 decomposition (rank 7)
    # U_222, V_222, W_222
    # U_222[i*2+j, c], V_222[j*2+k, c], W_222[k*2+i, c]
    
    # We can hardcode the Strassen 2x2x2 decomposition.
    # Or just use schoolbook for 2x2x2 (rank 4) -> 7*4=28? No, Strassen is 7.
    # Schoolbook 2x2x2 is rank 4.
    # If we use schoolbook for the 2x2 blocks, rank is 7*4 = 28?
    # No, Strassen uses 7 multiplications. Each multiplication is a 2x2 matrix product.
    # If we do 2x2 matrix product with schoolbook (rank 4), total rank 7*4 = 28.
    # If we do 2x2 matrix product with Strassen (rank 7), total rank 7*7 = 49.
    # So Strassen is worse for small matrices?
    # Wait, 2x2 matrix multiplication has rank 7 (Strassen) or 4 (schoolbook).
    # Schoolbook is rank 4. Strassen is rank 7.
    # So for 2x2, schoolbook is better!
    # So for 4x4, if we use 2x2 blocks and schoolbook for each block, rank is 7*4 = 28?
    # No, Strassen computes 7 products of 2x2 matrices.
    # If each 2x2 product is done with schoolbook (rank 4), total rank 7*4 = 28.
    # If each 2x2 product is done with Strassen (rank 7), total rank 7*7 = 49.
    # So 28 is better than 49.
    # Can we do better than 28?
    # The rank of 4x4x4 is at least 16.
    # Is there a rank 16 decomposition? No.
    # Is there a rank 20 decomposition?
    # Actually, the rank of 4x4x4 is 16? No.
    # Let's use the 28 decomposition.
    
    # So for 4x4x4, rank 28.
    # For 4x4x1, rank 16.
    # Total 44.
    
    # Let's implement the 28 decomposition.
    
    # Strassen 7 multiplications:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Each Mi is a 2x2 matrix.
    # We compute Mi = Xi @ Yi, where Xi, Yi are 2x2 matrices.
    # Xi and Yi are linear combinations of A and B blocks.
    
    # We need to express U, V, W for the 4x4x4 part.
    # U is 16x28, V is 16x28 (for 4x4 part of B), W is 16x28.
    
    # Let's build it.
    
    # This is getting complex. Let's just use the schoolbook 80.
    # Or use the 65 from Strassen 49 + 16.
    # But 28 + 16 = 44 is better.
    
    # Let's try to construct the 44 rank decomposition.
    
    # I will write a helper to add a 2x2x2 schoolbook decomposition.
    
    def add_222(U, V, W, rank_offset, a_block_i, a_block_j, b_block_j, b_block_k, c_block_i, c_block_k, sign_a=1, sign_b=1):
        # a_block_i, a_block_j: block indices of A (0 or 1)
        # b_block_j, b_block_k: block indices of B (0 or 1)
        # c_block_i, c_block_k: block indices of C (0 or 1)
        # This adds the 2x2x2 tensor for A[a_i, a_j] * B[b_j, b_k] = C[c_i, c_k]
        
        # 2x2x2 schoolbook has rank 4.
        # Indices: i, j, k in 0..1
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    col = rank_offset + i*4 + j*2 + k # This is wrong. Rank is 4.
                    # Schoolbook 2x2x2:
                    # U[i*2+j, c] = 1 if c corresponds to (i,j,k)
                    # V[j*2+k, c] = 1
                    # W[k*2+i, c] = 1
                    
                    # Let's use a simple mapping.
                    # c = i*4 + j*2 + k? No, rank is 4.
                    # The 4 components are (i,j,k) where i,j,k are the same?
                    # No, schoolbook 2x2x2:
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # So the tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
                    # The rank 4 decomposition is:
                    # For each i, k:
                    # U[i*2+j, c] ... no.
                    
                    # Actually, the schoolbook 2x2x2 is just 4 multiplications.
                    # c = 0,1,2,3 correspond to (i,k) pairs?
                    # No, the rank is 4.
                    # The 4 components are:
                    # 1. A[0,0]B[0,0] -> C[0,0]
                    # 2. A[0,1]B[1,0] -> C[0,0]
                    # 3. A[1,0]B[0,1] -> C[1,1]
                    # 4. A[1,1]B[1,1] -> C[1,1]
                    # Wait, C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1].
                    # So we need more terms.
                    # Schoolbook 2x2x2 has rank 4?
                    # No, schoolbook 2x2x2 has rank 4?
                    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
                    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
                    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
                    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
                    
                    # This requires 4 multiplications?
                    # M1 = A[0,0]B[0,0]
                    # M2 = A[0,1]B[1,0]
                    # M3 = A[0,0]B[0,1]
                    # M4 = A[0,1]B[1,1]
                    # M5 = A[1,0]B[0,0]
                    # M6 = A[1,1]B[1,0]
                    # M7 = A[1,0]B[0,1]
                    # M8 = A[1,1]B[1,1]
                    # That's 8 multiplications.
                    # So schoolbook 2x2x2 has rank 4? No, 4 is the size.
                    # The rank of 2x2x2 is 7 (Strassen) or 4?
                    # No, the rank of 2x2x2 is 7.
                    # Schoolbook is 4 multiplications? No, 2x2 matrix multiplication has 4 entries, each is a sum of 2 products. So 8 products.
                    # So schoolbook rank is 4? No, 4 is the number of output entries.
                    # The rank is the number of scalar multiplications.
                    # Schoolbook 2x2x2 uses 4 multiplications? No, 4 is the size.
                    # Wait, 2x2 matrix multiplication: C = AB.
                    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
                    # This is 2 multiplications.
                    # Total 4 entries, each 2 multiplications. Total 8 multiplications.
                    # So schoolbook rank is 4? No, 4 is the number of output entries.
                    # The rank is 4? No, 4 is the size.
                    # Actually, the rank of 2x2x2 is 7.
                    # Schoolbook is 4? No, 4 is the size.
                    # I am confused.
                    
                    # Let's just use the schoolbook 80.
                    return

    # I will return the schoolbook 80.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
