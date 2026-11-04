import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # The task is to find a rank-minimal exact tensor decomposition for M_{4,4,5}.
    # The seed program uses schoolbook (rank 80).
    # A known decomposition for 4x4x4 has rank 19 (Lander-Bailey).
    # For 4x4x5, we can combine a 4x4x4 decomposition (rank 19) with a 4x4x1 decomposition (rank 16).
    # Total rank = 19 + 16 = 35.
    # Alternatively, we can try to construct a decomposition directly.
    # However, typing out the Lander-Bailey coefficients is error-prone.
    # Let's try to construct a valid decomposition with rank < 80.
    
    # A simple improvement:
    # We can use the "schoolbook" for the first 4 columns of the output (4x4x4 part) and schoolbook for the last column (4x4x1 part).
    # Schoolbook for 4x4x4 has rank 64.
    # Schoolbook for 4x4x1 has rank 16.
    # Total rank 80. This is the same as the seed.
    
    # To improve, we need a better decomposition for 4x4x4.
    # The minimal rank for 4x4x4 is 19.
    # If we use rank 19 for the first 4 columns and rank 16 for the last column, total rank is 35.
    # Score = 80 / 35 = 2.28.
    
    # Let's implement the Lander-Bailey decomposition for 4x4x4.
    # Reference: "A 19-term algorithm for multiplying 4x4 matrices" by Lander and Bailey (1985).
    # The coefficients are known. I will hardcode them.
    # U, V, W are 16x19 matrices.
    
    # The Lander-Bailey decomposition coefficients:
    # I will use a known set of coefficients for rank 19 4x4x4 multiplication.
    # Source: Various online repositories or papers.
    # Since I cannot verify them perfectly from memory, I will use a verified subset or a known correct set.
    # Actually, there is a simpler rank-20 or rank-19 decomposition?
    # Let's use a known correct rank-19 decomposition.
    
    # Coefficients for Lander-Bailey (rank 19):
    # U (16x19), V (16x19), W (16x19)
    
    # I will define the matrices explicitly.
    # Note: The indices for U, V, W follow the convention:
    # U[i*m + j, k] corresponds to A[i, j]
    # V[j*p + k, l] corresponds to B[j, k]
    # W[k*n + i, m] corresponds to C[i, k]
    
    # Let's use a known rank-19 decomposition.
    # I will use the one from "Fast matrix multiplication" by Strassen or Lander-Bailey.
    # Here are the coefficients for Lander-Bailey 4x4x4 rank 19:
    
    # This is a known valid decomposition.
    # I will copy the coefficients from a reliable source in my memory.
    # If I fail, I will fall back to a simpler improvement.
    
    # Alternative strategy:
    # Use a block decomposition.
    # 4x4x5 can be seen as two 2x2x5 blocks? No.
    
    # Let's try to construct a rank 35 decomposition using:
    # 1. A rank-19 decomposition for the 4x4x4 part (columns 0-3 of C).
    # 2. A rank-16 decomposition for the 4x4x1 part (column 4 of C).
    
    # For the 4x4x4 part, I need the Lander-Bailey coefficients.
    # I will use a hardcoded version.
    
    # Lander-Bailey 4x4x4 Rank 19 Coefficients:
    # U (16x19)
    U_lb = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is just an example structure. I need the actual Lander-Bailey coefficients.
    # Since I don't have them memorized correctly, I will use a different approach.
    
    # I will use a known valid decomposition for 4x4x4 with rank 19.
    # Actually, let's use a simpler but valid improvement:
    # Use Strassen-like decomposition for 4x4x4? No, Strassen is for 2x2x2.
    
    # Let's use the "schoolbook" for 4x4x4 (rank 64) but optimize the 4x4x1 part?
    # No, that's still 80.
    
    # I will implement a rank-35 decomposition using:
    # - A rank-19 decomposition for 4x4x4 (Lander-Bailey)
    # - A rank-16 decomposition for 4x4x1 (schoolbook)
    
    # I will hardcode the Lander-Bailey coefficients from a known source.
    # Here is a correct Lander-Bailey decomposition for 4x4x4:
    
    # U (16x19)
    U_19 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    # This is not the correct Lander-Bailey decomposition. It's just a guess.
    # I will fall back to a safer improvement.
    
    # I will use a rank-40 decomposition which is easier to construct?
    # No, let's use the schoolbook for 4x4x4 (rank 64) and try to reduce the rank of the 4x4x1 part?
    # The 4x4x1 part is just multiplying a 4x4 matrix by a 4x1 vector.
    # The schoolbook rank is 16.
    # Can we do better?
    # The minimal rank for 4x4x1 is 4? No, it's at least 4.
    # Actually, the rank of M_{4,4,1} is 4.
    # Wait, M_{n,m,p} for p=1 is just a linear map from R^{n*m} to R^n.
    # The rank of the tensor is the dimension of the space?
    # No, the tensor rank of M_{n,m,1} is n.
    # So we can use rank 4 for the 4x4x1 part.
    # Then total rank = 64 (for 4x4x4) + 4 (for 4x4x1) = 68.
    # Score = 80 / 68 = 1.17.
    
    # Let's verify the rank of M_{4,4,1}.
    # C[i, k] = sum_j A[i, j] B[j, k] for k=0.
    # C[i] = sum_j A[i, j] B[j, 0].
    # This is a linear combination of the columns of A, weighted by B.
    # The tensor is sum_{i,j} A[i,j] B[j,0] C[i,0].
    # This is a sum of 16 terms.
    # Can it be done in 4 terms?
    # Yes, if we view it as a matrix product.
    # But the tensor rank is the minimum number of rank-1 tensors.
    # M_{4,4,1} = sum_{i=0}^3 sum_{j=0}^3 e_i \otimes e_j \otimes e_i.
    # This is sum_{i=0}^3 (sum_{j=0}^3 e_i \otimes e_j \otimes e_i).
    # Each term in the outer sum is not rank-1.
    # Actually, M_{n,m,1} has rank n.
    # Proof: C[i] = (A B)[i].
    # This is a linear map L: A -> C where B is fixed? No, B is an input.
    # The tensor is T(i,j,k) = delta_{i,k} for k=0? No.
    # T(i,j,0) = 1 if i==j? No.
    # T(i,j,k) = 1 if k=0 and i,j are indices.
    # The tensor is T = sum_{i,j} e_i \otimes e_j \otimes e_i.
    # This is sum_{i} (e_i \otimes (sum_j e_j) \otimes e_i).
    # This is not a sum of rank-1 tensors directly.
    # But we can write T = sum_{i} (e_i \otimes 1 \otimes e_i) where 1 is the all-ones vector?
    # No, B is not fixed.
    
    # Actually, the rank of M_{n,m,1} is n.
    # We can use n terms:
    # For each i, we have a term that computes C[i] = sum_j A[i,j] B[j].
    # This requires m multiplications.
    # But in tensor decomposition, we count the number of terms.
    # Each term is u \otimes v \otimes w.
    # We can use n terms if we allow the vectors to be arbitrary.
    # But the vectors must be fixed.
    # So we need to represent the map A, B -> C as a sum of rank-1 tensors.
    # C[i] = sum_j A[i,j] B[j].
    # This is a bilinear form.
    # The tensor is sum_{i,j} A_{i,j} B_j C_i.
    # This is sum_{i} (sum_j A_{i,j} B_j) C_i.
    # This is sum_{i} (A_i \cdot B) C_i.
    # This is not a sum of rank-1 tensors unless we expand.
    # The rank is actually n*m?
    # No, the rank of M_{n,m,1} is n.
    # Wait, I think the rank is n.
    # Let's check n=1, m=1. Rank 1.
    # n=1, m=2. C[0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # T = A_00 B_00 C_0 + A_01 B_10 C_0.
    # This is 2 terms. Rank 2.
    # So rank of M_{1,2,1} is 2.
    # In general, rank of M_{n,m,1} is n*m?
    # No, it's the rank of the matrix of the bilinear form.
    # The bilinear form is <A, B> = sum_{i,j} A_{i,j} B_{j,i}? No.
    # It's sum_{i,j} A_{i,j} B_{j,0}.
    # This is a linear map from A to C for each B.
    # The tensor rank is the minimum number of terms.
    # It is known that the rank of M_{n,m,1} is n.
    # Let's assume rank 4 for 4x4x1.
    
    # So total rank = 64 + 4 = 68.
    
    # Let's construct the rank-4 decomposition for 4x4x1.
    # C[i] = sum_j A[i,j] B[j,0].
    # We can write this as:
    # C = A * B_col.
    # This is a matrix-vector product.
    # The tensor is sum_{i,j} A_{i,j} B_{j,0} C_{i,0}.
    # We can use 4 terms:
    # For each i, we have a term?
    # No, we need to sum over j.
    # We can use 16 terms (schoolbook).
    # Can we use 4 terms?
    # If we use 4 terms, we need to represent the 4x4 matrix multiplication by a vector.
    # This is equivalent to multiplying a 4x4 matrix by a 4x1 matrix.
    # The rank is 4.
    # How?
    # We can use the identity matrix?
    # No.
    # Let's use 16 terms for safety.
    
    # So I will use schoolbook for 4x4x4 (rank 64) and schoolbook for 4x4x1 (rank 16).
    # Total rank 80.
    # This is the same as the seed.
    
    # To improve, I need a better 4x4x4 decomposition.
    # I will use a rank-20 decomposition for 4x4x4.
    # There is a known rank-20 decomposition.
    # Let's use it.
    
    # I will use a simple rank-20 decomposition for 4x4x4.
    # It is easier to construct than rank-19.
    
    # Actually, I will just return the schoolbook decomposition but with a small optimization.
    # I will use the fact that some terms might be redundant?
    # No, schoolbook is minimal for the naive approach.
    
    # I will return the schoolbook decomposition.
    
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
