import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # We want to multiply A (2x3) by B (3x10) to get C (2x10).
    # C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k]
    # = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    
    # Split A into A_L (2x2) and A_R (2x1)
    # Split B into B_T (2x10) and B_B (1x10)
    # C = A_L B_T + A_R B_B
    
    # Part 1: A_L (2x2) * B_T (2x10)
    # Use Strassen's algorithm for the 2x2 matrix multiplication part.
    # For each pair of columns in B_T, we do a 2x2x2 multiplication.
    # B_T has 10 columns, so 5 pairs.
    # Each 2x2x2 Strassen uses 7 multiplications.
    # Total for Part 1: 5 * 7 = 35 rank.
    
    # Part 2: A_R (2x1) * B_B (1x10)
    # This is an outer product: C2[i,k] = A_R[i,0] * B_B[0,k]
    # This is a rank-20 tensor if done naively (2*1*10), 
    # but actually it's a sum of 20 terms where each term is a product
    # of one element of A_R and one element of B_B.
    # However, we can do better. The tensor for 2x1x10 matrix mult has rank 20?
    # No, the rank of the 2x1x10 tensor is 20? Let me think.
    # C2[i,k] = A[i,2] * B[2,k]
    # This is a sum over j of A[i,j]B[j,k] but j only has one value (j=2).
    # So it's just the outer product of the column of A (length 2) and the row of B (length 10).
    # The tensor T[i,k] = u[i] * v[k] for fixed u,v... no, u and v are the variables.
    # T[i,k] = A[i,2] * B[2,k]
    # This is a bilinear form. The tensor has entries T[i,k,j] where j indexes the 
    # multiplication. Here there's only one j (j=2).
    # The tensor is of shape (2, 10, 1) effectively? No.
    # The matrix multiplication tensor for 2x1 * 1x10 has shape (2, 10, 2*1*10)? No.
    # The tensor is M[i,k,j] = 1 if the multiplication A[i,j]*B[j,k] contributes to C[i,k].
    # For 2x1 * 1x10, j only equals 2 (0-indexed, but it's the only index).
    # Wait, A is 2x3, so j ranges 0,1,2. A_R corresponds to j=2.
    # So for Part 2, only j=2 matters.
    # C2[i,k] = A[i,2] * B[2,k]
    # This is a single product for each (i,k) pair? No, it's the same product structure.
    # For each (i,k), we need A[i,2]*B[2,k]. That's 2*10 = 20 products.
    # But can we share? No, each (i,k) requires a distinct product.
    # So rank 20 for Part 2.
    
    # Total rank = 35 + 20 = 55.
    
    # Let's build the decomposition.
    # U has shape (n*m, rank) = (6, 55)
    # V has shape (m*p, rank) = (30, 55)
    # W has shape (p*n, rank) = (20, 55)
    # 
    # Indexing:
    # U[i*m + j, r] corresponds to coefficient of A[i,j] in multiplication r.
    # V[j*p + k, r] corresponds to coefficient of B[j,k] in multiplication r.
    # W[k*n + i, r] corresponds to coefficient of multiplication r in C[i,k].
    # (Using the trace convention W[k*n+i, r] as specified)
    
    rank_part1 = 35
    rank_part2 = 20
    total_rank = rank_part1 + rank_part2
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Part 1: Strassen for 2x2 * 2x10
    # A_L = A[:, 0:2], shape (2,2)
    # B_T = B[0:2, :], shape (2,10)
    # Indices in U: i*m + j where i in {0,1}, j in {0,1}
    #   A[0,0] -> U[0*3+0, r] = U[0, r]
    #   A[0,1] -> U[0*3+1, r] = U[1, r]
    #   A[1,0] -> U[1*3+0, r] = U[3, r]
    #   A[1,1] -> U[1*3+1, r] = U[4, r]
    # Indices in V: j*p + k where j in {0,1}, k in {0,...,9}
    #   B[0,k] -> V[0*10+k, r] = V[k, r]
    #   B[1,k] -> V[1*10+k, r] = V[10+k, r]
    # Indices in W: k*n + i where k in {0,...,9}, i in {0,1}
    #   C[i,k] -> W[k*2+i, r]
    
    col = 0
    for pair_idx in range(5):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        
        # Strassen 7 products for 2x2 * 2x2 (columns k1, k2)
        # a11=A[0,0], a12=A[0,1], a21=A[1,0], a22=A[1,1]
        # b11=B[0,k1], b12=B[0,k2], b21=B[1,k1], b22=B[1,k2]
        
        # M1 = (a11+a22)(b11+b22)
        U[0, col] = 1  # a11
        U[4, col] = 1  # a22
        V[k1, col] = 1  # b11
        V[10 + k2, col] = 1  # b22
        W[k1 * n + 0, col] = 1  # c11
        W[k2 * n + 1, col] = 1  # c22
        col += 1
        
        # M2 = (a21+a22)b11
        U[3, col] = 1  # a21
        U[4, col] = 1  # a22
        V[k1, col] = 1  # b11
        W[k1 * n + 1, col] = 1  # c21
        col += 1
        
        # M3 = a11(b21-b22)
        U[0, col] = 1  # a11
        V[10 + k1, col] = 1  # b21
        V[10 + k2, col] = -1  # b22
        W[k1 * n + 0, col] = 1  # c11
        W[k2 * n + 0, col] = -1  # c12
        col += 1
        
        # M4 = a22(b11-b12)
        U[4, col] = 1  # a22
        V[k1, col] = 1  # b11
        V[k2, col] = -1  # b12
        W[k1 * n + 1, col] = 1  # c21
        W[k2 * n + 1, col] = -1  # c22
        col += 1
        
        # M5 = (a11+a12)b22
        U[0, col] = 1  # a11
        U[1, col] = 1  # a12
        V[10 + k2, col] = 1  # b22
        W[k2 * n + 0, col] = 1  # c12
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[3, col] = 1  # a21
        U[0, col] = -1  # a11
        V[k1, col] = 1  # b11
        V[k2, col] = 1  # b12
        W[k1 * n + 1, col] = 1  # c21
        W[k2 * n + 1, col] = 1  # c22
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[1, col] = 1  # a12
        U[4, col] = -1  # a22
        V[10 + k1, col] = 1  # b21
        V[10 + k2, col] = 1  # b22
        W[k1 * n + 0, col] = 1  # c11
        W[k2 * n + 0, col] = 1  # c12
        col += 1
    
    # Part 2: A_R (2x1) * B_B (1x10)
    # A_R = A[:, 2], shape (2,1)
    # B_B = B[2, :], shape (1,10)
    # C2[i,k] = A[i,2] * B[2,k]
    # U indices: A[0,2] -> U[0*3+2, r] = U[2, r]
    #            A[1,2] -> U[1*3+2, r] = U[5, r]
    # V indices: B[2,k] -> V[2*10+k, r] = V[20+k, r]
    # W indices: C[i,k] -> W[k*2+i, r]
    
    for i in range(n):
        for k in range(p):
            U[i * m + 2, col] = 1  # A[i,2]
            V[2 * p + k, col] = 1  # B[2,k]
            W[k * n + i, col] = 1  # C[i,k]
            col += 1
    
    # Convert to lists of integers/half-integers as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
