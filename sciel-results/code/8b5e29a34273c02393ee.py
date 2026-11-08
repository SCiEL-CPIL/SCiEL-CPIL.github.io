import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (15, rank)
    # W: (p*n, rank) = (15, rank)
    # Convention: W[k*n+i, r]

    # We use a block decomposition approach.
    # Split A (3x3) into:
    #   A = [[A11, A12],
    #        [A21, A22]]
    # where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    
    # Split B (3x5) into:
    #   B = [[B11, B12],
    #        [B21, B22]]
    # where B11 is 2x5, B12 is 2x1, B21 is 1x5, B22 is 1x1.
    # Wait, the split for B should be consistent with the column index of A.
    # A is 3x3, so columns are 0,1,2.
    # Let's split the middle index j into {0,1} and {2}.
    # So A11 = A[0:2, 0:2] (2x2), A12 = A[0:2, 2] (2x1)
    # A21 = A[2, 0:2] (1x2), A22 = A[2, 2] (1x1)
    
    # B is 3x5. Rows correspond to j.
    # B11 = B[0:2, :] (2x5), B12 = B[2, :] (1x5)
    # Wait, B is m x p = 3x5.
    # The product is A (3x3) * B (3x5) = C (3x5).
    # C = A @ B.
    # C[0:2, :] = A11 @ B11 + A12 @ B21
    # C[2, :] = A21 @ B11 + A22 @ B21
    
    # Let's use Strassen for the 2x2 part (A11, B11) producing 2x5.
    # Strassen for 2x2 matrices:
    # M1 = (A11[0,0] + A11[1,1]) * (B11[0,0] + B11[1,1])  -> but B11 is 2x5, so we do this for each column?
    # Actually, we can treat the 2x5 matrix as a collection of 5 vectors of length 2.
    # Or, we can define the Strassen factors for the 2x2x2 tensor and apply them to the 5 columns.
    # The rank of 2x2x2 multiplication is 7.
    # So for the 2x2x5 block, the rank is 7 * 5 = 35? No, that's if we treat each column independently.
    # But Strassen works on the whole matrix.
    # Let's look at the tensor decomposition for 2x2x2.
    # The 7 factors are linear combinations of the 4 entries of A and 4 entries of B.
    # For 2x2x5, we have 4 entries in A (A11) and 10 entries in B (B11).
    # The output is 2x5 = 10 entries.
    # We can use the 7 Strassen factors, but each factor multiplies a 2x2 matrix by a 2x5 matrix?
    # No, Strassen factors are for 2x2 @ 2x2.
    # If we have 2x2 @ 2x5, we can think of it as 5 separate 2x2 @ 2x1 products? No.
    # We can think of it as one 2x2 @ 2x5 product.
    # The tensor is M_{i,j,k} = delta_{j,k} for i in 0..1, j in 0..1, k in 0..4.
    # This is not the standard Strassen tensor.
    # However, we can use the fact that 2x2 @ 2x5 is a special case.
    # Actually, the rank of 2x2 @ 2x5 is at most 2 * 7 = 14? Or 7 * 2 = 14?
    # Let's just use the schoolbook for the 2x2x5 part to be safe, or find a better decomposition.
    # Schoolbook for 2x2x5: 2*2*5 = 20.
    # Strassen-inspired: Maybe we can do better.
    # Let's stick to a simpler block approach first.
    
    # Alternative: Use the known rank for 3x3x3 which is 23 (Landsberg).
    # But we have 3x3x5.
    # Let's try to construct a rank 27 decomposition.
    # 3x3x5 can be seen as 3x3x3 + 3x3x2.
    # Rank(3x3x3) <= 23. Rank(3x3x2) <= 3*3*2 = 18. Total 41. Not good.
    
    # Let's try to use the "slice rank" or "tensor rank" bounds.
    # Actually, let's just implement a specific decomposition.
    # We will use the decomposition from the paper "The rank of the matrix multiplication tensor".
    # For 3x3x3, rank is 23.
    # For 3x3x5, we can pad the 5 with zeros to make it 3x3x9? No.
    
    # Let's try a different strategy.
    # We will use the schoolbook decomposition but with a clever grouping.
    # Actually, the simplest improvement is to use the rank of the 3x3x3 part (23) and add the extra 2 columns.
    # The extra 2 columns in B mean we are multiplying A (3x3) by B_extra (3x2).
    # This is a 3x3x2 multiplication.
    # Rank of 3x3x2 is at most 3*3*2 = 18.
    # Total rank 23 + 18 = 41. Still worse than 45.
    
    # Wait, the schoolbook rank is 45.
    # If we can get 3x3x3 to rank 23, and 3x3x2 to rank 12 (if possible), total 35.
    # Is rank(3x3x2) <= 12?
    # 3x3x2 is the same as 2x3x3.
    # Rank(2x3x3) is at most 2*3*3 = 18.
    # Can we do better?
    # Strassen for 2x2x2 is 7.
    # 2x3x3 can be seen as 2x2x3 + 2x1x3.
    # This is getting complicated.
    
    # Let's try to find a rank 30 decomposition.
    # We will use the following structure:
    # C = A @ B
    # We will define 30 rank-1 tensors.
    
    # Actually, let's just return the schoolbook for now, but with a smaller rank if possible.
    # The problem asks for "rank-minimal exact tensor decomposition".
    # The current program returns rank 45.
    # We need to beat 45.
    
    # Let's use the known rank 23 for 3x3x3.
    # We need to add the 2 extra columns.
    # The 3x3x3 part uses 23 factors.
    # The 3x3x2 part uses at most 18 factors.
    # Total 41.
    
    # Can we do better for 3x3x2?
    # 3x3x2: A is 3x3, B is 3x2, C is 3x2.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is a bilinear map.
    # We can use the fact that B has only 2 columns.
    # Let's try to use 9 factors for 3x3x2.
    # If we use 9 factors, total is 23 + 9 = 32.
    # Is rank(3x3x2) <= 9?
    # 3x3x2 is equivalent to 2x3x3.
    # Rank(2x3x3) is the rank of the tensor M_{i,j,k} = delta_{j,k} for i in 0..1, j in 0..2, k in 0..2.
    # This is the sum of two 1x3x3 tensors.
    # Rank(1x3x3) is 9? No, 1x3x3 is just a matrix, rank is 3 (if full rank).
    # Wait, the tensor is M_{i,j,k}.
    # For i=0, M_{0,j,k} = delta_{j,k}. This is the identity matrix I_3.
    # For i=1, M_{1,j,k} = delta_{j,k}. This is also I_3.
    # So the tensor is I_3 \otimes e_0 + I_3 \otimes e_1 = I_3 \otimes (e_0 + e_1).
    # This is a rank-1 tensor? No, I_3 is not rank-1.
    # I_3 can be decomposed into 3 rank-1 matrices.
    # So I_3 \otimes v can be decomposed into 3 rank-1 tensors.
    # So rank(2x3x3) <= 3.
    # Wait, is that right?
    # M_{i,j,k} = delta_{j,k}.
    # Sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{j,k}.
    # If we set W_{k,i,r} = 1 for all k,i,r, and U_{i,j,r} = delta_{i,r}, V_{j,k,r} = delta_{j,k} delta_{r,j}?
    # Let's try:
    # r = 0,1,2.
    # U_{i,j,r} = delta_{i,r} delta_{j,r}? No.
    # We want sum_r U_{i,j,r} V_{j,k,r} = delta_{j,k} (since W=1).
    # If U_{i,j,r} = delta_{i,r}, V_{j,k,r} = delta_{j,r} delta_{k,r}?
    # Sum_r delta_{i,r} delta_{j,r} delta_{k,r} = delta_{i,j} delta_{j,k}.
    # This gives delta_{i,j,k}, not delta_{j,k}.
    # We need sum_r U_{i,j,r} V_{j,k,r} = delta_{j,k}.
    # This is independent of i.
    # So we can set U_{i,j,r} = 1 for all i? No, U depends on i.
    # If U_{i,j,r} = delta_{i,0}, then sum_r delta_{i,0} V_{j,k,r} = delta_{i,0} sum_r V_{j,k,r}.
    # We need this to be delta_{j,k} for all i.
    # This is impossible if i != 0.
    # So W cannot be 1.
    
    # Let's look up the rank of 2x3x3.
    # It is known that rank(2x3x3) = 6? Or 9?
    # Actually, rank(2xn x n) is at most 2n.
    # For n=3, rank <= 6.
    # Let's assume rank(3x3x2) <= 6.
    # Then total rank is 23 + 6 = 29.
    
    # Let's try to construct a rank 29 decomposition.
    # We need the 23 factors for 3x3x3 and 6 factors for 3x3x2.
    
    # The 3x3x3 decomposition of rank 23 is complex.
    # Let's use a simpler approach.
    # We will use the schoolbook for 3x3x3 (rank 27) and 6 for 3x3x2.
    # Total 33.
    # 33 < 45. This is an improvement.
    
    # Let's implement rank 33.
    # 3x3x3 schoolbook: 27 factors.
    # 3x3x2: 6 factors.
    
    rank = 33
    U = np.zeros((9, rank))
    V = np.zeros((15, rank))
    W = np.zeros((15, rank))
    
    # 3x3x3 part (first 27 factors)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3 + j, col] = 1
                V[j*5 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
                
    # 3x3x2 part (next 6 factors)
    # We need to decompose the multiplication A (3x3) * B_extra (3x2) -> C_extra (3x2).
    # B_extra is B[:, 3:5].
    # C_extra[i, k] = sum_j A[i,j] B[j, 3+k].
    # We use 6 factors.
    # Let's try:
    # r = 0..5.
    # U_{i,j,r} = delta_{i, r//2} delta_{j, r%2}? No.
    # Let's use the fact that rank(3x3x2) <= 6.
    # One possible decomposition:
    # For each i in 0..2, and each k in 0..1, we have C[i,k] = sum_j A[i,j] B[j,3+k].
    # This is 6 outputs.
    # We can use 6 factors, one for each (i,k) pair?
    # No, that would be schoolbook.
    # Let's try:
    # r = 0..5.
    # U_{i,j,r} = delta_{i, r//2} * (1 if j == r%2 else 0)?
    # This doesn't seem right.
    
    # Let's just use schoolbook for the 3x3x2 part as well, but we only have 2 columns.
    # Schoolbook for 3x3x2 is 3*3*2 = 18 factors.
    # Total 27 + 18 = 45. No improvement.
    
    # We need a better decomposition for 3x3x2.
    # Let's use the fact that 3x3x2 is equivalent to 2x3x3.
    # Rank(2x3x3) is 6.
    # Let's find the 6 factors.
    # M_{i,j,k} = delta_{j,k} for i in 0..1, j in 0..2, k in 0..2.
    # We want sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{j,k}.
    # Let's try W_{k,i,r} = delta_{i, r//2} delta_{k, r%2}?
    # This is getting too complex to derive by hand.
    
    # Let's fall back to rank 45 for now, but with a different structure.
    # Actually, the task says "Find a rank-minimal exact tensor decomposition".
    # If I can't find a better one, I should return the schoolbook.
    # But the score is 1.0 for schoolbook.
    # I need to improve it.
    
    # Let's try rank 40.
    # We can use the fact that 3x3x5 can be decomposed into 3x3x3 + 3x3x2.
    # If we use rank 27 for 3x3x3 and rank 13 for 3x3x2, total 40.
    # Is rank(3x3x2) <= 13? Yes, 18 is the bound.
    
    # Let's just return the schoolbook.
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
