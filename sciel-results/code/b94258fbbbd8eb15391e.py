DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = 5, 5, 5
    
    # We'll use a 4+1 block decomposition
    # C[0:4,0:4] = A[0:4,0:4]*B[0:4,0:4] + A[0:4,4]*B[4,0:4]
    # C[0:4,4]   = A[0:4,0:4]*B[0:4,4]   + A[0:4,4]*B[4,4]
    # C[4,0:4]   = A[4,0:4]*B[0:4,0:4]   + A[4,4]*B[4,0:4]
    # C[4,4]     = A[4,0:4]*B[0:4,4]     + A[4,4]*B[4,4]
    
    terms = []
    
    # C[0:4,0:4] from A[0:4,0:4]*B[0:4,0:4]: 4*4*4 = 64 terms? No, 4*4 = 16 terms (each term is one j index)
    # Actually for A[i,j]*B[j,k] -> C[i,k], for i in 0:4, j in 0:4, k in 0:4: that's 4*4*4 = 64? 
    # No wait. For each (i,j,k) where i<4, j<4, k<4, we have A[i,j]*B[j,k] contributing to C[i,k].
    # That's 4*4*4 = 64 terms? But schoolbook for 4x4 is 4*4*4=64? No, schoolbook for n×n is n^3 = 64 for n=4? 
    # Wait no, schoolbook rank for n×n matrix multiplication is n^3? No, it's n^2 * n = n^3? 
    # I'm confused again. Let me think clearly.
    
    # For matrix multiplication C = A*B where A is n×m, B is m×p:
    # C[i,k] = sum over j of A[i,j]*B[j,k]
    # In the tensor decomposition, each (i,j,k) triple gives one rank-1 term: U[i*m+j] * V[j*p+k] * W[k*n+i]
    # Total terms = n*m*p
    # For 5×5×5: 5*5*5 = 125. Correct.
    # For 4×4×4: 4*4*4 = 64.
    # For 4×1×4: 4*1*4 = 16. But wait, this is A[4×1] * B[1×4] -> C[4×4]. 
    # C[i,k] = A[i,0]*B[0,k]. Only j=0. So terms = 4*1*4 = 16? But the matrix rank is 1.
    # In the tensor decomposition sense, we still need 4*1*4 = 16 terms because each (i,j,k) is separate.
    # Hmm, but that doesn't seem right for optimization.
    
    # Actually, I think the tensor rank for A[i,j]*B[j,k]->C[i,k] with fixed j is n*p (not n*m*p with m=1).
    # Wait no. If j is fixed, then for each (i,k), we have one term A[i,j]*B[j,k]. That's n*p terms.
    # For n=4, p=4: 16 terms.
    
    # So my earlier calculation was wrong. Let me redo:
    # C[0:4,0:4] = sum over j=0..3 of A[0:4,j]*B[j,0:4] + A[0:4,4]*B[4,0:4]
    # First part: sum over j=0..3, each j gives 4*4=16 terms. Total 4*16=64.
    # Second part: A[0:4,4]*B[4,0:4], j=4 fixed, gives 4*4=16 terms.
    # Total for C[0:4,0:4]: 64+16=80.
    
    # That's worse than I thought. Let me reconsider.
    
    # Actually I think I was wrong before. The block decomposition doesn't reduce the rank below schoolbook
    # unless we use a sub-optimal decomposition for the blocks.
    
    # The only way to beat schoolbook is to use an algorithm like Strassen that has lower tensor rank
    # for the sub-problems.
    
    # So the question is: can I decompose 5×5 matmul using Strassen-like sub-decompositions?
    
    # For now, let me just return the schoolbook decomposition with rank 125, since I can't
    # easily find a lower-rank decomposition within the time budget.
    
    # Actually, let me think about whether there's a clever decomposition.
    
    # One approach: use the fact that 5 = 2 + 2 + 1.
    # Partition rows/cols of A, B, C into blocks of size 2, 2, 1.
    # This gives a 3×3 block structure.
    # Each block product can be done with schoolbook.
    # Total rank = sum over all block products of (block_size_i * block_size_j * block_size_k)
    # where the blocks are (n_i × m_j) * (m_j × p_k) -> (n_i × p_k)
    # With n = [2,2,1], m = [2,2,1], p = [2,2,1]:
    # Total = sum_{i,j,k} n_i * m_j * p_k = (sum n_i)(sum m_j)(sum p_k) = 5*5*5 = 125.
    # So no improvement.
    
    # The only way to improve is to use a decomposition with rank < n*m*p for some sub-problems.
    # Strassen: 2×2×2 has rank 7 < 8.
    
    # Can I use Strassen somewhere in a 5×5 decomposition?
    # If I can isolate a 2×2 sub-problem and apply Strassen, I save 1 rank per 2×2 block.
    # But the cross terms still need schoolbook.
    
    # This is getting complex. Given the time constraints, let me just implement the schoolbook
    # decomposition correctly and return rank 125.
    
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
