def construct():
    n, m, p = 2, 4, 6
    
    import numpy as np
    
    # We need to find a rank-7 decomposition of the 2x2x2x2 matrix multiplication tensor.
    # The tensor T has T[a_idx, b_idx, c_idx] = 1 where:
    #   a_idx = i*m + j (i in 0..1, j in 0..1) -> [0,1,2,3]
    #   b_idx = j*p + k (j in 0..1, k in 0..1) -> [0,1,2,3]  
    #   c_idx = k*n + i (k in 0..1, i in 0..1) -> [0,1,2,3]
    # with the constraint that the middle index j must match between a and b,
    # and the outer indices i,k must match between b and c appropriately.
    
    # Actually, for the 2x2x2x2 sub-problem:
    # A is 2x2, B is 2x2, C is 2x2
    # C[i,k] = sum_j A[i,j] B[j,k]
    # T[a_idx, b_idx, c_idx] = 1 if a_idx=(i,j), b_idx=(j,k), c_idx=(k,i) for some valid i,j,k
    # i.e., a_idx = i*2+j, b_idx = j*2+k, c_idx = k*2+i
    
    # The 8 nonzero entries of T:
    # (i,j,k) = (0,0,0): a=0, b=0, c=0
    # (i,j,k) = (0,0,1): a=0, b=1, c=2
    # (i,j,k) = (0,1,0): a=1, b=2, c=1
    # (i,j,k) = (0,1,1): a=1, b=3, c=3
    # (i,j,k) = (1,0,0): a=2, b=0, c=0
    # (i,j,k) = (1,0,1): a=2, b=1, c=2
    # (i,j,k) = (1,1,0): a=3, b=2, c=1
    # (i,j,k) = (1,1,1): a=3, b=3, c=3
    
    # So T[0,0,0]=1, T[0,1,2]=1, T[1,2,1]=1, T[1,3,3]=1,
    #    T[2,0,0]=1, T[2,1,2]=1, T[3,2,1]=1, T[3,3,3]=1
    # All other entries are 0.
    
    # We want T = sum_{r=1}^{7} U_r ⊗ V_r ⊗ W_r
    # where U_r, V_r, W_r are 4-dimensional vectors.
    
    # Let me use numpy to find this decomposition by solving a system of equations.
    # Actually, let me just hardcode the known Strassen decomposition.
    
    # The correct Strassen 2x2x2x2 tensor decomposition (rank 7):
    # Using the standard Strassen algorithm mapped to tensor factors:
    
    # A = [[a, b], [c, d]]  -> indices: a=0, b=1, c=2, d=3
    # B = [[e, f], [g, h]]  -> indices: e=0, f=1, g=2, h=3
    # C = [[C00, C01], [C10, C11]] -> W convention: W[k*n+i]
    #   C00: k=0,i=0 -> W[0]
    #   C10: k=0,i=1 -> W[1]  (wait, k*n+i = 0*2+1 = 1)
    #   C01: k=1,i=0 -> W[2]  (k*n+i = 1*2+0 = 2)
    #   C11: k=1,i=1 -> W[3]  (k*n+i = 1*2+1 = 3)
    
    # So W indices: [C00, C10, C01, C11]
    
    # Strassen products:
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # M2 = (c+d)e = ce + de
    # M3 = a(f-h) = af - ah
    # M4 = d(g-c) = dg - dc  -- NOT SEPARABLE!
    
    # The issue is that the standard Strassen algorithm doesn't directly give separable tensor factors.
    # Let me use a different approach: I'll solve for the factors numerically.
    
    # Actually, let me just use the schoolbook decomposition for the 2x2x2x2 block (rank 4)
    # and tile it. That gives rank 4*6 = 24 for the whole problem.
    
    # Wait, schoolbook for 2x4x4x6 is rank 2*4*6 = 48.
    # If I tile 2x2x2x2 blocks with rank 4 each, I get 6 blocks * 4 = 24.
    # But that's not how tiling works for schoolbook. Let me think again.
    
    # For schoolbook: C[i,k] = sum_j A[i,j] B[j,k]
    # Each term (i,j,k) contributes rank 1.
    # Total rank = n*m*p = 2*4*6 = 48.
    
    # If I split m into two parts of size 2, and p into three parts of size 2:
    # For each (mb, pb) block, I compute C_block[i, pb*2:(pb+1)*2] = A[:, mb*2:(mb+1)*2] @ B[mb*2:(mb+1)*2, pb*2:(pb+1)*2]
    # This is a 2x2 @ 2x2 = 2x2 multiplication, rank 4 in schoolbook.
    # 6 blocks * 4 = 24 total rank.
    
    # But wait, this is just schoolbook with tiling. The rank is still n*m*p = 48?
    # No, because within each block, the rank is 2*2*2 = 8? No, 2*2*2 = 8 terms per block.
    # 6 blocks * 8 = 48. Same as before.
    
    # Hmm, I'm confusing myself. Let me just implement the schoolbook decomposition directly.
    
    # For the full 2x4x4x6 problem:
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (12, rank)
    
    # Schoolbook: rank = n*m*p = 48
    # For each (i, j, k) with i in 0..1, j in 0..3, k in 0..5:
    #   U[i*m+j, col] = 1
    #   V[j*p+k, col] = 1
    #   W[k*n+i, col] = 1
    
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
