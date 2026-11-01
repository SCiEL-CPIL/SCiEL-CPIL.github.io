"""
Rank-minimal decomposition for multiplying a 2x3 matrix by a 3x4 matrix.

Strategy:
  Use the known lower-bound-optimal rank for this tensor. The tensor is 
  <2,3,4> with tensor rank 24 (schoolbook), but we can potentially do better.
  
  Actually, let's think carefully. The tensor rank of <2,3,4> is known to be 
  at most 24 (schoolbook). The lower bound via border rank or other methods 
  suggests it might be less, but for exact integer decomposition, let's 
  construct a clever decomposition.
  
  We can use the fact that we're multiplying (2x3) * (3x4) = (2x4).
  
  One approach: Use bilinear algorithms. Let's try to find a rank-20 or lower 
  decomposition.
  
  Actually, the known tensor rank of <2,3,4> is 24? Let me reconsider.
  
  The tensor is T in R^{2x4} x R^{2x3} x R^{3x4} where C_{ik} = sum_j A_{ij} B_{jk}.
  
  Let's try a construction based on exploiting structure.
  
  We'll use a block approach: treat A as 2x3, B as 3x4.
  
  Let me try to find a rank-20 decomposition or similar.
  
  Actually, let me just construct the schoolbook (rank 24) and then try to 
  find a better one via known results or clever construction.
  
  For now, I'll construct a rank-24 solution that is correct, and see if we 
  can improve.
"""

def construct():
    n, m, p = 2, 3, 4
    # Schoolbook rank = n*m*p = 24
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x rank
    V = [[0] * rank for _ in range(m * p)]  # 12 x rank
    W = [[0] * rank for _ in range(p * n)]  # 8 x rank
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
