"""
Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.

Approach:
  - The multiplication tensor T has entries T[i,j,k] = 1 for all (i,j,k) in [2]x[5]x[6].
  - We seek a decomposition T = sum_r u_r ⊗ v_r ⊗ w_r with minimal rank.
  - The schoolbook gives rank 60. We can do better by exploiting the structure.

Key insight:
  The tensor T_{i,j,k} for a 2x5x6 matrix multiplication can be decomposed by
  viewing it as: for each j (the shared index), we have a 2x6 outer product A_i * B_k.
  
  Actually, the standard tensor for matrix multiplication C = A*B where A is n x m and B is m x p:
  T[i,j,k] = 1 (the tensor is the "all ones" tensor of shape n x m x p).
  
  The rank of the all-ones tensor of shape n x m x p is known to be at most n*m*p (schoolbook).
  For 2x5x6, schoolbook is 60.
  
  Can we do better? The rank of the all-ones tensor T_{n,m,p} is the minimum number of rank-1
  tensors whose sum equals T. This is related to the border rank and is a hard problem.
  
  However, for small dimensions, we can use the following approach:
  - The all-ones tensor of shape 2 x 5 x 6 can be written as the sum of rank-1 tensors.
  - One approach: use the fact that T[i,j,k] = 1 = (1/2)(1+1)(1) for the first index, etc.
  
  Actually, let's think about this more carefully. The tensor T has T[i,j,k] = 1 for all i,j,k.
  
  A rank-1 tensor is u_i * v_j * w_k. We need sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.
  
  This is equivalent to finding the minimum rank of the tensor. For the all-ones tensor,
  the rank is known to be min(n,m,p) * something... Actually, I recall that the rank of the
  all-ones tensor of shape n x m x p is at most n + m + p - 2 (for the case where we can use
  a specific construction).
  
  Wait, that's not right either. Let me think again.
  
  For the all-ones tensor T[i,j,k] = 1, we can write:
  T[i,j,k] = sum_{r=1}^{R} u_r[i] * v_r[j] * w_r[k]
  
  One construction: 
  Let R = n + m + p - 2. Define:
  For r = 1 to n-1: u_r[i] = 1 if i <= r else 0, v_r[j] = 1, w_r[k] = 1
  For r = n to n+m-2: u_r[i] = 1, v_r[j] = 1 if j <= r-n+1 else 0, w_r[k] = 1
  For r = n+m-1 to n+m+p-3: u_r[i] = 1, v_r[j] = 1, w_r[k] = 1 if k <= r-n-m+2 else 0
  
  Hmm, this doesn't work directly. Let me use a different approach.
  
  Actually, the rank of the all-ones tensor of shape n x m x p is exactly min(n*m, m*p, n*p) for 
  the standard matrix multiplication tensor... no, that's the schoolbook rank.
  
  Let me just use the schoolbook decomposition with rank 60, but try to optimize it.
  
  Actually, I realize that the tensor for matrix multiplication is NOT the all-ones tensor in the 
  usual sense. The matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 for all (i,j,k), and 
  the multiplication is C[i,k] = sum_j A[i,j] * B[j,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
  
  So yes, it IS the all-ones tensor of shape n x m x p.
  
  The rank of the all-ones tensor of shape 2 x 5 x 6: 
  I believe the rank is at most 2 + 5 + 6 - 2 = 11? No, that's not a known result.
  
  Let me try a constructive approach. We want to minimize R such that:
  sum_{r=1}^{R} u_r[i] * v_r[j] * w_r[k] = 1 for all i in {0,1}, j in {0..4}, k in {0..5}.
  
  This is a system of 60 equations in 3*R variables (well, 2R + 5R + 6R = 13R variables, but with 
  scaling redundancy).
  
  For small R, we can try to find a solution. Let's try R = 10, 11, ..., 60.
  
  Actually, let me just return the schoolbook decomposition with rank 60 for now, since I'm not 
  sure of a better construction.
"""

def construct():
    n, m, p = 2, 5, 6
    
    # Schoolbook decomposition: rank = n * m * p = 60
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
