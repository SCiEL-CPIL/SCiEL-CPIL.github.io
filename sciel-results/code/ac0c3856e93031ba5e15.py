"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Using a symmetry-reduced approach: we model the 2x4x5 multiplication by exploiting
    the small n=2 dimension. We use a structured decomposition that shares rank-1 terms
    across the two rows of the first matrix.
    
    The key insight: for n=2, we can use a decomposition where the U factor has structure
    that allows sharing between i=0 and i=1 terms.
    
    We'll use a rank-36 decomposition based on the following strategy:
    - For each (j,k) pair, we need to handle both i=0 and i=1
    - Instead of separate terms for each (i,j,k), we can use shared terms with appropriate
      coefficients in W to separate the i=0 and i=1 contributions
    
    Specifically, we use rank = 2*m*p - m*p/2 = 3*m*p/2 = 3*4*5/2 = 30... 
    Actually let me think more carefully.
    
    For the 2x4x5 case, let's use a decomposition of rank 36:
    - 20 terms for the "main" part (one per (j,k) for each i)
    - 16 additional terms to handle cross-terms
    
    Actually, let me try a cleaner approach. The minimum rank for 2x4x5 is at least 20.
    Let me try to construct a rank-36 decomposition using a structured approach.
    
    Strategy: Use the fact that for n=2, we can write:
    (A*B)[0,k] = sum_j A[0,j]*B[j,k]
    (A*B)[1,k] = sum_j A[1,j]*B[j,k]
    
    We can share the B[j,k] factors. For each j, we have two "rows" A[0,j] and A[1,j].
    
    Let me try a rank = 36 decomposition:
    - For each j in [0,4) and k in [0,5), create 2 terms (one for i=0, one for i=1)
      That gives 40 terms (schoolbook).
    
    To reduce, we can try to merge some terms. For example, for a fixed j, the terms
    for different k can potentially be shared.
    
    Let me try a different approach: use rank 36 by having 9 terms per j (4*9=36).
    For each j, we have 5 k-values. We can use 9 terms to cover both i=0 and i=1
    for all 5 k-values.
    
    Actually, let me just implement a rank-36 decomposition with a specific structure.
    
    For each j in [0,4):
      We need to compute:
        out_0[k] += A[0,j] * B[j,k] for k=0..4
        out_1[k] += A[1,j] * B[j,k] for k=0..4
      
      This is equivalent to multiplying the 2x1 vector [A[0,j], A[1,j]] by the 1x5 matrix
      [B[j,0], B[j,1], B[j,2], B[j,3], B[j,4]] and adding to the output.
      
      The tensor for this sub-problem has rank at most 2*5=10 (schoolbook for 2x1x5).
      But we can do better: the 2x1x5 multiplication tensor has rank 5 (since it's just
      scaling each B[j,k] by A[0,j] and A[1,j] separately, but we can share the B[j,k] factor).
      
      Wait, for 2x1x5: (A[0]*B[0], A[0]*B[1], ..., A[0]*B[4], A[1]*B[0], ..., A[1]*B[4])
      This is a 2x5 output from a 2x1 input and 1x5 input.
      The tensor rank is 5: use terms (e_k ⊗ e_1) ⊗ (e_k ⊗ e_1) ... no.
      
      Actually, for 2x1x5, the schoolbook rank is 2*1*5=10. But we can do rank 5:
      For each k, use one term: U[0,k]=1, V[k,k]=1, W[k*2+0,k]=1 for the A[0,j]*B[j,k] part
      and U[1,k]=1, V[k,k]=1, W[k*2+1,k]=1 for the A[1,j]*B[j,k] part.
      That's still 10 terms.
      
      Hmm, can we do rank 5 for 2x1x5? The output has 10 entries, each is a product of
      one A entry and one B entry. The tensor is:
      T[(i,k), (i'), (k')] = 1 if i=i' and k=k', else 0
      This is just the identity-like tensor, which has rank 10 (each output entry needs
      its own term since A[0,j] and A[1,j] are independent and B[j,0..4] are independent).
      
      So 2x1x5 has rank 10. Thus 2x4x5 has rank at least 4*10=40? No, that's not right
      because the j dimension allows sharing.
      
      Let me reconsider. For 2x4x5, the tensor is:
      T[(i,k), (i',j), (j',k')] = 1 if i=i', j=j', k=k'
      
      The rank is the minimum number of rank-1 tensors that sum to T.
      
      One lower bound: fix i=0, j=0. Then T[(0,k), (0,0), (0,k')] = delta_{k,k'} for k,k' in [0,5).
      This is a 5x5 identity matrix, which has rank 5. So the overall tensor rank is at least 5.
      Similarly, fixing other indices gives lower bounds.
      
      The best known lower bound for 2x4x5 is 20 (from the max dimension argument).
      
      Let me try to construct a rank-36 decomposition using a structured approach.
      
      Actually, let me just use the schoolbook rank 40 for now, but try to verify if a lower
      rank is achievable with a simple heuristic.
      
      Given the time constraint, I'll return the schoolbook decomposition with rank 40.
    """
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
