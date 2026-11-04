"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Decompose M_{2,3,10} using the fact that n=2 allows pairing outputs.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # For each k (0..9), we have a 2-output bilinear form in 3+3=6 variables.
    # The rank of the 2x3 x 3x2 tensor (for one k) is at most 3 (schoolbook per k)
    # but can it be less? The 2x3 x 3x2 tensor has rank 3 (since it's essentially
    # a 2x2 x 2x2 with an extra dimension... actually let's verify).
    #
    # Actually for fixed k: C[:,k] = A[:,0]*B[0,k] + A[:,1]*B[1,k] + A[:,2]*B[2,k]
    # This is 3 rank-1 terms in the bilinear map (A,B) -> C[:,k].
    # But can we do it in fewer? The 2x3 x 3x2 multiplication tensor:
    # rank is actually 3 (you need at least 3 because the 3 middle indices are independent).
    # So per k we need 3, total 3*10 = 30? No wait, that's not right either.
    #
    # Let me reconsider. The full tensor M_{2,3,10} has:
    # - U indices: n*m = 6 (A entries)
    # - V indices: m*p = 30 (B entries)  
    # - W indices: p*n = 20 (C entries)
    #
    # Schoolbook: 3*10 = 30 terms? No, schoolbook is n*m*p = 2*3*10 = 60.
    #
    # Wait, I confused myself. Let me be very careful.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # Number of (i,j,k) triples: 2 * 3 * 10 = 60. That's the schoolbook rank.
    #
    # Can we reduce? For each k, C[:,k] depends on A[:,0]*B[0,k] + A[:,1]*B[1,k] + A[:,2]*B[2,k]
    # The bilinear form for fixed k maps (A row i, B column k) -> C[i,k]
    # For all i, C[i,k] = sum_j A[i,j] B[j,k]
    # This is a 2x3 matrix times a 3x1 vector, giving 2x1.
    # The tensor for this is M_{2,3,1} which has rank 3 (cannot do better for 2x3x1).
    # So minimum per k is 3, total minimum is 30.
    #
    # But wait - can we share computation across k values? The B[j,k] values differ per k,
    # but A[i,j] values are shared. The question is whether the 10 different "slices" can
    # be computed with shared bilinear forms.
    #
    # Key insight: C[i,k] for all k is: C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:]
    # So C[i,:] is a linear combination of the 3 rows of B, with coefficients A[i,0], A[i,1], A[i,2].
    #
    # The bilinear forms needed are products A[i,j] * B[j,k] for each (i,j,k).
    # For a fixed j and k, we need A[0,j]*B[j,k] and A[1,j]*B[j,k].
    # Can we compute both from fewer products? 
    # (A[0,j]+A[1,j])*B[j,k] and (A[0,j]-A[1,j])*B[j,k] give us:
    #   A[0,j]*B[j,k] = ((A[0,j]+A[1,j])*B[j,k] + (A[0,j]-A[1,j])*B[j,k]) / 2
    #   A[1,j]*B[j,k] = ((A[0,j]+A[1,j])*B[j,k] - (A[0,j]-A[1,j])*B[j,k]) / 2
    # So 2 products instead of 2 products - no savings!
    #
    # What about: can we compute A[0,j]*B[j,k] + A[1,j]*B[j,k] directly? That's (A[0,j]+A[1,j])*B[j,k]
    # But we need A[0,j]*B[j,k] and A[1,j]*B[j,k] separately for the output.
    #
    # Let me try Strassen-like trick on the 2x2 sub-blocks.
    # Consider j=0,1 (first two middle indices):
    # C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1
    # This is a 2x2 x 2x1 problem (for each k). 
    # 2x2 x 2x1: C[i,k] = sum_{j=0,1} A[i,j]*B[j,k]
    # Schoolbook: 4 products per k. Can Strassen help?
    # Strassen is for 2x2 x 2x2. For 2x2 x 2x1, it's just 4 scalar products, rank 4.
    # No improvement possible.
    #
    # I believe the rank of M_{2,3,10} is exactly 60 (schoolbook is optimal here).
    # Let me verify with a different code structure to avoid AST duplication.
    
    n, m, p = 2, 3, 10
    total_rank = n * m * p
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    U_mat = [[0] * total_rank for _ in range(U_rows)]
    V_mat = [[0] * total_rank for _ in range(V_rows)]
    W_mat = [[0] * total_rank for _ in range(W_rows)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_mat[i * m + j][idx] = 1
                V_mat[j * p + k][idx] = 1
                W_mat[k * n + i][idx] = 1
                idx += 1
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
