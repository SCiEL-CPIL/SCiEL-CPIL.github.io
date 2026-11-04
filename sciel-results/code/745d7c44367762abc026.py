"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose into 3 blocks of M_{2,3,3}, each with rank 9
    # Total rank = 27
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each block t (0,1,2), compute C[i, 3t+r] = sum_j A[i,j] * B[j, 3t+r]
    # Use rank-9 decomposition of M_{2,3,3}
    # Rank-9 construction: for each (j, r) pair where j,r in {0,1,2},
    # create a term that handles A[i,j] * B[j, 3t+r] for both i=0 and i=1
    # But a rank-1 tensor must have single non-zero entries in U, V, W
    # So we need 2 terms per (j,r) pair (one for i=0, one for i=1) = 18 per block
    # That gives 54 total, which is schoolbook.
    
    # Better: use the fact that M_{2,3,3} has rank 9.
    # Construction: use 9 terms indexed by s=0..8
    # s = 3*j + r where j,r in {0,1,2}
    # For term s: 
    #   U has non-zero at (0,j) and (1,j) -> but that's 2 entries, not rank 1!
    
    # The correct rank-9 construction for M_{2,3,3}:
    # We can write the tensor as sum of 9 rank-1 tensors using
    # the identity based on the 3x3 structure.
    
    # Actually, let me use a verified construction:
    # For M_{2,3,3}, use rank 9 with the following pattern:
    # Terms are indexed by (a, b) for a,b in {0,1,2}
    # U[i*m+j, s] = 1 if j == a (for both i values? No...)
    
    # I'll use the straightforward approach: since I'm not certain of the 
    # rank-9 construction, I'll use rank 27 with a valid construction.
    # Each block uses rank 9 via: for each j in 0..2 and r in 0..2,
    # we need to compute A[0,j]*B[j,3t+r] and A[1,j]*B[j,3t+r]
    # That's 2*3*3 = 18 terms per block = 54 total.
    
    # To get 27, I need a rank-9 per block. Let me use this known fact:
    # M_{2,3,3} can be decomposed into 9 rank-1 tensors.
    # One construction uses the "outer product" structure:
    # T = sum_{j=0}^{2} (sum_{i=0}^{1} e_i) ⊗ e_j ⊗ (sum_{r=0}^{2} e_r)
    # But each (sum_i e_i) ⊗ e_j ⊗ (sum_r e_r) is NOT rank 1 in the 
    # standard tensor sense (it has multiple non-zero entries).
    
    # In the bilinear form decomposition sense:
    # C[i,r] = sum_s U[i,j,s] * V[j,r,s] * W[r,i,s]... 
    # The interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # and the decomposition must satisfy the matrix multiplication identity.
    
    # I'll just use rank 54 schoolbook to be safe, but that's the same as parent.
    # Let me try rank 27 with a careful construction.
    
    # For M_{2,3,3}, a valid rank-9 decomposition:
    # Use the 9 terms corresponding to the 9 entries of a 3x3 matrix.
    # Term (j,r): 
    #   U[0*m+j] = 1, U[1*m+j] = 1  -> 2 non-zeros, NOT rank 1
    
    # OK the issue is clear: in this formulation, each rank-1 term has 
    # exactly one non-zero in U, one in V, one in W. So for M_{2,3,3},
    # the minimum rank is 18 (one for each (i,j,r) triple).
    
    # Wait, that means M_{2,3,3} has rank 18, not 9!
    # And M_{2,3,9} has rank 54 (schoolbook).
    
    # But that can't be right - the literature says matrix multiplication 
    # tensors have lower rank than schoolbook.
    
    # I think the confusion is in the tensor definition. Let me just use 
    # rank 54 schoolbook which is guaranteed correct.
    
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
