"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8  # A is n x m, B is m x p, C is n x p
    # W index: k*n + i, so W has p*n = 24 rows
    # U index: i*m + j, so U has n*m = 12 rows
    # V index: j*p + k, so V has m*p = 32 rows
    
    # Schoolbook gives rank 96.
    # Strassen-based: group j into 2 blocks of size 2, k into 4 blocks of size 2.
    # For each (j_block, k_block) pair, use Strassen's 7-term decomposition for the 2x2 sub-blocks.
    # This gives 8 blocks * 7 = 56 rank terms? No, that's not right either.
    
    # Let me think more carefully. The tensor T has components T[i, j, k] = delta_{i,j,k}
    # meaning C[i,k] = sum_j A[i,j] * B[j,k].
    # The tensor is sum over (i,j,k) of e_i ⊗ e_j ⊗ e_k for the multiplication tensor.
    # Actually the multiplication tensor for (n,m,p) has n*m*p components indexed by (i,j,k).
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The condition is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k} (1 if this triple is a valid term, 0 otherwise)
    # Wait, that's not quite right. Let me re-read.
    
    # The decomposition: C[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor: T = sum_{i,j,k} e_{i} ⊗ e_{j} ⊗ e_{k} where T[i,j,k] = 1 for all valid (i,j,k)
    # No, that's not right either. The multiplication tensor T_mul satisfies:
    # T_mul[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    # Actually, T_mul has 1 at every (i,j,k) triple, and the tensor contraction gives:
    # C[i,k] = sum_{j} A[i,j] B[j,k] = <T_mul, A ⊗ B ⊗ something>
    
    # The correct formulation: the tensor T has entries T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # A rank-1 decomposition term r contributes: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k).
    
    # Schoolbook: one term per (i,j,k) triple, with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
    # This gives rank n*m*p = 3*4*8 = 96.
    
    # Strassen improvement: For the 2x2x2 case, rank is 7 instead of 8.
    # For 3x4x8, we can try to exploit structure.
    
    # Let's use a simple approach: try to find a lower-rank decomposition using known bounds.
    # The minimum rank for 3x4x8 is not known exactly, but upper bounds exist.
    # A trivial upper bound is 96 (schoolbook).
    # Strassen's method for 2x2x2 gives rank 7.
    
    # One approach: decompose into blocks.
    # Let's try: group the j dimension (size 4) into two groups of 2.
    # Group the k dimension (size 8) into four groups of 2.
    # For each pair (j_group, k_group), we have a 2x2 sub-problem in (j,k) for each i.
    # But i is shared, so it's not independent.
    
    # Alternative: use the fact that rank(3x4x8) <= rank(4x4x8) * something?
    # Not straightforward.
    
    # Let's just use schoolbook for correctness, as deriving an optimal Strassen-based
    # decomposition for 3x4x8 in real-time is complex.
    
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
# EVOLVE-BLOCK-END
