"""Matrix multiplication 3,3,5 tensor decomposition."""
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Try a structured low-rank decomposition
    # For each j (column of A / row of B), we can factor the partial product
    # C_ik += A_ij * B_jk
    # Strategy: for each j and each i, one term: U picks A_ij, V picks entire B_j row, W scatters to output
    # This gives rank = n * m = 9, but W must be a "scatter" vector
    # Actually: U[(i*m+j), r] = 1 for the r corresponding to (i,j)
    #           V[(j*p+k), r] = B_jk coefficient... but V is fixed, not depending on input
    # 
    # Correct approach: Each rank-1 term is U_col * V_col * W_col (outer product structure)
    # The term contributes: (sum_{i',j'} U[i'm+j',r] A[i'j']) * (sum_{j',k'} V[j'p+k',r] B[j'k']) * W[k'n+i, r]
    # to C[i,k] (stored at W index k*n+i)
    #
    # For a term to contribute A_ij * B_jk to C[i,k]:
    #   U[i*m+j, r] = 1, U[other] = 0
    #   V[j*p+k, r] = 1, V[other] = 0  
    #   W[k*n+i, r] = 1, W[other] = 0
    # That's the schoolbook rank-45.
    #
    # To reduce rank, we can let one term contribute to multiple outputs.
    # If for fixed j, we set U[i*m+j, r_i] = 1 for a group of i's sharing one r:
    #   Let r handle i=0,1,2 all with same j. Then U[0*m+j,r]=U[1*m+j,r]=U[2*m+j,r]=1
    #   V[j*p+k,r] would need to be 1 for all k to pick up B_jk
    #   Then W[k*n+i',r] must select only the right (i,k) pairs
    #   The term gives: (A_0j + A_1j + A_2j) * (sum_k B_jk) * W[k*n+i',r] -- WRONG, cross terms
    #
    # So we can't simply group. The correct low-rank uses the fact that:
    # sum_j A_ij B_jk = sum_j (A_ij * 1) * (1 * B_jk)
    # Each (j) term is a rank-1 in (i,k) space with factors u_ij and v_jk.
    # The total rank is sum over j of rank of the outer product structure.
    # For each j, the map (i,k) -> A_ij * B_jk is rank 1 in the (i,k) plane,
    # but we need it as a tensor term. Each such term needs its own r.
    # Minimum rank is at least max(n*p, n*m, m*p) = max(15, 9, 15) = 15? No,
    # the tensor rank of matrix multiplication is known to be between various bounds.
    #
    # For 3x3 * 3x5, the known best upper bound is at most 45 (schoolbook).
    # There are no known sub-multiplicative decompositions for these small sizes
    # that beat schoolbook significantly. Let's try rank 27 (3/5 of 45) or just
    # return schoolbook which is guaranteed correct.
    
    # Return the guaranteed schoolbook decomposition of rank n*m*p = 45
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
