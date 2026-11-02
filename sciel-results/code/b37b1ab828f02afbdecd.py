"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank = n*m*p = 54)
    def schoolbook():
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
    
    # Kronecker product structure: decompose 2x3 * 3x9 into three parallel 2x3 * 3x3 multiplications.
    # The 9-dimensional second factor is split into 3 blocks of 3.
    # For each block, we use a 2x3 * 3x3 multiplication.
    # 
    # Key insight: The tensor for 2x3 * 3x3 multiplication has rank at most 18 (schoolbook).
    # But we can do better: the tensor M_{2,3,3} has known rank bounds.
    # Actually, let's use the fact that 2x3 * 3x3 can be done with rank 18 via schoolbook,
    # but there might be a better decomposition.
    #
    # However, the directive says to leverage the fact that the tensor rank of 2x3x3 is 6.
    # Wait, that's not right. The tensor rank of the 2x3x3 matrix multiplication tensor is not 6.
    # The schoolbook rank is 2*3*3 = 18.
    #
    # Let me reconsider. The directive says "the tensor rank of 2 x 3 x 3 is 6". This is incorrect
    # for the standard matrix multiplication tensor. The tensor M_{2,3,3} has rank at most 18.
    #
    # But perhaps the directive is referring to a different tensor or a different interpretation.
    # Let's re-read: "leveraging the fact that the tensor rank of 2 x 3 x 3 is 6, thereby constructing
    # a valid rank-18 decomposition that exploits the block-diagonal nature of the second factor's dimension."
    #
    # This is confusing. If the tensor rank of 2x3x3 is 6, then three blocks would give rank 18.
    # But the schoolbook rank of 2x3x3 is 18, not 6.
    #
    # Perhaps the directive is mistaken, or perhaps it's referring to a different tensor.
    # Let's just implement the Kronecker product structure with the schoolbook decomposition
    # for each 2x3 * 3x3 block, giving total rank 3 * 18 = 54. This is the same as the original.
    #
    # To actually get a rank-18 decomposition, we would need the rank of M_{2,3,3} to be 6,
    # which is not the case. The best known upper bound for R(2,3,3) is 18 (schoolbook),
    # and lower bounds suggest it's at least 9.
    #
    # Given the ambiguity, let's try a different approach: use a rank-18 decomposition
    # if we can find one, otherwise fall back to schoolbook.
    #
    # Actually, let's just implement the Kronecker product structure properly.
    # The 2x3 * 3x9 multiplication can be viewed as:
    #   A (2x3) * B (3x9) = C (2x9)
    # Split B into three 3x3 blocks: B = [B1 | B2 | B3]
    # Then C = [A*B1 | A*B2 | A*B3]
    # Each A*Bb is a 2x3 * 3x3 multiplication.
    #
    # The tensor for the full multiplication is the direct sum of three copies of M_{2,3,3}.
    # So rank(M_{2,3,9}) <= 3 * rank(M_{2,3,3}).
    #
    # If we use schoolbook for each block, rank = 3 * 18 = 54.
    # If we can find a rank-6 decomposition for M_{2,3,3}, then rank = 3 * 6 = 18.
    #
    # But rank(M_{2,3,3}) >= 9 (by standard lower bounds), so rank-6 is impossible.
    #
    # Let's just implement the Kronecker product structure with schoolbook, which gives rank 54.
    # This is the same as the original, but structured differently.
    
    # Kronecker product structure: three parallel 2x3 * 3x3 multiplications
    def kron_structure():
        # Each 2x3 * 3x3 uses schoolbook (rank 18)
        rank_per_block = n * m * 3  # 2*3*3 = 18
        num_blocks = p // 3  # 9 // 3 = 3
        rank = rank_per_block * num_blocks  # 18 * 3 = 54
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for b in range(num_blocks):
            for i in range(n):
                for j in range(m):
                    for k in range(3):
                        # U index: i*m + j
                        U[i * m + j][col] = 1
                        # V index: j*p + (b*3 + k)
                        V[j * p + b * 3 + k][col] = 1
                        # W index: (b*3 + k)*n + i
                        W[(b * 3 + k) * n + i][col] = 1
                        col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Check if kron_structure gives a valid decomposition
    def is_valid(decomp):
        U = np.array(decomp["U"])
        V = np.array(decomp["V"])
        W = np.array(decomp["W"])
        rank = U.shape[1]
        
        # Verify: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k}
        # Actually, the tensor decomposition condition is:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where T[i,j,k] = 1 if this is a valid multiplication tensor entry.
        
        # For matrix multiplication, the tensor is:
        # T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k
        # i.e., T[i*m+j, j*p+k, k*n+i] = 1
        
        # Let's verify by checking a few entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try the Kronecker structure
    kron_decomp = kron_structure()
    
    # Verify it's valid
    if is_valid(kron_decomp):
        # Check if it's better than schoolbook
        kron_rank = kron_decomp["U"][0].__len__() if kron_decomp["U"] else 0
        schoolbook_rank = n * m * p
        
        if kron_rank < schoolbook_rank:
            return kron_decomp
        else:
            # Same rank, but let's return the Kronecker structure as it's more structured
            return kron_decomp
    else:
        # Fall back to schoolbook
        return schoolbook()
# EVOLVE-BLOCK-END
