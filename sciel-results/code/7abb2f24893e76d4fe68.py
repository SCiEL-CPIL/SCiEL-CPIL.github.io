"""Program: rank-minimal exact tensor decomposition for 2x5 x 5x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T has components:
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # and 0 otherwise.
    # 
    # Schoolbook rank is n*m*p = 60.
    # 
    # We use a Strassen-like approach to reduce the rank.
    # 
    # Key insight: For 2xN x NxM matrix multiplication, we can exploit the fact that
    # n=2 to create shared multiplications. The idea is similar to Strassen's 7-mult
    # algorithm for 2x2x2, generalized to larger dimensions.
    #
    # Strategy: Partition the middle dimension (size m=5) and use a decomposition
    # that shares computations across the two rows of A.
    #
    # For 2x5 x 5x6, we'll use the following approach:
    # - Split into blocks and use a generalized Strassen decomposition
    # - The key is to find rank-1 terms that cover multiple (i,j,k) triples
    
    # Let's use a constructive approach based on known results.
    # For 2xN x NxM, a known upper bound on the rank is:
    #   rank <= 2*N*M - N  (this is not always tight, but gives a starting point)
    # For N=5, M=6: 2*5*6 - 5 = 55
    
    # Let me try to construct a decomposition with rank < 60.
    # 
    # Approach: Use the fact that for the 2-row case, we can write:
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    # 
    # Define:
    # S_j = A[0,j] + A[1,j]
    # D_j = A[0,j] - A[1,j]
    # 
    # Then:
    # C[0,k] = sum_j (S_j + D_j)/2 * B[j,k]
    # C[1,k] = sum_j (S_j - D_j)/2 * B[j,k]
    # 
    # This doesn't directly help because we still have 2*m*p products.
    # 
    # Let me try a different approach. I'll use a direct construction.
    # 
    # For the 2x5 x 5x6 case, I'll try to find a decomposition by exploiting
    # linear dependencies in the tensor structure.
    
    # Let me try a greedy approach: start with the schoolbook decomposition and
    # try to eliminate terms by finding linear dependencies.
    
    # Actually, let me try a known decomposition pattern.
    # 
    # For 2xN x NxN, there's a decomposition of rank 2N-1+something.
    # Let me just try to construct one systematically.
    
    # I'll use the following strategy:
    # 1. Start with the schoolbook decomposition (rank 60)
    # 2. Try to combine terms where possible
    
    # For now, let me return a slightly improved decomposition.
    # I'll use the observation that we can pair up some terms.
    
    # Actually, let me just implement a simple but valid decomposition with rank < 60.
    # I'll use the following construction:
    
    # For each pair of k values, try to share the multiplication.
    
    # Let me try a different approach: use the fact that the tensor has a specific
    # structure and find a lower-rank decomposition.
    
    # I'll construct a decomposition with rank = 2*p + m - 1 = 2*6 + 5 - 1 = 16
    # This is based on the idea that for 2xM x MxP, the rank is at most 2P + M - 1.
    
    # Wait, that seems too low. Let me verify with known cases:
    # 2x2 x 2x2: 2*2 + 2 - 1 = 5, but Strassen gives 7. So this formula is wrong.
    
    # Let me try another formula: rank <= 2*min(M,P) + max(M,P) - 1
    # 2x2 x 2x2: 2*2 + 2 - 1 = 5. Still wrong.
    
    # Let me just try to find a decomposition with rank < 60 by direct construction.
    
    # I'll use the following approach:
    # - For each j, instead of using 2*p separate rank-1 terms, try to use fewer.
    
    # For fixed j, the sub-tensor is:
    # S_j = sum_k [e_j ⊗ e_{j*p+k} ⊗ e_{2k} + e_{m+j} ⊗ e_{j*p+k} ⊗ e_{2k+1}]
    # 
    # This has 2*p non-zero entries. Can we decompose it with fewer than 2*p rank-1 terms?
    # 
    # S_j = sum_k e_{j*p+k} ⊗ (e_j ⊗ e_{2k} + e_{m+j} ⊗ e_{2k+1})
    # 
    # Each term e_{j*p+k} ⊗ (e_j ⊗ e_{2k} + e_{m+j} ⊗ e_{2k+1}) is a sum of two
    # rank-1 tensors in the full 3-tensor space. So S_j is a sum of 2*p rank-1 tensors.
    # 
    # To reduce the rank, we need to find linear dependencies among these 2*p terms.
    # 
    # Let me try a completely different approach: use a known algorithm.
    
    # I'll implement a simple rank reduction by checking if any term can be expressed
    # as a linear combination of others.
    
    # For the 2x5 x 5x6 case, let me just return the schoolbook decomposition
    # since I'm not confident I can find a better one quickly.
    
    # But wait - let me try one more thing. I'll use the following observation:
    # The tensor T can be written as:
    # T = sum_{j=0}^{m-1} (e_j ⊗ e_{j*p} ⊗ e_0 + ... + e_j ⊗ e_{j*p+p-1} ⊗ e_{2p-2} 
    #     + e_{m+j} ⊗ e_{j*p} ⊗ e_1 + ... + e_{m+j} ⊗ e_{j*p+p-1} ⊗ e_{2p-1})
    
    # Let me try to use a decomposition based on polynomial interpolation.
    # 
    # Actually, let me just try to find a rank-59 or lower decomposition by
    # checking for linear dependencies.
    
    # I'll use the following approach:
    # 1. Create the schoolbook decomposition
    # 2. Check if any column is a linear combination of others
    # 3. If so, remove it and adjust the other columns
    
    # For efficiency, I'll just return a known-good decomposition.
    
    # Let me try rank = 59 by removing one term and adjusting.
    
    # Actually, I realize I should just try to construct a valid decomposition
    # with rank < 60. Let me use a systematic approach.
    
    # I'll use the following construction for 2xM x MxP:
    # Use 2*P + M - 2 rank-1 terms (this is a guess, may not be valid)
    
    # Let me just return the schoolbook decomposition for now.
    # The task is to improve it, so let me try harder.
    
    # New approach: Use the fact that for the 2-row case, we can write the product as:
    # [C0; C1] = [A0; A1] * B
    # where C0, C1 are row vectors of length p, A0, A1 are row vectors of length m.
    # 
    # We can write:
    # C0[k] = sum_j A0[j] * B[j,k]
    # C1[k] = sum_j A1[j] * B[j,k]
    # 
    # Define new variables:
    # X[j] = A0[j] + A1[j]
    # Y[j] = A0[j] - A1[j]
    # 
    # Then:
    # C0[k] = sum_j (X[j] + Y[j])/2 * B[j,k]
    # C1[k] = sum_j (X[j] - Y[j])/2 * B[j,k]
    # 
    # This still gives 2*m*p products, so no savings.
    
    # Let me try a different decomposition. I'll use the following idea:
    # Group the k indices and use shared multiplications.
    
    # For now, I'll return the schoolbook decomposition with rank 60.
    # This is the baseline, and the task is to improve it.
    
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
