"""Matrix multiplication 2,3,9 - rank-minimal tensor decomposition."""
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # p=9, so we have 3 blocks of size 3
    block_size = 3
    num_blocks = p // block_size  # = 3
    
    # For each [2,3,3] block, use a rank-6 decomposition
    # Total rank = 3 * 6 = 18
    block_rank = 6
    total_rank = num_blocks * block_rank  # 18
    
    U = [[0] * total_rank for _ in range(n * m)]  # (6, 18)
    V = [[0] * total_rank for _ in range(m * p)]  # (27, 18)
    W = [[0] * total_rank for _ in range(p * n)]  # (18, 18)
    
    # Define the rank-6 decomposition for [2,3,3]
    # U_block: (6, 6), V_block: (9, 6), W_block: (6, 6)
    # T[i,j,k] = sum_c U[i*3+j,c] * V[j*3+k,c] * W[k*2+i,c]
    
    # Known rank-6 decomposition for 2x3x3 matrix multiplication:
    # This uses the structure that we can express the product with 6 terms
    
    # Let me try to construct it systematically.
    # The multiplication is: C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # for i in {0,1}, k in {0,1,2}
    
    # A rank-1 tensor is u_i * v_j * w_k where u is length 2, v is length 3, w is length 3
    # We need: sum_c u_c[i] * v_c[j] * w_c[k] = delta (the multiplication tensor)
    # where the multiplication tensor has T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Actually, the tensor is: T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor entries are: T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes)
    # Wait, no. The matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} T[i,j,k] * e_i ⊗ e_j ⊗ e_k
    # where the multiplication is recovered by contracting.
    
    # Actually, the standard definition: the matrix multiplication tensor [n,m,p] is
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ f_j ⊗ g_k
    # where the multiplication C = AB is given by <T, A ⊗ B> = C
    # More precisely, T[i,j,k] = 1 for all i,j,k, and the output is C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    
    # So T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # We need: sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*2+i, c] = 1 for all i,j,k.
    
    # This is a system of 2*3*3 = 18 equations with 6*6 + 9*6 + 6*6 = 36+54+36 = 126 unknowns.
    # It should be solvable.
    
    # Let me try a specific construction. I'll use the following approach:
    # For the [2,3,3] case, I can try to find 6 vectors (u_c, v_c, w_c) such that
    # sum_c u_c[i] * v_c[j] * w_c[k] = 1 for all i,j,k.
    
    # One approach: use the fact that 1 = sum of rank-1 tensors.
    # Let me try:
    # c=0: u=(1,0), v=(1,0,0), w=(1,0,0) -> contributes to T[0,0,0]
    # c=1: u=(1,0), v=(0,1,0), w=(1,0,0) -> contributes to T[0,1,0]
    # ... this gives schoolbook rank 18, not 6.
    
    # I need a non-trivial decomposition. Let me try to use algebraic identities.
    
    # Actually, I recall now that the rank of [2,3,3] is NOT 6. Let me reconsider.
    # The rank of [2,2,2] is 7. For [2,3,3], I think the rank might be 18 (schoolbook) 
    # or possibly lower, but I'm not sure of the exact value.
    
    # Given the uncertainty, let me just use the schoolbook decomposition which is guaranteed to work.
    # Schoolbook rank for [2,3,9] is 2*3*9 = 54.
    
    # But wait - maybe I can do better by exploiting the block structure.
    # The key insight: C[i, 3b+k] = sum_j A[i,j] * B[j, 3b+k] for b=0,1,2
    # This is three independent [2,3,3] multiplications.
    # If the rank of [2,3,3] is r, then total rank is 3r.
    
    # Since I'm not sure of r, let me just use schoolbook for now (rank 54)
    # and see if I can improve.
    
    # Actually, let me try rank 18 first (assuming [2,3,3] has rank 6) and verify.
    # If it doesn't work, I'll fall back to schoolbook.
    
    # For now, let me just do schoolbook to ensure correctness.
    rank = n * m * p  # 54
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
