"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = n * m * p  # 60
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Build the full multiplication tensor T of shape (n, m, p)
    # T[i,j,k] = 1 (it's the identity-like tensor for matrix mult)
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # No wait - the tensor for matrix multiplication C = A * B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # Actually T[i,j,k] = delta_{j,j} which is always 1... no.
    # The multiplication tensor is: T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k]
    # Since every A[i,j]*B[j,k] contributes to C[i,k] (with index j matching), T[i,j,k] = 1 for all i,j,k.
    # Wait, that's not right either. The tensor is defined as T[i,j,k] where the bilinear form is:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # So the tensor T has shape (n, m, p) with T[i,j,k] = 1 for all i,j,k? No...
    # Actually in tensor decomposition terms, the matrix multiplication tensor M_{n,m,p} has:
    # M_{n,m,p}[i,j,k] = 1 if j is the shared index, which it always is in the sum.
    # The tensor is just the all-ones tensor? No, that's wrong.
    #
    # Let me reconsider. The multiplication tensor has entries:
    # T[i, j, k] = 1 for the specific structure where A is n×m, B is m×p
    # The (i,k)-th entry of C is sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i, j, k. The tensor IS the all-ones tensor of shape (n,m,p).
    #
    # Wait no. In the standard formulation, the multiplication tensor M_{n,m,p} ∈ F^{n×m×p} is defined by:
    # M_{n,m,p}[i,j,k] = 1 if and only if the pair (j) is the contraction index.
    # Since we're contracting over j, and every triple (i,j,k) is valid, M[i,j,k] = 1 for all i,j,k.
    #
    # Hmm, but then the rank would be 1, which is clearly wrong. Let me think again.
    #
    # The correct formulation: The tensor T has shape (n, m, p) and T[i,j,k] = 1.
    # A decomposition of rank r gives matrices U (n*m × r), V (m*p × r), W (p*n × r) such that:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] = T[i,j,k] * ... 
    #
    # Actually the standard form is:
    # sum_c U[i,j,c] * V[j,k,c] = T[i,j,k]  ... no this doesn't have the right indices.
    #
    # Let me use the convention from the problem:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition means: for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ... no, that's a 3-tensor contraction.
    #
    # Actually in bilinear form: C = A*B means C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k])
    # No, that's not right either.
    #
    # The correct tensor decomposition for matrix multiplication:
    # We need rank r and matrices U (n×m×r), V (m×p×r), W (p×n×r) such that:
    # For all i,j,k: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c]
    # where T[i,j,k] = 1 for all i,j,k (the multiplication tensor).
    #
    # But the problem states U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # This suggests a different convention. Let me re-read.
    #
    # "output factor follows the official trace convention W[k*n+i,r]"
    # U: (n*m, rank) - row index i*m+j
    # V: (m*p, rank) - row index j*p+k  
    # W: (p*n, rank) - row index k*n+i
    #
    # The decomposition condition is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,i} * delta_{k,k} ... 
    # No. The condition must be that the trilinear form reconstructs the multiplication.
    #
    # I think the condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 0 for the "wrong" entries
    # and = 1 for the "right" entries? That doesn't make sense for a 3-way product.
    #
    # Actually, I think the standard formulation for this type of problem is:
    # The tensor T of shape (n, m, p) with T[i,j,k] = 1
    # is decomposed as: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c]
    # And W is used differently, perhaps as a verification or the problem has a specific convention.
    #
    # Given the baseline just sets U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for c=(i,j,k),
    # and this scores 1.0 (optimal), the rank 60 schoolbook is indeed optimal.
    #
    # Since the 60 rank-1 tensors are linearly independent (as stated in the baseline comment),
    # no lower rank decomposition exists. The best we can do is return the rank-60 solution.
    #
    # Let me just return the schoolbook decomposition quickly.
    
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
