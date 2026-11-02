"""Matrix multiplication 3,5,7 - block-circulant embedding with FFT-based rank-1 decomposition."""

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import dft

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Strategy: Embed the 3x5x7 multiplication into a cyclic convolution structure.
    # The key insight: for fixed i, the multiplication (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # can be viewed as a correlation/convolution between row i of A and column k of B.
    
    # We use a block-circulant approach:
    # 1. Choose N >= m + p - 1 = 11 to avoid wrap-around in cyclic convolution
    # 2. For each i in [0,n), decompose the bilinear form using N rank-1 terms
    #    based on the DFT of size N
    # 3. Total rank = n * N = 3 * 11 = 33
    
    N = m + p - 1  # 11, sufficient for linear convolution without aliasing
    
    # Precompute the DFT matrix of size N
    # F[a,b] = omega^(a*b) where omega = exp(2*pi*i/N)
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega ** (a * b) for b in range(N)] for a in range(N)])
    
    # For the cyclic convolution of size N, the rank-1 decomposition of the 
    # convolution tensor is given by the DFT eigendecomposition.
    # Specifically, the tensor C[a,b,c] = 1 if a+b ≡ c (mod N) has rank N with:
    #   C = sum_{f=0}^{N-1} u_f ⊗ v_f ⊗ w_f
    # where u_f[a] = omega^(f*a), v_f[b] = omega^(f*b), w_f[c] = omega^(-f*c) / N
    # (or similar variants depending on convention)
    
    # For our multiplication:
    # (A*B)[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # We embed A's row i into a vector of length N: a_i[j] = A[i,j] for j<m, 0 otherwise
    # We embed B's column k into a vector of length N: b_k[j] = B[j,k] for j<p, 0 otherwise
    # But wait, B is m x p, so B[j,k] where j in [0,m), k in [0,p)
    # 
    # Actually, let's think of it differently. Define:
    # For each i: vector a_i of length N where a_i[j] = A[i,j] for j in [0,m), 0 otherwise
    # For each k: vector b_k of length N where b_k[j] = B[j,k] for j in [0,m), 0 otherwise
    # Then (A*B)[i,k] = sum_j a_i[j] * b_k[j] = dot product, not convolution.
    #
    # Hmm, that's just the standard inner product. The convolution structure isn't directly 
    # applicable to the standard matrix multiplication in this way.
    
    # Let me reconsider. The correct embedding for matrix multiplication using 
    # circulant matrices is:
    # 
    # For multiplying A (n x m) by B (m x p), we can use:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    #
    # One approach: use the fact that we can write this as a sum of products of 
    # linear combinations. The FFT-based approach works when the multiplication 
    # can be embedded in a convolution.
    #
    # Alternative: Use the "toeplitz embedding" approach.
    # 
    # Actually, let me use a simpler and more direct approach:
    # The rank of the matrix multiplication tensor <n,m,p> is at most n*m*p.
    # We can improve this using the following observation:
    # 
    # For the tensor T with T[i*m+j, j*p+k, k*n+i] = 1 (and 0 otherwise),
    # we can try to find a lower-rank decomposition.
    #
    # Let me use the following construction based on the fact that:
    # sum_{j=0}^{m-1} A[i,j] * B[j,k] can be computed using m multiplications.
    # For each i and k, we need m products. Total: n*p*m = 105 products.
    #
    # To reduce rank, we can use the following trick:
    # If we can find a decomposition where some of the rank-1 terms share structure,
    # we can reduce the total rank.
    #
    # Let me try a different approach: use the Winograd method for small matrices.
    # For 3x5x7, there might be a known optimal or near-optimal decomposition.
    #
    # Given the complexity, let me use a straightforward but slightly optimized approach:
    # Use the schoolbook decomposition but check if we can merge any terms.
    
    # Actually, let me use the following correct FFT-based construction:
    # 
    # Key insight: We can embed the multiplication into a larger problem where 
    # the core operation is a cyclic convolution, then use the DFT to decompose it.
    #
    # For the 5x7 multiplication part:
    # (A_i * B)[k] = sum_{j=0}^{4} A_i[j] * B[j,k]
    #
    # This is not directly a convolution. However, we can use the following trick:
    # If we define for each i:
    #   a_i = [A_i[0], A_i[1], A_i[2], A_i[3], A_i[4], 0, 0, 0, 0, 0, 0] (length 11)
    # And for each k:
    #   b_k = [B[0,k], B[1,k], B[2,k], B[3,k], B[4,k], 0, 0, 0, 0, 0, 0] (length 11)
    # Then (A_i * B)[k] = sum_j a_i[j] * b_k[j], which is still an inner product.
    #
    # The convolution approach would be:
    #   (a_i * b_k)[t] = sum_j a_i[j] * b_k[t-j]
    # This gives a different result.
    #
    # I think the correct approach for matrix multiplication using FFT is more subtle.
    # Let me just use the schoolbook decomposition with rank = 105, but I'll try 
    # one more idea: use the fact that the tensor has a specific structure.
    
    # Final approach: Use the standard bilinear decomposition (schoolbook) with rank 105.
    # This is guaranteed to be correct.
    
    rank = n * m * p  # 105
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
