"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 3, 5, 7
    
    # Block-circulant embedding strategy:
    # Embed the 3x5x7 multiplication into a larger structure using cyclic convolution.
    # We use a DFT-based decomposition where the cyclic convolution is diagonalized.
    
    # The key insight: matrix multiplication (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # can be viewed as a bilinear form. We can use the structure of cyclic
    # convolution to reduce the rank.
    
    # Strategy: Use the fact that we can embed the multiplication into a 
    # circulant structure and use DFT to diagonalize it.
    
    # For the 5x7 part, we can use a 7-point cyclic convolution structure.
    # The DFT of size 7 gives 7 orthogonal frequency components.
    
    # However, we need to be careful about the embedding and the exact decomposition.
    
    # Let me use a verified approach: the rank of 3x5x7 multiplication can be 
    # reduced using structured decompositions.
    
    # Use the following construction:
    # 1. For each row i of A (i in [0,3)), we compute the product with B.
    # 2. The product (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # 3. We can use the DFT to diagonalize the convolution-like structure.
    
    # Let's use a 7-point DFT for the k dimension (since p=7).
    # omega = exp(2*pi*i/7)
    
    # The DFT matrix F of size 7: F[a,b] = omega^(a*b)
    # F^(-1) = (1/7) * conj(F)^T
    
    # For each i in [0,3) and each frequency f in [0,7):
    # We can write the product using the DFT basis.
    
    # Actually, let me use a cleaner approach:
    # The multiplication tensor T[i,j,k] = 1 for valid (i,j,k)
    # We want: T = sum_c U_c ⊗ V_c ⊗ W_c
    
    # Using the DFT structure:
    # T[i,j,k] = sum_f (1/7) * conj(omega^(f*j)) * omega^(f*k) * delta[i,i]
    # But this doesn't quite work because we need T[i,j,k] = 1, not a delta.
    
    # Let me use a different approach: exploit the fact that we can group
    # the (j,k) pairs by their sum modulo 7.
    
    # For a cyclic convolution of size 7, we have:
    # (a * b)[k] = sum_j a[j] * b[k-j mod 7]
    
    # This is a rank-7 decomposition using the DFT basis.
    
    # For our case, the multiplication is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can view B as a matrix where each row j is a vector in R^7.
    # The product with A[i,:] gives a linear combination of the rows of B.
    
    # Using the DFT:
    # Let B_hat[f] = DFT(B[:, f]) for each frequency f
    # Then (A*B)[i,k] can be computed using the DFT of B.
    
    # But this still doesn't directly give a lower-rank decomposition of the tensor.
    
    # Let me use a known result: the rank of m×n×p matrix multiplication is at most
    # mn + np + mp - 2 for certain structures, but this is for specific cases.
    
    # For 3x5x7, let me try a direct construction using the DFT of size 7.
    
    N = 7
    omega = np.exp(2j * np.pi / N)
    
    # DFT matrix
    F = np.array([[omega**(a*b) for b in range(N)] for a in range(N)])
    F_inv = np.conj(F).T / N
    
    # Construct the decomposition:
    # rank = n * N = 3 * 7 = 21
    # For each i in [0,3) and f in [0,7):
    #   U[i*m + j, c] = 1 for all j (this is the A part)
    #   V[j*p + k, c] = conj(omega^(f*j)) / sqrt(N) (this is the B part, DFT)
    #   W[k*n + i, c] = omega^(f*k) / sqrt(N) (this is the output part, IDFT)
    
    # But this gives a convolution, not exact multiplication.
    
    # Let me verify: the sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = sum_f (1/N) * conj(omega^(f*j)) * omega^(f*k)
    # = delta_{j,k} (by orthogonality of DFT)
    
    # But we need T[i,j,k] = 1 for all valid (i,j,k), not delta_{j,k}.
    
    # So this approach doesn't work directly.
    
    # Let me use a different strategy: use the schoolbook but with a small optimization.
    # The rank is at most 105, and I'll try to reduce it slightly.
    
    # Actually, let me use a verified lower-rank construction.
    # For 3x5x7, a known upper bound is less than 105.
    
    # Use the following: decompose using 2x2 blocks and apply Strassen-like ideas.
    
    # Split the 5x7 matrix B into blocks:
    # B = [B11 B12; B21 B22] where B11 is 3x4, B12 is 3x3, B21 is 2x4, B22 is 2x3
    
    # But this doesn't directly help with the tensor rank.
    
    # Given the complexity, let me use a clean construction that's guaranteed to work:
    # The schoolbook decomposition with rank = n*m*p = 105.
    
    # But I'll try one more thing: use the fact that we can combine some terms.
    
    # For now, return the schoolbook decomposition.
    
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
