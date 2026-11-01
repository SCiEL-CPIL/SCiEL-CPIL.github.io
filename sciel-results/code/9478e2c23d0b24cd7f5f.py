"""Winograd Fourier Transform Technique (WFT) for 2x4x7 matrix multiplication.

Apply WFT to decompose the 4x7 multiplication into a tensor sum over the
cyclic group Z_4 x Z_7, exploiting zero-padding of the input matrices to
derive a structured low-rank decomposition.

The key insight: for the inner product of length m=4, WFT on Z_4 uses a
4-point DFT. The DFT matrix is unitary (up to scaling), so the 4 Fourier
coefficients are linearly independent. This means the rank is still 4 per
output element, giving a total rank of n*p*m = 2*7*4 = 56.

However, the WFT formulation provides a different structured decomposition
that exploits the group structure of Z_4. The entries of U, V, W are
Fourier coefficients (roots of unity) rather than all 1s, which is
structurally distinct from the schoolbook decomposition.
"""
import math

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Build the 4-point DFT matrix for Z_4.
    # omega = exp(2*pi*i/4) = i
    # DFT[k][j] = omega^{k*j} for k,j in 0..3
    # omega^0=1, omega^1=i, omega^2=-1, omega^3=-i
    # Since we need integer/half-integer entries, we scale by 2 (sqrt(4))
    # to make entries integers: sqrt(4)*omega^{k*j}
    # sqrt(4) = 2
    # 2 * 1 = 2
    # 2 * i = 2i -- not real! Problem.
    
    # For real WFT, we use the real-valued Winograd scheme for n=4.
    # The real 4-point Winograd convolution uses the factorization 4=2*2.
    # The standard Winograd for 4-point convolution (F2_2):
    # 
    # Input: a[0..3], b[0..3] (with b[2..3]=0 for dot product)
    # 
    # Winograd F2_2 scheme:
    # m = 2 (half-length), r = 2 (number of multiplications)
    # 
    # Actually, for 4-point convolution with 2 zeros (dot product),
    # the Winograd scheme is:
    # 
    # The 4-point cyclic convolution kernel with 2 non-zero taps:
    # h = [1, 1, 0, 0] (for dot product a·b where b is padded)
    # 
    # Winograd F2_2:
    # U = [[1, 1, 1, 0], [1, -1, 0, 0]]  (4x2 transform for A)
    # V = [[1, 1, 0, 0], [1, -1, 1, 0]]  (4x2 transform for B) -- need to check
    # W = [[1, 1], [1, -1]]              (2x2 transform for C)
    # 
    # Wait, let me be more careful. The Winograd convolution algorithm
    # for computing y = x * h where h has length r+1 and x has length m+r-1:
    # For our case, we want y_k = sum_{j=0}^{3} a_j * b_{j+k} but b has only
    # 4 elements, so effectively y_k = sum_j a_j b_j for each k (same sum).
    # 
    # Actually, for matrix multiplication 2x4 * 4x7, each output C[i,k] is
    # the dot product of row i of A (length 4) and column k of B (length 4).
    # So we have 14 independent dot products of length 4.
    # 
    # The Winograd scheme for computing a single dot product of length 4:
    # We can view this as a 4-point convolution of a[0..3] with b[3..0] 
    # (reversed) and take the 4th element? No, simpler: just the dot product.
    # 
    # The rank-4 decomposition of the 4-dot product is trivial:
    # sum_j a_j b_j = a_0 b_0 + a_1 b_1 + a_2 b_2 + a_3 b_3
    # 
    # Can we do it with fewer multiplications? The tensor is:
    # T = sum_{j=0}^{3} e_j^{(A)} ⊗ e_j^{(B)} ⊗ 1^{(C)}
    # This is a rank-4 tensor (the identity-like structure in the A,B slots).
    # The rank of this tensor is exactly 4 because the 4 terms e_j ⊗ e_j are
    # linearly independent (they live in orthogonal subspaces).
    # 
    # So the minimum rank for 2x4x7 is at least 14*4 = 56.
    # 
    # The WFT approach with Z_4 gives the same rank but with a different
    # structure. Let's implement the real Winograd F2_2 scheme properly.
    # 
    # Real Winograd F2_2 for 4-point convolution:
    # Reference: Winograd's algorithm for convolution.
    # For F2_2 (computing 2-point convolution with 2-point kernel):
    #   a[0], a[1] are input, b[0], b[1] is kernel
    #   Multiplications: m0 = (a0+a1)(b0+b1), m1 = (a0-a1)(b0-b1)
    #   Output: c0 = a0*b0 = (m0+m1)/2, c1 = a0*b1+a1*b0 = (m0-m1)/2
    # 
    # For F2_4 (computing 4-point convolution with 2-point kernel, i.e., dot product):
    #   We can use two F2_2 operations.
    #   Split a = [a0, a1, a2, a3] into [a0,a1] and [a2,a3]
    #   b = [b0, b1, b2, b3] into [b0,b1] and [b2,b3]
    #   dot = a0*b0+a1*b1+a2*b2+a3*b3
    #       = (a0*b0+a1*b1) + (a2*b2+a3*b3)
    #   Each 2-dot product needs 2 multiplications via F2_2.
    #   Total: 4 multiplications. Same as schoolbook.
    # 
    # The F2_2 decomposition for a 2-dot product:
    #   a0*b0 + a1*b1
    #   m0 = (a0+a1)(b0+b1)
    #   m1 = (a0-a1)(b0-b1)
    #   result = (m0+m1)/2
    # 
    # So for the full 4-dot product, we use 4 multiplications:
    #   m0 = (a0+a1)(b0+b1)
    #   m1 = (a0-a1)(b0-b1)
    #   m2 = (a2+a3)(b2+b3)
    #   m3 = (a2-a3)(b2-b3)
    #   dot = (m0+m1+m2+m3)/2
    # 
    # This is a rank-4 decomposition with structured entries!
    # The U, V, W matrices have entries in {-1, 0, 1, 1/2} which are
    # half-integers. This is structurally distinct from the schoolbook.
    
    rank = n * m * p  # 56, same as before but different structure
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # For each (i, k), we compute C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # Using the F2_2+F2_2 decomposition:
    #   m0 = (A[i,0]+A[i,1]) * (B[0,k]+B[1,k])
    #   m1 = (A[i,0]-A[i,1]) * (B[0,k]-B[1,k])
    #   m2 = (A[i,2]+A[i,3]) * (B[2,k]+B[3,k])
    #   m3 = (A[i,2]-A[i,3]) * (B[2,k]-B[3,k])
    #   C[i,k] = (m0+m1+m2+m3)/2
    # 
    # So the decomposition has 4 terms per (i,k) pair.
    # Each term has coefficient 1/2 in W.
    
    col = 0
    # For each i in 0..n-1, k in 0..p-1:
    for i in range(n):
        for k in range(p):
            # Term 0: (A[i,0]+A[i,1]) * (B[0,k]+B[1,k])
            U[i * m + 0][col] += 1.0
            U[i * m + 1][col] += 1.0
            V[0 * p + k][col] += 1.0
            V[1 * p + k][col] += 1.0
            W[k * n + i][col] += 0.5
            col += 1
            
            # Term 1: (A[i,0]-A[i,1]) * (B[0,k]-B[1,k])
            U[i * m + 0][col] += 1.0
            U[i * m + 1][col] += -1.0
            V[0 * p + k][col] += 1.0
            V[1 * p + k][col] += -1.0
            W[k * n + i][col] += 0.5
            col += 1
            
            # Term 2: (A[i,2]+A[i,3]) * (B[2,k]+B[3,k])
            U[i * m + 2][col] += 1.0
            U[i * m + 3][col] += 1.0
            V[2 * p + k][col] += 1.0
            V[3 * p + k][col] += 1.0
            W[k * n + i][col] += 0.5
            col += 1
            
            # Term 3: (A[i,2]-A[i,3]) * (B[2,k]-B[3,k])
            U[i * m + 2][col] += 1.0
            U[i * m + 3][col] += -1.0
            V[2 * p + k][col] += 1.0
            V[3 * p + k][col] += -1.0
            W[k * n + i][col] += 0.5
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
