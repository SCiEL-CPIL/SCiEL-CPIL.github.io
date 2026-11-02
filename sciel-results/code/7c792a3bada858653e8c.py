"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Block-circulant embedding strategy:
    # Embed the 5x7 multiplication into a cyclic convolution of size N = m + p - 1 = 11
    # The tensor T[i,j,k] = 1 for i<3, j<5, k<7
    # 
    # Key insight: For each fixed i, the operation sum_j A[i,j] * B[j,k] can be
    # viewed as evaluating a bilinear form. We use the DFT structure of Z_N where
    # N >= m + p - 1 to avoid wrap-around.
    #
    # We use N = 12 (next number >= 11 with good factorization for exact arithmetic)
    # The DFT of size 12 over the rationals can be done exactly using roots of unity
    # in a number field, but for simplicity we use a different approach.
    #
    # Alternative: Use the fact that we can decompose using the polynomial method.
    # The product of two polynomials of degree <5 and <7 has degree <12.
    # We evaluate at 12 points and interpolate, giving 12 rank-1 terms per i.
    # Total rank: 3 * 12 = 36.
    #
    # But we can do better! Use the structure more carefully.
    #
    # Let me use a direct construction with the cyclic group Z_12.
    # The DFT matrix F of size 12 has entries F[a,b] = omega^(a*b) where
    # omega = exp(2*pi*i/12). We work over the reals with exact rational
    # approximations where possible.
    
    N = 12  # Size of cyclic convolution (>= m + p - 1 = 11)
    
    # Precompute the DFT matrix using exact trigonometric values
    # omega = exp(2*pi*i/N)
    # We'll use the property that the DFT diagonalizes circulant matrices
    
    # Construct the decomposition:
    # For each i in [0,n) and each frequency f in [0,N):
    #   We have a rank-1 term
    #   U[i*m + j, c] = 1 for all j
    #   V[j*p + k, c] = omega^(-f*j) for j < m, k < p  
    #   W[k*n + i, c] = omega^(f*k) for k < p
    #
    # But this gives a convolution, not exact multiplication.
    # The issue: we need to handle the truncation j<5, k<7.
    #
    # Better approach: Use the polynomial evaluation/interpolation method.
    # For each i, compute sum_j A[i,j] * B[j,k] by:
    # 1. Form polynomial a(x) = sum_j A[i,j] * x^j (degree < 5)
    # 2. Form polynomial b(x) = sum_k B[j,k] * x^k (degree < 7) for each j
    # 3. The product has degree < 12
    #
    # Actually, let me use a cleaner construction:
    # The tensor can be decomposed as:
    # T[i,j,k] = sum_f (1/N) * omega^(-f*j) * omega^(f*k) * delta_valid(i,j,k)
    # where delta_valid accounts for the truncation.
    #
    # For exactness, I'll use the following verified construction:
    # Use the schoolbook as a baseline but try to reduce using structure.
    
    # Let me use a different strategy: exploit that n=3 is small.
    # For each of the 3 rows of A, we need to compute a weighted sum of rows of B.
    # The weights are A[i,j] for j=0..4.
    # 
    # We can precompute 5 linear combinations of B's rows, but that's just schoolbook.
    # 
    # To reduce rank, we need to find a non-trivial bilinear decomposition.
    # 
    # Let me try: use the fact that the multiplication tensor for 3x5x7 can be
    # embedded in a larger tensor with more structure.
    
    # Final approach: Use the DFT-based construction with N=12, but verify
    # that it's exact for the truncated case.
    
    # The key: for the cyclic convolution of size N >= m+p-1, the first (m+p-1)
    # coefficients of the cyclic convolution match the linear convolution.
    # Since we only need k < p = 7 < N - m + 1 = 8, we're safe.
    
    # Construct using DFT of size N=12:
    # omega = exp(2*pi*i/12)
    
    # For exact arithmetic, use the minimal polynomial of omega.
    # omega = exp(pi*i/6) = cos(pi/6) + i*sin(pi/6) = sqrt(3)/2 + i/2
    # This is in Q(sqrt(3), i)
    
    # For simplicity, let me use a numerical construction and verify it's exact
    # by checking that the reconstruction gives integer/half-integer entries.
    
    # Actually, let me use a completely different approach that's guaranteed exact:
    # Use the standard bilinear decomposition but group terms using the
    # circulant structure to reduce the effective rank.
    
    # The rank of the 3x5x7 multiplication tensor is at most 105.
    # Let me try to achieve a lower rank using the following:
    # 
    # Split the 5x7 matrix B into blocks and use smaller decompositions.
    # But this is complex.
    
    # Given the constraints, let me use a verified construction:
    # The rank is at most (n-1)*(m-1)*(p-1) + n*m + m*p + n*p - n - m - p + 2
    # for some structured decompositions, but I'm not sure of the exact formula.
    
    # Let me just use the schoolbook for now, which is guaranteed correct.
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
