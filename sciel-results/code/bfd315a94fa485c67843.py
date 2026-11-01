"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # Winograd FFT technique for middle dimension m=3
    # Use DFT over Z_3: omega = e^(2*pi*i/3)
    # Rank = n * p * m_fft where m_fft = 3 (same as m for prime 3, but structure differs)
    # Actually for FFT, we need m_fft points. For m=3, we use 3 points.
    # The key insight: instead of m separate multiplications per (i,k), 
    # we use m_fft complex multiplications that are "parallelized" via DFT.
    # For m=3: rank = n * p * m_fft = 2 * 7 * 3 = 42? No...
    # 
    # Let me think more carefully. The tensor is T[i,j,k] = 1 if the multiplication 
    # involves A[i,j] * B[j,k] -> C[i,k].
    # 
    # Winograd: For each fixed i,k, we compute sum_j A[i,j]*B[j,k].
    # Using DFT: let omega = e^(2*pi*i/3)
    # Define for each frequency l in {0,1,2}:
    #   U_l[i] = sum_j A[i,j] * omega^(j*l)  (but this doesn't quite work directly)
    #
    # Actually, the standard Winograd approach for matrix mult:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Using the fact that multiplication in Z_3[x]/(x^3-1) can be done via DFT:
    # If we define polynomials a(x) = sum_j A[i,j] x^j and b(x) = sum_k B[j,k] x^k...
    # Wait, B is indexed by (j,k), not (j) alone.
    #
    # Let me reconsider. The standard Winograd for n×m × m×p:
    # For each i (row of A) and k (col of B), we need dot product of A[i,:] and B[:,k]
    # 
    # FFT-based: Use m_fft >= 2m-1 points for full convolution, but for cyclic 
    # convolution of length m, we need exactly m points if m is prime.
    #
    # For m=3 (prime), we can use 3-point DFT:
    # omega = e^(2*pi*i/3)
    # 
    # For each i and k:
    #   Let a_j = A[i,j], b_j = B[j,k] for j=0,1,2
    #   C[i,k] = sum_j a_j * b_j
    #
    # Using DFT: 
    #   A_hat[l] = sum_j a_j * omega^(-j*l) for l=0,1,2
    #   B_hat[l] = sum_j b_j * omega^(j*l) for l=0,1,2
    #   Then sum_j a_j * b_j = (1/3) * sum_l A_hat[l] * B_hat[l]
    #
    # So for each (i,k) pair, we need 3 "multiplications" A_hat[l]*B_hat[l], 
    # but these can be shared across different i's and k's!
    #
    # Specifically:
    #   For each i and l: compute A_hat[i,l] = sum_j A[i,j] * omega^(-j*l)
    #   For each k and l: compute B_hat[k,l] = sum_j B[j,k] * omega^(j*l)
    #   Then C[i,k] = (1/3) * sum_l A_hat[i,l] * B_hat[k,l]
    #
    # The rank-1 terms are: for each (i,k,l), the product A_hat[i,l] * B_hat[k,l]
    # Total rank = n * p * m_fft = 2 * 7 * 3 = 42
    #
    # Hmm, that's the same as schoolbook. Let me reconsider.
    #
    # Actually, I think the savings come from the fact that in the schoolbook 
    # decomposition, each (i,j,k) triple gives one rank-1 term. With FFT, 
    # we still have n*p*m_fft terms, which for m=3 is the same.
    #
    # Wait - I think I'm confusing myself. Let me look at this differently.
    #
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 (all ones 
    # in the "multiplication" pattern). The rank of this tensor is what we want.
    #
    # For 2×3×7, the schoolbook rank is 2*3*7 = 42.
    #
    # Winograd's FFT technique can reduce this when m is composite or when 
    # we can exploit structure. For m=3 (prime), the FFT approach gives:
    # rank = n * p * m_fft where m_fft is the number of FFT points needed.
    # For cyclic convolution of length m, we need m points if m is prime.
    # So rank = 2 * 7 * 3 = 42. Same as schoolbook.
    #
    # However, if we use a different approach: since m=3 is small, we might 
    # be able to use a sparse decomposition. Let me try a different strategy.
    #
    # Alternative: Use the fact that we only need half-integer entries.
    # Try rank 28 = 2 * 7 * 2, using 2 multiplications per (i,k) pair.
    #
    # For m=3, we can write:
    #   sum_{j=0}^{2} A[i,j] * B[j,k]
    #   = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Can we express this with 2 multiplications? Let's try:
    #   Let u = A[i,0] + A[i,1], v = B[0,k] + B[1,k]
    #   Then u*v = A[i,0]*B[0,k] + A[i,0]*B[1,k] + A[i,1]*B[0,k] + A[i,1]*B[1,k]
    #   We need A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #   Difference: u*v - target = A[i,0]*B[1,k] + A[i,1]*B[0,k] - A[i,2]*B[2,k]
    #   Not obviously factorable.
    #
    # Let me just implement the FFT approach with rank 42 and verify it works,
    # then try to find a lower rank.
    
    # FFT decomposition with m_fft = 3
    m_fft = 3
    rank = n * p * m_fft
    
    # omega = e^(2*pi*i/3)
    # omega^0 = 1, omega^1 = -1/2 + i*sqrt(3)/2, omega^2 = -1/2 - i*sqrt(3)/2
    # For exact arithmetic, we'll use the minimal polynomial approach.
    # omega satisfies omega^2 + omega + 1 = 0, so omega^2 = -omega - 1
    
    # U: shape (n*m, rank) = (6, 42)
    # V: shape (m*p, rank) = (21, 42)  
    # W: shape (p*n, rank) = (14, 42)
    
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    # Use fractions for exact arithmetic
    from fractions import Fraction
    
    # omega = -1/2 + sqrt(3)/2 * i, but we need to handle complex numbers.
    # Since we need integer/half-integer entries, we can't use complex numbers directly.
    # 
    # Alternative: Use the fact that the DFT matrix for m=3 has entries that are
    # algebraic numbers. We can work in the ring Z[omega] where omega^2 + omega + 1 = 0.
    # 
    # But the output requires integer/half-integer entries, which suggests we should
    # find a real decomposition.
    #
    # Let me try a different approach: use a real-valued sparse decomposition.
    #
    # For m=3, consider:
    #   A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Try: 
    #   t1 = (A[i,0] + A[i,1]) * (B[0,k] + B[1,k])
    #   t2 = (A[i,0] - A[i,1]) * (B[0,k] - B[1,k])
    #   t3 = A[i,2] * B[2,k]
    #
    # Then: t1 + t2 = 2*(A[i,0]*B[0,k] + A[i,1]*B[1,k])
    # So: A[i,0]*B[0,k] + A[i,1]*B[1,k] = (t1 + t2) / 2
    # And we need to add A[i,2]*B[2,k]
    #
    # This gives 3 multiplications per (i,k), same as schoolbook.
    #
    # Try 2 multiplications:
    #   t1 = (A[i,0] + A[i,1] + A[i,2]) * (B[0,k] + B[1,k] + B[2,k])
    #   t2 = ...
    #
    # This is getting complicated. Let me just implement the schoolbook and return it,
    # as the FFT approach for m=3 prime doesn't obviously reduce rank.
    
    # Actually, let me try rank = n * p * 2 = 28 and see if it's possible.
    # If not, fall back to schoolbook rank 42.
    
    # For now, return schoolbook to ensure correctness.
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
