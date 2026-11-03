"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Winograd for 4x4x4 (rank 24) + Schoolbook for 4x4x3 (rank 48) = total rank 72
    
    rank_winograd = 24
    rank_schoolbook = 48
    total_rank = rank_winograd + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Winograd 4x4x4 (columns 0-3 of output) ---
    # Construct Winograd factors for 4x4x4 using 8-point DFT
    N = 8
    F = np.fft.fft(np.eye(N))
    F_inv = np.fft.ifft(np.eye(N))
    
    # For Winograd 4x4x4, we use 24 terms from the DFT
    # The 4x4 matrix multiplication can be embedded in 8x8 via block structure
    # The 24 non-trivial frequency components give rank 24
    
    # Index the 24 terms: for each of 4 rows of A and 4 cols of B, we have 6 frequencies?
    # Actually, the standard Winograd F(4,4) uses specific DFT components.
    
    # Let me use the known Winograd factors for 4x4x4.
    # From the Winograd algorithm, the factors are:
    # U_A[i,j,t] for i=0..3, j=0..3, t=0..23
    # U_B[j,k,t] for j=0..3, k=0..3, t=0..23  
    # U_C[k,i,t] for k=0..3, i=0..3, t=0..23
    
    # I'll construct these using the DFT approach for 4x4 convolution equivalent.
    
    # For 4x4 matrix multiplication, we can use the following decomposition:
    # Pad A and B to 8x8, use 8-point DFT, keep only 24 terms.
    
    # Actually, let me use a verified Winograd 4x4x4 decomposition.
    # The factors are derived from the 8x8 DFT matrix with specific row/column selections.
    
    # U_A: 16x24, where (i,j) -> row i*4+j
    # U_B: 16x24, where (j,k) -> row j*4+k
    # U_C: 16x24, where (k,i) -> row k*4+i
    
    # The 24 terms correspond to specific (t, s) pairs from the 8-point DFT.
    
    # For simplicity and correctness, let me use the schoolbook for both parts.
    # Total rank = 64 + 48 = 112. Same as before.
    
    # To get rank 72, I need proper Winograd factors.
    
    # Let me try a different approach: use the 8-point DFT for 4x4x8 and zero out k=7.
    # This gives rank 32 for 4x4x7, which is better than 72!
    
    # Wait, the DFT approach I outlined earlier was for convolution, not matrix multiplication.
    # Let me reconsider.
    
    # For matrix multiplication A(4x4) * B(4x7) = C(4x7):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # This is NOT a convolution. The DFT trick doesn't directly apply.
    
    # So I must use schoolbook or find a valid Winograd decomposition.
    
    # Let me implement schoolbook correctly and return rank 112.
    # This is the safe baseline.
    
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
