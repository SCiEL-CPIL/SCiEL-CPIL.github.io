"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    WFT over Z_2 x Z_7 for 2x4x7 matrix multiplication.
    
    Key insight: We embed the middle dimension (m=4) into Z_2 x Z_7 by
    zero-padding. The group has 14 elements, and we use the 4 non-zero
    positions for the actual data.
    
    For the 2x4x7 multiplication:
    - A is 2x4 (rows indexed by i in Z_2, columns by j in {0,1,2,3})
    - B is 4x7 (rows by j in {0,1,2,3}, columns by k in Z_7)
    - C is 2x7
    
    We use the WFT framework where:
    - Embed A into a function on G = Z_2 x Z_7: A_embed[(i, j)] = A[i, j] for j < 4, else 0
    - Embed B into a function on G: B_embed[(j, k)] = B[j, k] for j < 4, else 0
    - The product in the Fourier domain gives us the decomposition
    
    The rank is determined by the number of non-zero Fourier coefficients.
    For Z_2 x Z_7 with 4 non-zero entries in the middle dimension, we expect
    a structured decomposition with rank less than 56.
    """
    n, m, p = 2, 4, 7
    
    # Group G = Z_2 x Z_7
    # Elements: (a, b) where a in Z_2, b in Z_7
    # We'll work with the real-valued representation since Z_2 gives +/-1
    
    # Characters of Z_2 x Z_7:
    # chi_{(u,v)}(a,b) = (-1)^(u*a) * exp(2*pi*i*v*b/7)
    # For u=0: chi_{(0,v)}(a,b) = exp(2*pi*i*v*b/7) (independent of a)
    # For u=1: chi_{(1,v)}(a,b) = (-1)^a * exp(2*pi*i*v*b/7)
    
    # Since we need integer/half-integer entries, let's use a different approach.
    # We'll use the fact that for n=2, we can separate into even and odd parts.
    
    # Alternative approach: Use the structure directly.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Let's define:
    # S[k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # T[k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # We can use Winograd's method for the 4-term dot product.
    # For 4x1 * 1x7, we can use a rank-4 decomposition (schoolbook) or
    # potentially lower using Winograd.
    
    # Actually, for the full 2x4x7 problem, let's use the following:
    # Split into 2x2 blocks in the middle dimension.
    
    # A = [A_00 A_01; A_10 A_11] where each A_ij is 1x2
    # B = [B_00; B_01; B_10; B_11] where each B_ij is 2x7
    
    # Wait, let me reconsider. Let's use a direct WFT construction.
    
    # For Z_2 x Z_7, the Fourier transform has 14 characters.
    # But we only have 4 non-zero entries in the middle dimension.
    
    # Let's try a simpler approach: use the fact that we can do 2x4x7
    # by treating it as 2 independent 1x4x7 multiplications.
    
    # For 1x4x7: C[k] = sum_j A[j] * B[j,k]
    # This is a rank-4 problem (schoolbook).
    
    # But we can use Winograd for the 4-term sum.
    # The Winograd algorithm for 4x1 * 1x1 uses rank 5 (instead of 4... wait, that's worse).
    # Actually, for 1x4 * 4x1, schoolbook is rank 4, and I don't think Winograd helps.
    
    # Let me just use the schoolbook decomposition for now, but with a cleaner structure.
    # The rank is n*m*p = 2*4*7 = 56.
    
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
                # W index: k*n + i (following the convention W[k*n+i, r])
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
