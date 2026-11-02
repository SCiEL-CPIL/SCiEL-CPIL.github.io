"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """WFT over Z_2 x Z_7 for 2x4x7 matrix multiplication."""
    import numpy as np
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # WFT over G = Z_2 x Z_7 (order 14)
    # Group elements: (a, b) for a in {0,1}, b in {0,...,6}
    # We need to embed the multiplication tensor into a convolution on G.
    
    # The key idea: for matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k],
    # we can view this as a bilinear form. Using WFT, we diagonalize via characters.
    
    # Characters of Z_2 x Z_7: chi_{(u,v)}(a,b) = (-1)^(u*a) * exp(2*pi*i*v*b/7)
    # For exact integer/half-integer entries, we work with the real form.
    
    # Since we need exact integer/half-integer entries, let's use a different approach:
    # We'll use the fact that WFT over Z_7 gives a 7x7 DFT matrix, and handle Z_2 separately.
    
    # Strategy: 
    # 1. Pad A's columns from 4 to 7 (zero-pad 3 entries)
    # 2. Use WFT over Z_7 for the dot products
    # 3. Handle the n=2 dimension with a simple 2x2 transform
    
    # For each row i of A (i in {0,1}) and each column k of B (k in {0,...,6}):
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # = sum_{j=0}^{6} A_padded[i,j] * B[j,k]  (since A_padded[i,4..6]=0)
    
    # Using WFT over Z_7: let F be the 7x7 DFT matrix over Z_7 (or complex)
    # Then sum_j X[j]*Y[j] = (1/7) * sum_v (F*X)[v] * conj(F*Y)[v]
    
    # But we need exact entries. Let's use the integer-friendly Winograd scheme.
    
    # Alternative: use the fact that for prime p=7, we can use the standard
    # Winograd scheme which gives rank p for a dot product of length <= p.
    
    # For n=2, m=4, p=7:
    # We can decompose as:
    # C = A * B where A is 2x4, B is 4x7
    # Pad A to 2x7: A_pad = [A | 0_{2x3}]
    # Then C[i,k] = sum_{j=0}^{6} A_pad[i,j] * B[j,k]
    
    # Using WFT over Z_7 for each (i,k) pair:
    # This gives rank 7 for each i, so total rank 2*7 = 14? No, that's not right.
    
    # Actually, the WFT rank for (n x m) * (m x p) with m <= p and using Z_p is:
    # rank = n * p if we use the full Fourier transform
    
    # Let me construct the WFT decomposition explicitly.
    
    # The WFT over Z_p for matrix multiplication:
    # U[i*m + j, t] = F[t, j] * delta_{i, i_t}  (for appropriate mapping)
    # This is getting complicated. Let me use a direct construction.
    
    # Direct construction using the character table:
    # For G = Z_2 x Z_7, characters are indexed by (u, v) where u in {0,1}, v in {0,...,6}
    # chi_{(u,v)}(a,b) = (-1)^(u*a) * omega^v*b where omega = exp(2*pi*i/7)
    
    # Since we need real/rational entries, let's use the real form of the DFT.
    # For Z_7, the real DFT can be represented using cos and sin, but that gives irrationals.
    
    # Let me try a completely different approach: use the known optimal rank for 2x4x7.
    # The lower bound is max(8, 28, 14) = 28.
    # Can we achieve rank 28?
    
    # Actually, let's use a simple block decomposition:
    # Split B into two 2x7 blocks: B = [B1; B2]
    # A = [A1 | A2] where A1, A2 are 2x2
    # C = A1*B1 + A2*B2
    
    # Each 2x2 * 2x7 can be done with rank 14 (schoolbook) or better.
    # Using Winograd for 2x2*2x2 gives rank 5, so for 2x2*2x7 we can do 7*5 = 35? No.
    
    # Let me just implement the WFT over Z_7 properly.
    
    # For the dot product of two length-4 vectors (padded to 7), using WFT over Z_7:
    # The rank is 7 (the size of the group).
    
    # For 2x4x7: we have 2 rows of A, each requiring a length-4 dot product with each column of B.
    # Using WFT: rank = 2 * 7 = 14? That seems too low.
    
    # I think the correct WFT rank is: for (n x m) * (m x p) with m <= p, using Z_p,
    # the rank is n * p.
    
    # Let me verify: n=2, p=7, so rank = 14.
    
    # Constructing the WFT:
    # Let F be the 7x7 Fourier matrix over the complex numbers.
    # F[v, j] = exp(-2*pi*i*v*j/7) for v, j in {0,...,6}
    
    # For the decomposition:
    # U[i*m + j, (u, v)] = F[v, j] * delta_{i, u}  for u in {0,1}, v in {0,...,6}
    # But this doesn't quite work because we need to match the tensor structure.
    
    # Let me use a simpler, verified approach: the schoolbook with some optimization.
    
    # Actually, let me just return the schoolbook for now to ensure correctness,
    # and then try to improve.
    
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
