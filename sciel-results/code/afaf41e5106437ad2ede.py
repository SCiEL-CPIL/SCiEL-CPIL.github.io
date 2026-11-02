"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    WFT over Z_2 x Z_7 for 2x4x7 matrix multiplication.
    
    We use the group G = Z_2 x Z_7 with 14 elements.
    The key insight: we embed the 4-dimensional middle dimension into Z_7 by zero-padding,
    and use the Z_2 structure for the 2-dimensional output.
    
    For the multiplication C = A * B where A is 2x4, B is 4x7, C is 2x7:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We map j in {0,1,2,3} to elements of Z_7 (zero-padding positions 4,5,6).
    The group G = Z_2 x Z_7 has characters indexed by (u,v) where u in Z_2, v in Z_7.
    
    The WFT decomposition gives rank equal to the number of non-zero Fourier coefficients,
    which for this structured problem should be less than 56 (schoolbook).
    
    The construction:
    - U has shape (n*m, rank) = (8, rank)
    - V has shape (m*p, rank) = (28, rank)  
    - W has shape (p*n, rank) = (14, rank)
    
    We use the fact that for each character (u,v), we get one rank-1 term.
    The entries involve roots of unity, but for Z_2 x Z_7 we can work with
    algebraic numbers. However, the task requires integer/half-integer entries.
    
    Let me reconsider: the standard WFT over finite fields or rings that contain
    the necessary roots of unity. For Z_7, we need a 7th root of unity.
    This doesn't exist in Q, so we need to extend.
    
    Alternative approach: Use a different decomposition that stays in Q.
    
    Let me use the following strategy:
    1. Split the 4 middle dimensions into two groups of 2.
    2. Use the Winograd scheme for 2x2 * 2x7 subproblems.
    
    For 2x2 * 2x7:
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    
    Using Winograd for the 2x2 part:
    Define:
    a0 = A[0,0] + A[0,1]
    a1 = A[1,0] + A[1,1]  
    a2 = A[0,0] - A[0,1]
    a3 = A[1,0] - A[1,1]
    
    b0_k = B[0,k] + B[1,k]
    b1_k = B[0,k] - B[1,k]
    
    Then:
    C[0,k] = (a0*b0_k + a2*b1_k) / 2
    C[1,k] = (a1*b0_k + a3*b1_k) / 2
    
    This gives 4 multiplications per k, so for 7 values of k, that's 28 multiplications.
    Same as schoolbook for 2x2*2x7.
    
    Let me try a different approach. Use the fact that we can share computations
    across the two blocks.
    
    Actually, let me use a proper WFT construction over a ring extension.
    Since we need integer/half-integer entries, let me use a different strategy.
    
    Key insight: For n=2, we can use the following decomposition:
    
    C[0,k] = sum_j A[0,j] * B[j,k]
    C[1,k] = sum_j A[1,j] * B[j,k]
    
    Define S_j = A[0,j] + A[1,j] and D_j = A[0,j] - A[1,j]
    Then:
    A[0,j] = (S_j + D_j) / 2
    A[1,j] = (S_j - D_j) / 2
    
    So:
    C[0,k] = (1/2) * sum_j (S_j + D_j) * B[j,k]
           = (1/2) * (sum_j S_j * B[j,k] + sum_j D_j * B[j,k])
    C[1,k] = (1/2) * sum_j (S_j - D_j) * B[j,k]
           = (1/2) * (sum_j S_j * B[j,k] - sum_j D_j * B[j,k])
    
    Let X_k = sum_j S_j * B[j,k] and Y_k = sum_j D_j * B[j,k]
    Then C[0,k] = (X_k + Y_k) / 2 and C[1,k] = (X_k - Y_k) / 2
    
    But computing X_k and Y_k still requires 4 multiplications each (for j=0..3),
    so 8 multiplications per k, total 56. No improvement.
    
    Let me try yet another approach. Use the structure of m=4.
    
    For m=4, we can use a 4-point Winograd scheme:
    a0 = A[i,0] + A[i,1] + A[i,2] + A[i,3]
    a1 = A[i,0] + A[i,1] - A[i,2] - A[i,3]
    a2 = A[i,0] - A[i,1] + A[i,2] - A[i,3]
    a3 = A[i,0] - A[i,1] - A[i,2] + A[i,3]
    
    Similarly for B:
    b0_k = B[0,k] + B[1,k] + B[2,k] + B[3,k]
    b1_k = B[0,k] + B[1,k] - B[2,k] - B[3,k]
    b2_k = B[0,k] - B[1,k] + B[2,k] - B[3,k]
    b3_k = B[0,k] - B[1,k] - B[2,k] + B[3,k]
    
    Then C[i,k] = (1/4) * (a0*b0_k + a1*b1_k + a2*b2_k + a3*b3_k)
    
    This gives 4 multiplications per (i,k) pair, so 2*7*4 = 56 multiplications.
    Still schoolbook.
    
    The issue is that the Winograd scheme for m=4 doesn't reduce the rank below m.
    We need to exploit the fact that n=2 is small.
    
    Let me combine the n=2 and m=4 structures:
    
    For each i in {0,1} and k in {0,...,6}:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Define for each i:
    S_i = A[i,0] + A[i,1] + A[i,2] + A[i,3]
    D_i = A[i,0] + A[i,1] - A[i,2] - A[i,3]
    E_i = A[i,0] - A[i,1] + A[i,2] - A[i,3]
    F_i = A[i,0] - A[i,1] - A[i,2] + A[i,3]
    
    And for each k:
    s_k = B[0,k] + B[1,k] + B[2,k] + B[3,k]
    d_k = B[0,k] + B[1,k] - B[2,k] - B[3,k]
    e_k = B[0,k] - B[1,k] + B[2,k] - B[3,k]
    f_k = B[0,k] - B[1,k] - B[2,k] + B[3,k]
    
    Then C[i,k] = (1/4) * (S_i*s_k + D_i*d_k + E_i*e_k + F_i*f_k)
    
    This is 4 multiplications per (i,k), total 56. No gain.
    
    I think the schoolbook rank of 56 might be optimal for this problem, or at least
    very hard to beat with simple techniques. Let me just return the schoolbook
    decomposition as a safe baseline.
    
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 56
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
