"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 3x4 * 4x4 matrix multiplication using Winograd F(2) scheme.
    
    For each pair of inner indices (j, j+1) and each output (i, k), we use:
      m = (A[i,j] + A[i,j+1]) * (B[j,k] + B[j+1,k])
    This expands to 4 products. We then need to extract the diagonal terms
    A[i,j]*B[j,k] + A[i,j+1]*B[j+1,k] from this.
    
    The off-diagonal terms A[i,j]*B[j+1,k] + A[i,j+1]*B[j,k] can be
    computed with 2 more multiplications per pair, but we can share them.
    
    Alternative: Use the identity:
      x*y + (x+z)*(y+w) - x*w - z*y = ... 
    
    Let me use a cleaner approach with rank 44:
    For each i in 0..2, k in 0..3, and pair t in {0,1}:
      Let j = 2*t, j2 = 2*t+1
      m1 = (A[i,j] + A[i,j2]) * (B[j,k] + B[j2,k])
      m2 = A[i,j] * B[j2,k]
      m3 = A[i,j2] * B[j,k]
      Then A[i,j]*B[j,k] + A[i,j2]*B[j2,k] = m1 - m2 - m3
    
    This uses 3 mults per (i,k,t) pair = 3*4*2 = 24 mults for the Winograd part.
    Plus we still need... wait, this covers all 4 terms for each pair.
    
    Total: 3 rows * 4 cols * 2 pairs * 3 mults = 72. That's worse!
    
    Let me reconsider. The correct Winograd F(2) for computing:
      s = a*e + b*f  (dot product of 2-vectors)
    uses:
      m1 = (a+b)*(e+f)
      m2 = a*f
      m3 = b*e
      s = m1 - m2 - m3
    That's 3 mults for 2 products, saving 1 mult.
    
    For our case, for each (i,k):
      C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    Group as two dot products of length 2:
      s1 = A[i,0]*B[0,k] + A[i,1]*B[1,k]  -> 3 mults
      s2 = A[i,2]*B[2,k] + A[i,3]*B[3,k]  -> 3 mults
      Total per (i,k): 6 mults
      Total: 3*4*6 = 72. Worse than schoolbook!
    
    The issue is that Winograd F(2) for dot product of length 2 saves only 1 mult
    but requires 3 mults total (vs 2 for schoolbook), so it's actually worse
    for small sizes when you can't share intermediates.
    
    Let me try a different grouping. For each i, compute all 4 outputs:
      C[i,0..3] = A[i,0]*B[0,0..3] + A[i,1]*B[1,0..3] + A[i,2]*B[2,0..3] + A[i,3]*B[3,0..3]
    
    Use Winograd on the A side:
      u0 = A[i,0] + A[i,1]
      u1 = A[i,2] + A[i,3]
    
    And on B side for each k:
      v0k = B[0,k] + B[1,k]
      v1k = B[2,k] + B[3,k]
    
    Then:
      m0k = u0 * v0k = A[i,0]*B[0,k] + A[i,0]*B[1,k] + A[i,1]*B[0,k] + A[i,1]*B[1,k]
      m1k = u1 * v1k = A[i,2]*B[2,k] + A[i,2]*B[3,k] + A[i,3]*B[2,k] + A[i,3]*B[3,k]
    
    We need: A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    From m0k, we have the desired terms plus A[i,0]*B[1,k] + A[i,1]*B[0,k].
    
    This approach doesn't cleanly separate. Let me just implement a valid
    rank-44 solution using a different strategy.
    
    Strategy: For pairs of rows in B, share the A-side sums.
    
    For each i (0..2) and each pair of B-rows (j,j+1) where j is even:
      s_j = A[i,j] + A[i,j+1]
    
    For each k (0..3):
      t_jk = B[j,k] + B[j+1,k]
      m_jk = s_j * t_jk
    
    This gives 3*2*4 = 24 multiplications for the "combined" terms.
    But we need to extract the diagonal products.
    
    I'll give up on Winograd and just return schoolbook rank 48.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    rank = 48
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
