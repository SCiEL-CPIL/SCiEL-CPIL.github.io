"""Program for 3x3x3 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
      """
      Returns a rank-27 (schoolbook) decomposition of the 3x3x3 matrix multiplication tensor.
      This is the standard, known-correct decomposition.
      """
      n, m, p = DIMENSIONS
      rank = 27
      
      U = [[0] * rank for _ in range(n * m)]
      V = [[0] * rank for _ in range(m * p)]
      W = [[0] * rank for _ in range(p * n)]
      
      col = 0
      for i in range(n):
          for j in range(m):
              for k in range(p):
                  # U index: i, j -> i*m + j
                  # V index: j, k -> j*p + k
                  # W index: k, i -> k*n + i (Note: Task says W[k*n+i, r])
                  
                  U[i * m + j][col] = 1
                  V[j * p + k][col] = 1
                  W[k * n + i][col] = 1
                  col += 1
                  
      return {"U": U, "V": V, "W": W}
  # EVOLVE-BLOCK-END
