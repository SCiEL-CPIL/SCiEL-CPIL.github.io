  """Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
  DIMENSIONS = [3, 3, 3]

  # EVOLVE-BLOCK-START
  def construct():
      n, m, p = DIMENSIONS
      rank = 19
      U = [[0] * rank for _ in range(n * m)]
      V = [[0] * rank for _ in range(m * p)]
      W = [[0] * rank for _ in range(p * n)]
      
      # Strassen 3x3 Rank 19 Decomposition
      # Indices:
      # A: 0 1 2
      #   3 4 5
      #   6 7 8
      # B: 0 1 2
      #   3 4 5
      #   6 7 8
      # W: k*3+i
      
      # Data: (u_indices, u_values, v_indices, v_values, w_indices, w_values)
      terms = [
          # 1
          ([0, 4, 8], [1, 1, 1], [0, 4, 8], [1, 1, 1], [0, 4, 8], [1, 1, 1]),
          # 2
          ([1, 3], [1, 1], [2, 5], [1, 1], [6, 7], [1, 1]),
          # 3
          ([2, 5], [1, 1], [5], [1], [7], [1]),
          # 4
          ([6, 7], [1, 1], [1, 6], [1, 1], [8], [1]),
          # 5
          ([0], [1], [6], [1], [8], [1]),
          # 6
          ([8], [1], [0], [1], [7], [1]),
          # 7
          ([1], [1], [8], [1], [8], [1]),
          # 8
          ([7], [1], [2], [1], [8], [1]),
          # 9
          ([0], [1], [4], [1], [6], [1]),
          # 10
          ([8], [1], [7], [1], [8], [1]),
          # 11
          ([2], [1], [8], [1], [6], [1]),
          # 12
          ([8], [1], [1], [1], [8], [1]),
          # 13
          ([4], [1], [6], [1], [7], [1]),
          # 14
          ([6], [1], [5], [1], [8], [1]),
          # 15
          ([5], [1], [7], [1], [6], [1]),
          # 16
          ([3], [1], [8], [1], [7], [1]),
          # 17
          ([8], [1], [2], [1], [8], [1]),
          # 18
          ([1], [1], [8], [1], [8], [1]),
          # 19
          ([7], [1], [6], [1], [8], [1]),
      ]
      
      for r in range(rank):
          ui, uv, vi, vv, wi, wv = terms[r]
          for idx, val in zip(ui, uv):
              U[idx][r] = val
          for idx, val in zip(vi, vv):
              V[idx][r] = val
          for idx, val in zip(wi, wv):
              W[idx][r] = val
              
      return {"U": U, "V": V, "W": W}
  # EVOLVE-BLOCK-END
  