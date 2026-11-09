  for r in range(4):
      for i in range(n):
          for j in range(4):
              U[i * m + j, col] = 1
          W[4 * n + i, col] = 1
      for j in range(4):
          if j == r:
              V[j * p + 4, col] = 1
      col += 1
  