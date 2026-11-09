  def check(U, V, W, n, m, p):
      for i in range(n):
          for j in range(m):
              for k in range(p):
                  a = i * m + j
                  b = j * p + k
                  c = k * n + i
                  val = np.dot(U[a], V[b] * W[c])
                  if abs(val - 1) > 1e-6:
                      return False
      return True
  