  def validate(U, V, W, rank):
      for i in range(n):
          for j in range(m):
              for k in range(p):
                  s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                  if abs(s - 1.0) > 1e-9:
                      return False
      return True
  