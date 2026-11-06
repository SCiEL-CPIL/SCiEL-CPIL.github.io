  for i in range(k + 1):
      left_part = L[i]
      for j in range(k, n):  # BUG: should be range(0, n - k + 1) for R indices?
          right_part = R[j - k]
  