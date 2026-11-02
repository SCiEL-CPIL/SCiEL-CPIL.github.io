      # T[i,j,k,l] = delta_{i,l} * delta_{j,k}
      # Decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
      # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
      # T[i,j,k,l] = sum_r U[2*i+j, r] * V[7*j+k, r] * W[2*k+l, r]
      # Expected: T[i,j,k,l] = 1 if l==i and k==j, else 0
      