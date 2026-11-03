lhs = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+ip, c]
rhs = 1.0 if (i == ip and j == k) else 0.0
