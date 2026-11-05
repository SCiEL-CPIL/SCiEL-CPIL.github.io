total += W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
err = abs(total - 1.0)
