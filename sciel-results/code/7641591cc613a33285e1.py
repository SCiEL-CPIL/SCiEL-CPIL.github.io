s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(len(U[0])))
if abs(s - 1) > 1e-9:
