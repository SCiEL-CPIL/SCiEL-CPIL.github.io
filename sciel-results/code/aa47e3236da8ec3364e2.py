s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
if abs(s - 1.0) > tol:
    return False
