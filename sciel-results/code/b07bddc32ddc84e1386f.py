tensor_val = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
if abs(tensor_val - 1.0) > 1e-10: return False
