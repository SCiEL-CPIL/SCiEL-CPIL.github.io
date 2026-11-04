a_form[i] = sum_j U[i*m+j][c] * A[i,j]
b_form[k] = sum_j V[j*p+k][c] * B[j,k]
C_comp[i,k] += W[k*n+i][c] * a_form[i] * b_form[k]
