U[i*m+j][column] = 1   # U has shape (n*m, rank), indexed by (i,j)
V[j*p+k][column] = 1   # V has shape (m*p, rank), indexed by (j,k)
W[k*n+i][column] = 1   # W has shape (p*n, rank), indexed by (k,i)
