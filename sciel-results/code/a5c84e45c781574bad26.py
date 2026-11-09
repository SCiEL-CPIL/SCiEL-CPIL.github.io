    # Add 1 term for k=4 slice
    U[i*m+j][col] = 1 for all i, j
    V[j*p+4][col] = 1 for all j
    W[4*n+i][col] = 1 for all i
    col += 1
