    # Update W
    for k in range(p):
        for i in range(n):
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # For fixed k,i: sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
            # A[j,c] = U[i*m+j,c] * V[j*p+k,c]
            A = U.reshape(n, m, -1)[i, :, :] * V.reshape(m, p, -1)[:, k, :].T  # (m, rank)
    