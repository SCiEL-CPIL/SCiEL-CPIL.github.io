    for i in range(4):
        for j in range(4):
            if time.time() > deadline:
                break
            # C[k,i] += A[i,j] * B[j,k] for k=0..3 (4x4 part)
            for k in range(4):
                if col >= max_rank:
                    break
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
            # C[4,i] += A[i,j] * B[j,4] (5th column)
            if col >= max_rank:
                break
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
