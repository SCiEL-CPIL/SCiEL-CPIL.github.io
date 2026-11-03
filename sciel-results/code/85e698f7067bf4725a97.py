        # Part A: j=0
        # We can do this in rank p=10
        for k in range(p):
            # U[i, 0] = 1 for i=0,1
            U[0*m + 0, col] = 1
            U[1*m + 0, col] = 1
            # V[0, k] = 1 (index 0*p + k)
            V[0*p + k, col] = 1
            # W[k, i] = 1 for i=0,1 (index k*n + i)
            W[k*n + 0, col] = 1
            W[k*n + 1, col] = 1
            col += 1
