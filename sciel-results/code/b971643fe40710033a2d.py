def construct():
    n, m, p = 2, 5, 5
    rank = 10
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for r_prime in range(m): # r' from 0 to 4
                r = i * m + r_prime
                U[i * m + j, r] = 1.0
                
                # V[j * p + k, r] = delta(j, r')
                # V depends on k in its index, but value is independent of k?
                # Yes, V[j*p+k, r] is the same for all k for a fixed j, r.
                for k in range(p):
                    V[j * p + k, r] = 1.0 if j == r_prime else 0.0
                    
                # W[k * n + i, r] = delta(k, r')
                for k in range(p):
                    W[k * n + i, r] = 1.0 if k == r_prime else 0.0
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
