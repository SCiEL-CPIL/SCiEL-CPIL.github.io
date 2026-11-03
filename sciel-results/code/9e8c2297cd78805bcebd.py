    def construct():
        import numpy as np
        n, m, p = 3, 3, 8
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        r = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, r] = 1.0
                    V[k * p + j, r] = 1.0
                    W[j * n + i, r] = 1.0
                    r += 1
        
        return {'U': U, 'V': V, 'W': W}
    