    def construct():
        n, m, p = 3, 3, 7
        R = n * p # 21? No, R = m * p = 3 * 7 = 21.
        # Wait, R = sum over k of Rank(M_{3,3,1}) = 7 * 3 = 21.
        
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        
        for k in range(p):
            for j in range(m):
                r = k * m + j
                # U[i, j, r] = 1 for all i
                for i in range(n):
                    U[i*m + j][r] = 1
                # V[j, k, r] = 1
                V[j*p + k][r] = 1
                # W[k, i, r] = 1 for all i
                for i in range(n):
                    W[k*n + i][r] = 1
                    
        return {"U": U, "V": V, "W": W}
    