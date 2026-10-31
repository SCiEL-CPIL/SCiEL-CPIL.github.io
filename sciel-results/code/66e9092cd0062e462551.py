DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen factors
    # U2[i, j, r2]
    U2 = {}
    U2[(0, 0, 0)] = 1
    U2[(1, 1, 0)] = 1
    U2[(1, 0, 1)] = 1
    U2[(1, 1, 1)] = 1
    U2[(0, 0, 2)] = 1
    U2[(1, 1, 3)] = 1
    U2[(0, 0, 4)] = 1
    U2[(0, 1, 4)] = 1
    U2[(1, 0, 5)] = 1
    U2[(0, 0, 5)] = -1
    U2[(0, 1, 6)] = 1
    U2[(1, 1, 6)] = -1
    
    # V2[j, k, r2]
    V2 = {}
    V2[(0, 0, 0)] = 1
    V2[(1, 1, 0)] = 1
    V2[(0, 0, 1)] = 1
    V2[(0, 1, 2)] = 1
    V2[(1, 1, 2)] = -1
    V2[(1, 0, 3)] = 1
    V2[(0, 0, 3)] = -1
    V2[(1, 1, 4)] = 1
    V2[(0, 0, 5)] = 1
    V2[(0, 1, 5)] = 1
    V2[(1, 0, 6)] = 1
    V2[(1, 1, 6)] = 1
    
    # W2[k, i, r2]
    W2 = {}
    W2[(0, 0, 0)] = 1
    W2[(1, 1, 0)] = 1
    W2[(0, 1, 1)] = 1
    W2[(1, 1, 1)] = -1
    W2[(1, 0, 2)] = 1
    W2[(1, 1, 2)] = 1
    W2[(0, 0, 3)] = 1
    W2[(0, 1, 3)] = 1
    W2[(0, 0, 4)] = -1
    W2[(1, 0, 4)] = 1
    W2[(1, 1, 5)] = 1
    W2[(0, 0, 6)] = 1
    
    # Outer Strassen coefficients
    # alpha[r1, I1, J1]
    alpha = {}
    alpha[(0, 0, 0)] = 1
    alpha[(0, 1, 1)] = 1
    alpha[(1, 1, 0)] = 1
    alpha[(1, 1, 1)] = 1
    alpha[(2, 0, 0)] = 1
    alpha[(3, 1, 1)] = 1
    alpha[(4, 0, 0)] = 1
    alpha[(4, 0, 1)] = 1
    alpha[(5, 1, 0)] = 1
    alpha[(5, 0, 0)] = -1
    alpha[(6, 0, 1)] = 1
    alpha[(6, 1, 1)] = -1
    
    # beta[r1, J1, K1]
    beta = {}
    beta[(0, 0, 0)] = 1
    beta[(0, 1, 1)] = 1
    beta[(1, 0, 0)] = 1
    beta[(2, 0, 1)] = 1
    beta[(2, 1, 1)] = -1
    beta[(3, 1, 0)] = 1
    beta[(3, 0, 0)] = -1
    beta[(4, 1, 1)] = 1
    beta[(5, 0, 0)] = 1
    beta[(5, 0, 1)] = 1
    beta[(6, 1, 0)] = 1
    beta[(6, 1, 1)] = 1
    
    # gamma[r1, K1, I1]
    gamma = {}
    gamma[(0, 0, 0)] = 1
    gamma[(0, 1, 1)] = 1
    gamma[(1, 0, 1)] = 1
    gamma[(1, 1, 1)] = -1
    gamma[(2, 1, 0)] = 1
    gamma[(2, 1, 1)] = 1
    gamma[(3, 0, 0)] = 1
    gamma[(3, 0, 1)] = 1
    gamma[(4, 0, 0)] = -1
    gamma[(4, 1, 0)] = 1
    gamma[(5, 1, 1)] = 1
    gamma[(6, 0, 0)] = 1
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for I1 in range(2):
                for J1 in range(2):
                    a = alpha.get((r1, I1, J1), 0)
                    if a == 0:
                        continue
                    for I0 in range(2):
                        for J0 in range(2):
                            u2 = U2.get((I0, J0, r2), 0)
                            if u2 == 0:
                                continue
                            I = I1 * 2 + I0
                            J = J1 * 2 + J0
                            U[I * 4 + J][r] = a * u2
                            
            for J1 in range(2):
                for K1 in range(2):
                    b = beta.get((r1, J1, K1), 0)
                    if b == 0:
                        continue
                    for J0 in range(2):
                        for K0 in range(2):
                            v2 = V2.get((J0, K0, r2), 0)
                            if v2 == 0:
                                continue
                            J = J1 * 2 + J0
                            K = K1 * 2 + K0
                            V[J * 4 + K][r] = b * v2
                            
            for K1 in range(2):
                for I1 in range(2):
                    g = gamma.get((r1, K1, I1), 0)
                    if g == 0:
                        continue
                    for K0 in range(2):
                        for I0 in range(2):
                            w2 = W2.get((K0, I0, r2), 0)
                            if w2 == 0:
                                continue
                            K = K1 * 2 + K0
                            I = I1 * 2 + I0
                            W[K * 4 + I][r] = g * w2
                            
    return {"U": U, "V": V, "W": W}
