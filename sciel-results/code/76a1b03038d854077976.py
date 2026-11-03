def construct():
    import numpy as np

    n, m, p = 4, 4, 5
    
    strassen_u = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [1, 0, -1, 0],
        [0, 1, 0, -1] 
    ]
    strassen_v = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [0, 1, -1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    strassen_w = [
        [1, 0, 0, 1], 
        [0, 0, 1, 0], 
        [0, 1, 0, 0], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    alpha = [
        [[1, 0], [0, 1]], 
        [[0, 0], [1, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]  
    ]
    beta = [
        [[1, 0], [0, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 1], [0, -1]],
        [[0, 1], [-1, 0]],
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[0, 0], [1, 1]]  
    ]

    U_factors = []
    V_factors = []
    W_factors = []

    for k in range(7):
        for s in range(7):
            u_s = strassen_u[s]
            v_s = strassen_v[s]
            w_s = strassen_w[s]
            
            U_vec = [0.0] * 16
            V_vec = [0.0] * 20
            W_vec = [0.0] * 20
            
            for p_ in range(2):
                for q_ in range(2):
                    coeff = alpha[k][p_][q_]
                    if coeff != 0:
                        for i in range(2):
                            for j in range(2):
                                idx = (p_ * 2 + i) * 4 + (q_ * 2 + j)
                                u_idx = i * 2 + j
                                U_vec[idx] += coeff * u_s[u_idx]
                                
            for r_ in range(2):
                for s_ in range(2):
                    coeff = beta[k][r_][s_]
                    if coeff != 0:
                        for i in range(2):
                            for j in range(2):
                                idx = (r_ * 2 + i) * 4 + (s_ * 2 + j)
                                v_idx = i * 2 + j
                                V_vec[idx] += coeff * v_s[v_idx]
                                
            for t_ in range(2):
                for u_ in range(2):
                    w_idx = t_ * 2 + u_
                    val = w_s[w_idx]
                    if val != 0:
                        for i in range(2):
                            for j in range(2):
                                idx = (t_ * 2 + i) * 4 + (u_ * 2 + j)
                                W_vec[idx] += val
                                
            U_factors.append(U_vec)
            V_factors.append(V_vec)
            W_factors.append(W_vec)

    for i in range(4):
        for j in range(4):
            U_vec = [0.0] * 16
            V_vec = [0.0] * 20
            W_vec = [0.0] * 20
            
            U_vec[i * 4 + j] = 1.0
            V_vec[j * 5 + 4] = 1.0
            W_vec[4 * 4 + i] = 1.0
            
            U_factors.append(U_vec)
            V_factors.append(V_vec)
            W_factors.append(W_vec)
            
    U = np.array(U_factors).T
    V = np.array(V_factors).T
    W = np.array(W_factors).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
