# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = 24
    U = np.zeros((n*m, rank))  # 8 x 24
    V = np.zeros((m*p, rank))  # 24 x 24
    W = np.zeros((p*n, rank))  # 12 x 24
    
    # U index: i*m + k
    # V index: k*p + j
    # W index: j*n + i
    
    for j in range(p):
        for k in range(m):
            t = j * m + k
            for i in range(n):
                U[i*m + k, t] = 1.0
                V[k*p + j, t] = 1.0
                W[j*n + i, t] = 1.0
                
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
