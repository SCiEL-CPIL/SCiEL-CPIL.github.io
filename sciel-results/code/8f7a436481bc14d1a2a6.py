import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    # X_forms[s][i*2+jp] = coefficient for A[i, b*2+jp] in Strassen step s
    # Y_forms[s][jp*2+kp] = coefficient for B[b*2+jp, 2*g+kp] in Strassen step s  
    # Z_combos[i*2+kp][s] = coefficient for C[2*g+kp, i] from Strassen step s
    
    X_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    
    Y_forms = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    num_groups = 4    # k-blocks of size 2
    num_jblocks = 2   # j-blocks of size 2
    num_strassen = 7
    rank = num_groups * num_jblocks * num_strassen  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    r = 0
    for g in range(num_groups):
        for b in range(num_jblocks):
            for s in range(num_strassen):
                # U: A[i, b*2+jp] -> X_forms[s][i*2+jp]
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx][r] = X_forms[s][i * 2 + jp]
                
                # V: B[b*2+jp, 2*g+kp] -> Y_forms[s][jp*2+kp]
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]
                
                # W: C[2*g+kp, i] -> Z_combos[i*2+kp][s]
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]
                
                r += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == '__main__':
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 2, 4, 8
    assert len(U) == n*m
    assert len(U[0]) == len(V[0])
    assert len(V) == m*p
    assert len(W) == p*n
    assert len(W[0]) == len(U[0])
    print("Rank:", len(U[0]))
