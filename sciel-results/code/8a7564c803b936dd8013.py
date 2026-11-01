# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 5
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_strassen(k1, k2):
        nonlocal current_rank
        
        idx_B_0_k1 = 0 * p + k1
        idx_B_1_k1 = 1 * p + k1
        idx_B_0_k2 = 0 * p + k2
        idx_B_1_k2 = 1 * p + k2
        
        idx_C_0_k1 = k1 * n + 0
        idx_C_1_k1 = k1 * n + 1
        idx_C_0_k2 = k2 * n + 0
        idx_C_1_k2 = k2 * n + 1
        
        terms = [
            {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
            {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 1, 0, -1]},
            {'u': [1, 0, 0, 0], 'v': [0, 0, 1, -1], 'w': [0, 0, 1, 1]},
            {'u': [0, 0, 0, 1], 'v': [-1, 1, 0, 0], 'w': [1, 1, 0, 0]},
            {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 0, 1, 0]},
            {'u': [-1, 0, 1, 0], 'v': [1, 0, 1, 0], 'w': [0, 0, 0, 1]},
            {'u': [0, 1, 0, -1], 'v': [0, 1, 0, 1], 'w': [1, 0, 0, 0]}
        ]
        
        for t in terms:
            u = t['u']
            v = t['v']
            w = t['w']
            
            U[0][current_rank] += u[0]
            U[1][current_rank] += u[1]
            U[2][current_rank] += u[2]
            U[3][current_rank] += u[3]
            
            V[idx_B_0_k1][current_rank] += v[0]
            V[idx_B_1_k1][current_rank] += v[1]
            V[idx_B_0_k2][current_rank] += v[2]
            V[idx_B_1_k2][current_rank] += v[3]
            
            W[idx_C_0_k1][current_rank] += w[0]
            W[idx_C_1_k1][current_rank] += w[1]
            W[idx_C_0_k2][current_rank] += w[2]
            W[idx_C_1_k2][current_rank] += w[3]
            
            current_rank += 1

    def add_schoolbook(k):
        nonlocal current_rank
        
        idx_B_0_k = 0 * p + k
        idx_B_1_k = 1 * p + k
        idx_C_0_k = k * n + 0
        idx_C_1_k = k * n + 1
        
        terms = [
            {'u': [1, 0, 0, 0], 'v_idx': 0, 'w_idx': 0, 'val': 1},
            {'u': [0, 1, 0, 0], 'v_idx': 1, 'w_idx': 0, 'val': 1},
            {'u': [0, 0, 1, 0], 'v_idx': 0, 'w_idx': 1, 'val': 1},
            {'u': [0, 0, 0, 1], 'v_idx': 1, 'w_idx': 1, 'val': 1}
        ]
        
        for t in terms:
            u = t['u']
            U[0][current_rank] += u[0]
            U[1][current_rank] += u[1]
            U[2][current_rank] += u[2]
            U[3][current_rank] += u[3]
            
            if t['v_idx'] == 0:
                V[idx_B_0_k][current_rank] += t['val']
            else:
                V[idx_B_1_k][current_rank] += t['val']
                
            if t['w_idx'] == 0:
                W[idx_C_0_k][current_rank] += t['val']
            else:
                W[idx_C_1_k][current_rank] += t['val']
                
            current_rank += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_schoolbook(4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
