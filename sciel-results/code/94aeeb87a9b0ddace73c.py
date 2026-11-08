import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    strassen_terms = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(1, 0, 1), (0, 0, 1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)])
    ]
    
    def get_strassen_decomposition(size):
        if size == 1:
            return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
        
        half = size // 2
        U_rec, V_rec, W_rec = get_strassen_decomposition(half)
        rank_rec = U_rec.shape[1]
        
        rank_total = 0
        for term in strassen_terms:
            a_blocks, b_blocks, c_blocks = term
            num_pairs = 0
            for ib, jb, c_a in a_blocks:
                for jb2, kb, c_b in b_blocks:
                    if jb == jb2:
                        num_pairs += 1
            rank_total += num_pairs * rank_rec
            
        U = np.zeros((size * size, rank_total))
        V = np.zeros((size * size, rank_total))
        W = np.zeros((size * size, rank_total))
        
        col_offset = 0
        for term in strassen_terms:
            a_blocks, b_blocks, c_blocks = term
            for ib, jb, c_a in a_blocks:
                for jb2, kb, c_b in b_blocks:
                    if jb != jb2:
                        continue
                    for r_rec in range(rank_rec):
                        col = col_offset + r_rec
                        for i_l in range(half):
                            for j_l in range(half):
                                I = 2 * ib + i_l
                                J = 2 * jb + j_l
                                U[I * size + J, col] += c_a * U_rec[i_l * half + j_l, r_rec]
                                
                        for j_l in range(half):
                            for k_l in range(half):
                                J = 2 * jb + j_l
                                K = 2 * kb + k_l
                                V[J * size + K, col] += c_b * V_rec[j_l * half + k_l, r_rec]
                                
                        for kb2, ib2, c_c in c_blocks:
                            for k_l in range(half):
                                for i_l in range(half):
                                    K = 2 * kb2 + k_l
                                    I = 2 * ib2 + i_l
                                    W[K * size + I, col] += c_c * W_rec[k_l * half + i_l, r_rec]
                    
                    col_offset += rank_rec
                    
        return U, V, W

    U_444, V_444, W_444 = get_strassen_decomposition(4)
    rank_444 = U_444.shape[1]
    
    U = np.zeros((n * m, 2 * rank_444))
    V = np.zeros((m * p, 2 * rank_444))
    W = np.zeros((p * n, 2 * rank_444))
    
    U[:, :rank_444] = U_444
    U[:, rank_444:] = U_444
    
    for r in range(rank_444):
        for j in range(m):
            for k in range(4):
                V[j * p + k, r] = V_444[j * 4 + k, r]
                V[j * p + k + 4, r + rank_444] = V_444[j * 4 + k, r]
                
        for k in range(4):
            for i in range(n):
                W[k * n + i, r] = W_444[k * 4 + i, r]
                W[(k + 4) * n + i, r + rank_444] = W_444[k * 4 + i, r]
                
    return {"U": U, "V": V, "W": W}
