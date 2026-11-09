import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Partition indices
    # 0: 0-1 (size 2)
    # 1: 2-3 (size 2)
    # 2: 4   (size 1)
    offsets = [0, 2, 4]
    sizes = [2, 2, 1]
    
    rank_total = 117
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    current_rank = 0
    
    def add_term(i_local, j_local, k_local, u_val, v_val, w_val, i_off, j_off, k_off, n_i, m_j, p_k):
        nonlocal current_rank
        r = current_rank
        current_rank += 1
        
        # Map local to global
        # U index: (i_global * m + j_global)
        i_g = i_off + i_local
        j_g = j_off + j_local
        idx_u = i_g * m + j_g
        U[idx_u][r] = u_val
        
        # V index: (j_global * p + k_global)
        idx_v = j_g * p + (k_off + k_local)
        V[idx_v][r] = v_val
        
        # W index: (k_global * n + i_global)
        k_g = k_off + k_local
        idx_w = k_g * n + i_g
        W[idx_w][r] = w_val
        
    # Loop over blocks i, j, k
    for i_idx in range(3):
        for j_idx in range(3):
            for k_idx in range(3):
                n_i = sizes[i_idx]
                m_j = sizes[j_idx]
                p_k = sizes[k_idx]
                
                i_off = offsets[i_idx]
                j_off = offsets[j_idx]
                k_off = offsets[k_idx]
                
                # Determine rank and terms
                if n_i == 2 and m_j == 2 and p_k == 2:
                    # Strassen
                    # m1
                    add_term(0, 0, 0, 1, 1, 0, i_off, j_off, k_off, n_i, m_j, p_k) # a00 b00 -> c00? No, Strassen U/V are sums.
                    # We need to add the Strassen terms specifically.
                    # It's easier to just hardcode the 7 terms for 2x2x2.
                    pass
                else:
                    # Use schoolbook / small rank
                    pass
