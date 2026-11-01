    # Block 1: k in {0, 1}
    # V local: 0=e, 1=f, 2=g, 3=h
    # V global: 0, 1, 4, 5
    v_map_1 = [0, 1, 4, 5]
    # W local: 0=p, 1=r, 2=q, 3=s
    # W global: 0, 1, 2, 3
    w_map_1 = [0, 1, 2, 3]
    
    for t_idx, (u_vec, v_vec, w_vec) in enumerate(terms):
        col = t_idx
        for i in range(4):
            U[i][col] = u_vec[i]
        for j in range(4):
            V[v_map_1[j]][col] = v_vec[j]
        for k in range(4):
            W[w_map_1[k]][col] = w_vec[k]
            
    # Block 2: k in {2, 3}
    # V local: 0=e', 1=f', 2=g', 3=h'
    # V global: 2, 3, 6, 7
    v_map_2 = [2, 3, 6, 7]
    # W local: 0=p', 1=r', 2=q', 3=s'
    # W global: 4, 5, 6, 7
    w_map_2 = [4, 5, 6, 7]
    
    for t_idx, (u_vec, v_vec, w_vec) in enumerate(terms):
        col = 7 + t_idx
        for i in range(4):
            U[i][col] = u_vec[i]
        for j in range(4):
            V[v_map_2[j]][col] = v_vec[j]
        for k in range(4):
            W[w_map_2[k]][col] = w_vec[k]
