def has_path_4(u, w, c, adj, forbidden):
    # forbidden is the vertex v
    nu = adj[c][u]
    if nu & (1 << forbidden):
        nu &= ~(1 << forbidden)
    if nu == 0: return False
    
    x_mask = nu
    while x_mask:
        x_bit = x_mask & -x_mask
        x = x_bit.bit_length() - 1
        x_mask ^= x_bit
        
        nx = adj[c][x]
        if nx & (1 << forbidden):
            nx &= ~(1 << forbidden)
        if nx == 0: continue
        
        y_mask = nx
        while y_mask:
            y_bit = y_mask & -y_mask
            y = y_bit.bit_length() - 1
            y_mask ^= y_bit
            
            ny = adj[c][y]
            if ny & (1 << forbidden):
                ny &= ~(1 << forbidden)
            if ny == 0: continue
            
            z_mask = ny
            while z_mask:
                z_bit = z_mask & -z_mask
                z = z_bit.bit_length() - 1
                z_mask ^= z_bit
                
                if (adj[c][z] >> w) & 1:
                    return True
    return False
