def cons_func(z):
    cons = np.zeros(4*N + len(pairs))
    idx = 0
    # Walls
    for i in range(N):
        x, y, r = z[3*i], z[3*i+1], z[3*i+2]
        cons[idx] = x - r
        cons[idx+1] = 1 - x - r
        cons[idx+2] = y - r
        cons[idx+3] = 1 - y - r
        idx += 4
        
    # Pairs
    for i, j in pairs:
        dx = z[3*i] - z[3*j]
        dy = z[3*i+1] - z[3*j+1]
        dr = z[3*i+2] + z[3*j+2]
        cons[idx] = dx*dx + dy*dy - dr*dr
        idx += 1
        
    return cons
