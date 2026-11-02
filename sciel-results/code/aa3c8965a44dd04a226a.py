def min_side_length_fast(centers, angles, refine=True):
    # ... setup X_flat, Y_flat ...
    
    def calc_S(alpha):
        max_proj = 0
        for k in range(6):
            theta = alpha + k * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            projs = X_flat * nx + Y_flat * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        return max_proj * 2.0 / math.sqrt(3)
        
    best_S = float('inf')
    best_alpha = 0
    
    if refine:
        alphas = np.linspace(0, math.pi/3, 100, endpoint=False)
    else:
        alphas = np.linspace(0, math.pi/3, 10, endpoint=False)
        
    for a in alphas:
        s = calc_S(a)
        if s < best_S:
            best_S = s
            best_alpha = a
            
    if refine:
        alphas_fine = np.linspace(best_alpha - math.pi/60, best_alpha + math.pi/60, 100)
        for a in alphas_fine:
            a_mod = a % (math.pi/3)
            s = calc_S(a_mod)
            if s < best_S:
                best_S = s
                best_alpha = a_mod
                
    return best_S, best_alpha
