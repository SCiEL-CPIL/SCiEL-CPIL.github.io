def hexagon_packing_12():
    start_time = time.time()
    
    angles = np.pi/6 + np.arange(6) * np.pi/3
    n_in = np.array([[np.cos(a), np.sin(a)] for a in angles])
    max_proj = np.sqrt(3) / 2.0
    
    # 6x2 grid initial guess
    centers = []
    for i in range(6):
        for j in range(2):
            x = i * 1.5
            y = j * np.sqrt(3) + (i % 2) * np.sqrt(3)/2
            centers.append([x, y])
    centers = np.array(centers)
    centers -= centers.mean(axis=0)
    
    s0 = 4.0
    x0 = np.concatenate([centers.flatten(), [s0]])
    
    def objective(x):
        if time.time() - start_time > 580:
            return 1e10
        c = x[:24].reshape(12, 2)
        s = x[24]
        
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                diff = c[i] - c[j]
                min_dist = np.inf
                for k in range(6):
                    ang = np.pi/6 + k * np.pi/3
                    n = np.array([np.cos(ang), np.sin(ang)])
                    dist = abs(np.dot(diff, n))
                    if dist < min_dist:
                        min_dist = dist
                
                # Need gap > 1e-6, so dist > sqrt(3) + 1e-6
                if min_dist < np.sqrt(3) + 1e-5:
                    penalty += (np.sqrt(3) + 1e-5 - min_dist)**2
        
        viol = 0.0
        for i in range(12):
            for k in range(6):
                lhs = np.dot(c[i], n_in[k])
                rhs = max_proj * (s - 1) - 1e-5
                if lhs > rhs:
                    viol += (lhs - rhs)**2
        
        return s + 1000.0 * penalty + 1000.0 * viol

    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    x_best = res.x
    c_best = x_best[:24].reshape(12, 2)
    s_best = x_best[24]
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = c_best[i, 0]
        inner_hex_data[i, 1] = c_best[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_best
