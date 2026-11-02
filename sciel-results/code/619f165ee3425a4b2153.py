<<<<<<< SEARCH
def _nelder_mead_polish(inner_hexes, max_iter=200, tol=1e-6):
    """Polish inner hexagon positions using Nelder-Mead to minimize enclosing hexagon side."""
    from scipy.optimize import minimize
    
    def objective(params):
        # params: flattened [x1, y1, a1, x2, y2, a2, ...]
        new_inner = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            new_inner.append([x, y, a])
        
        # Check disjointness - if violated, return large penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = new_inner[i][0] - new_inner[j][0]
                dy = new_inner[i][1] - new_inner[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < math.sqrt(3) - 0.005:
                    return 1000.0 + (math.sqrt(3) - dist) * 100
        
        cx, cy, side, angle = _min_enclosing_hex_side(new_inner)
        return side
    
    # Initial params
    x0 = []
    for h in inner_hexes:
        x0.extend([h[0], h[1], h[2]])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
    
    if result.success and result.fun < 1000.0:
        polished = []
        for i in range(N):
            polished.append([result.x[i*3], result.x[i*3+1], result.x[i*3+2]])
        return polished
    
    return inner_hexes
=======
def _nelder_mead_polish(inner_hexes, max_iter=500, tol=1e-7):
    """Polish inner hexagon positions using Nelder-Mead to minimize enclosing hexagon side."""
    from scipy.optimize import minimize
    
    def objective(params):
        # params: flattened [x1, y1, a1, x2, y2, a2, ...]
        new_inner = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            new_inner.append([x, y, a])
        
        # Check disjointness - if violated, return large penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = new_inner[i][0] - new_inner[j][0]
                dy = new_inner[i][1] - new_inner[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < math.sqrt(3) - 0.001:
                    penalty += (math.sqrt(3) - dist) * 500.0
        if penalty > 0.01:
            return 1000.0 + penalty
        
        cx, cy, side, angle = _min_enclosing_hex_side(new_inner)
        return side
    
    # Initial params
    x0 = []
    for h in inner_hexes:
        x0.extend([h[0], h[1], h[2]])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
    
    if result.fun < 1000.0:
        polished = []
        for i in range(N):
            polished.append([result.x[i*3], result.x[i*3+1], result.x[i*3+2]])
        return polished
    
    return inner_hexes
>>>>>>> REPLACE

