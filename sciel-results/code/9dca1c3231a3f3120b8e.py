    # Refine center and beta for the best combo
    def obj_center_beta(params):
        cx, cy, b = params
        normals = []
        for k in range(6):
            ang = b + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_proj = 0.0
        for nx, ny in normals:
            for px, py in pts:
                proj = abs((px - cx) * nx + (py - cy) * ny)
                if proj > max_proj:
                    max_proj = proj
        D = max_proj + APOTHEM
        S = (2.0 / math.sqrt(3.0)) * D
        return S

    from scipy.optimize import minimize
    res2 = minimize(obj_center_beta, [0.0, 0.0, best_beta], method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    if res2.fun < best_S:
        best_S = res2.fun
        best_center = (res2.x[0], res2.x[1])
        best_beta = res2.x[2]
    else:
        best_center = (0.0, 0.0)
