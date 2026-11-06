def _nelder_mead(func, x0, max_iter=1000, tol=1e-6, deadline=None):
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1
        xi[i] += step
        simplex.append(xi)
    
    fvals = [func(v) for v in simplex]
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        idx = np.argsort(fvals)
        simplex = [simplex[i] for i in idx]
        fvals = [fvals[i] for i in idx]
        
        if np.abs(fvals[0] - fvals[-1]) < tol:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            if fr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < min(fr, fvals[-1]):
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
                
    idx_min = np.argmin(fvals)
    return simplex[idx_min], fvals[idx_min]
