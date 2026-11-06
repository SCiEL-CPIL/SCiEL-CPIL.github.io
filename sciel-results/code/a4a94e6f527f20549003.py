def construct():
    # ... init ...
    
    def obj(x):
        t = np.clip(x[:N], 1e-6, np.pi - 1e-6)
        p = np.mod(x[N:], 2*np.pi)
        pts = _sph_to_cart(t, p)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        return _spectral_energy(pts, max_l=25)
        
    bounds = [(0.0, np.pi)] * N + [(0.0, 2*np.pi)] * N
    
    # Global
    de_res = differential_evolution(obj, bounds, maxiter=30, popsize=10, seed=42, polish=False)
    best_x = de_res.x
    
    # Local
    res = minimize(obj, best_x, method='Nelder-Mead', options={'maxiter': 5000, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-10})
    
    # ... extract ...
