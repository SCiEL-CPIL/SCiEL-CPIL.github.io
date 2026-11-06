def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Get initial design
    init_labels, init_thicknesses = _design_direct(problem)
    indices = [materials[label] for label in init_labels]
    n_layers = len(indices)
    
    def make_f(indices):
        def f(x):
            return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                              max_layer_nm, max_total_nm)
        return f
    
    best_x = None
    best_f = np.inf
    best_labels = init_labels
    
    # Try different layer configurations
    configs = []
    for n_layers_try in [5, 7, 9, 11, 13, 15, 17]:
        if n_layers_try > max_layers:
            continue
        labels_1 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers_try)]
        labels_2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers_try)]
        configs.append(labels_1)
        configs.append(labels_2)
        
    # Also include the initial best
    configs.append(init_labels)
    
    for labels in configs:
        if time.time() > deadline - 10:
            break
            
        indices_try = [materials[label] for label in labels]
        n_try = len(indices_try)
        bounds_try = [(0.0, float(max_layer_nm))] * n_try
        
        # Constraint: sum(thicknesses) <= max_total_nm
        constraints = [
            {'type': 'ineq', 'fun': lambda x: max_total_nm - np.sum(x)}
        ]
        
        f_try = make_f(indices_try)
        
        # Generate initial points
        initial_designs = _generate_initial_designs(indices_try, n_try, center_wl, 
                                                     max_total_nm, max_layer_nm, n_starts=10)
        
        for x0 in initial_designs:
            if time.time() > deadline - 5:
                break
            
            x0 = np.clip(x0, 0, max_layer_nm)
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0))
            
            # Run SLSQP
            try:
                res = scipy.optimize.minimize(f_try, x0, method='SLSQP', bounds=bounds_try, 
                                              constraints=constraints, 
                                              options={'maxiter': 50, 'ftol': 1e-8})
                
                # Check if result is valid and better
                if hasattr(res, 'x'):
                    x_opt = res.x
                    x_opt = np.clip(x_opt, 0, max_layer_nm)
                    if np.sum(x_opt) > max_total_nm:
                        x_opt = x_opt * (max_total_nm / np.sum(x_opt))
                    
                    f_opt = f_try(x_opt)
                    
                    if f_opt < best_f:
                        best_f = f_opt
                        best_x = x_opt.copy()
                        best_labels = labels.copy()
            except Exception:
                pass
        
        # If we found a very good solution, we can stop early
        if best_f < 0.5: # 0.5% reflectance is excellent
            break

    if best_x is None:
        best_x = np.array(init_thicknesses)
        best_labels = init_labels
    
    thicknesses = np.array(best_x)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
