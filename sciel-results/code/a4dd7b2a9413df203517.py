def angles(n, coupling, field, layers):
    start_time = time.time()
    # Use a safety margin of 20s from the 150s limit
    deadline = start_time + 130
    
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    best_params = np.zeros((layers, n))
    best_energy = float('inf')
    
    # Reduced set of initializations to save time
    inits = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 4),
        np.full((layers, n), np.pi / 2),
    ]
    
    # Add a couple of random inits if time permits
    if time.time() < deadline - 20:
        inits.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
        inits.append(np.random.uniform(-np.pi, np.pi, (layers, n)))

    # Tighter optimization bounds to prevent timeout
    for init in inits:
        if time.time() > deadline - 5:
            break
        
        x0 = init.flatten()
        
        try:
            # Use L-BFGS-B with reduced iterations
            result = opt.minimize(
                lambda x: _objective(x.reshape(layers, n), n, coupling, field, layers, ground_energy, deadline),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 100}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape(layers, n)
                
        except Exception:
            pass
            
    # Additional random restarts if time permits
    while time.time() < deadline - 10:
        x0 = np.random.uniform(-np.pi, np.pi, (layers, n)).flatten()
        try:
            result = opt.minimize(
                lambda x: _objective(x.reshape(layers, n), n, coupling, field, layers, ground_energy, deadline),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 30, 'maxfun': 50}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
            
    return best_params
