def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Initial guess
    pts = _generate_hex_seed()
    x0 = pts.flatten()
    
    best_x = x0.copy()
    best_score = _score_hard(best_x)
    
    # Annealing schedule for beta
    betas = [10, 20, 50, 100, 200, 500, 1000]
    
    bounds = [(-HALF, HALF)] * (N * 2)
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Callback to check time and update best
        def callback(xk):
            nonlocal best_x, best_score
            if time.time() > deadline:
                return True # Stop optimization
            
            # Check hard score of current point
            c = xk.reshape(N, 2)
            if np.any(np.abs(c) > HALF + 0.5):
                return False
            
            s = _score_hard(c)
            if s > best_score:
                best_score = s
                best_x = xk.copy()
            return False
        
        # Run L-BFGS-B
        try:
            res = minimize(
                _objective_and_grad, 
                x0, 
                args=(beta,),
                method='L-BFGS-B', 
                jac=True,
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxls': 50
                },
                callback=callback
            )
            
            # Check if this result is better
            x_res = res.x
            c_res = x_res.reshape(N, 2)
            s_res = _score_hard(c_res)
            
            if s_res > best_score:
                best_score = s_res
                best_x = x_res.copy()
            
            # Use the result as the starting point for the next beta
            x0 = x_res.copy()
            
        except Exception as e:
            print(f"Optimization error: {e}")
            
    # Final check
    final_pts = best_x.reshape(N, 2)
    
    return final_pts
