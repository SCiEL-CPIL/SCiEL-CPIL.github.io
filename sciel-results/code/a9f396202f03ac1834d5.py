<<<<<<< SEARCH
    # Run Differential Evolution with multiple restarts
    # We have ~580s. DE is stochastic.
    # Let's run a few independent DEs with different seeds and settings.
    
    seeds = [42, 123, 456, 789, 101, 202]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 120:  # Leave 120s for basin hopping and polish
            break
        
        try:
            # Use smaller popsize and maxiter for speed, relying on multiple restarts
            # popsize=15, maxiter=100 is usually fast for 12D.
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=200, 
                popsize=20, 
                tol=1e-10, 
                polish=True,  # L-BFGS-B polish
                init='sobol' # Better initial distribution
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass
=======
    # Run Differential Evolution with multiple restarts
    # We have ~580s. DE is stochastic.
    # Let's run a few independent DEs with different seeds and settings.
    
    seeds = [42, 123, 456, 789, 101, 202]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 120:  # Leave 120s for basin hopping and polish
            break
        
        try:
            # Use smaller popsize and maxiter for speed, relying on multiple restarts
            # popsize=15, maxiter=100 is usually fast for 12D.
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=300, 
                popsize=25, 
                tol=1e-12, 
                polish=True,  # L-BFGS-B polish
                init='sobol' # Better initial distribution
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass
>>>>>>> REPLACE

