<<<<<<< SEARCH
def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients using symmetric Nelder-Mead.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Fallback: Simple Strang splitting (symmetric)
    # a = [1/12]*12, b = [1/6, 0]*6
    # Symmetric version:
    # sym_a = [1/12]*6
    # sym_b = [1/6, 0, 1/6, 0, 1/6, 0]
    # Wait, sum(sym_b) = 0.5. 0.5/0.5 = 1. So norm_b = sym_b.
    # full_b = [1/6, 0, 1/6, 0, 1/6, 0, 0, 1/6, 0, 1/6, 0, 1/6]? No.
    # Symmetry: b[11-i] = b[i].
    # If sym_b = [1/6, 0, 1/6, 0, 1/6, 0], then full_b = [1/6, 0, 1/6, 0, 1/6, 0, 0, 1/6, 0, 1/6, 0, 1/6].
    # Sum = 1. Correct.
    
    fallback_sym_a = np.array([1.0/12.0] * 6)
    fallback_sym_b = np.array([1.0/6.0, 0.0, 1.0/6.0, 0.0, 1.0/6.0, 0.0])
    
    fallback_a, fallback_b = _symmetric_to_full(fallback_sym_a, fallback_sym_b)
    if fallback_a is None:
        fallback_a = [1.0/12.0] * 12
        fallback_b = [0.0] * 12
        for i in range(0, 12, 2):
            fallback_b[i] = 1.0/6.0
    
    best_params = (list(fallback_a), list(fallback_b))
    best_error = _compute_error(fallback_a, fallback_b)
    
    # Define objective function for Nelder-Mead
    def objective(params):
        sym_a = params[:6]
        sym_b = params[6:12]
        a, b = _symmetric_to_full(sym_a, sym_b)
        if a is None:
            return 1e10
        return _compute_error(a, b)
    
    # Initial guess for Nelder-Mead
    x0 = np.concatenate([fallback_sym_a, fallback_sym_b])
    
    # Bounds for symmetric parameters
    # Since norm = sym * (0.5/sum), and sum can vary, bounds on sym are tricky.
    # Just use wide bounds [-10, 10] for sym parameters.
    bounds = [(-10.0, 10.0)] * 12
    
    # Run Nelder-Mead
    try:
        best_x, best_val = _nelder_mead_sym(objective, x0, bounds, max_iter=800, time_limit=130.0)
        
        # Convert best symmetric params to full
        sym_a_best = best_x[:6]
        sym_b_best = best_x[6:12]
        a_best, b_best = _symmetric_to_full(sym_a_best, sym_b_best)
        
        if a_best is not None and best_val < best_error:
            best_error = best_val
            best_params = (list(a_best), list(b_best))
    except Exception as e:
        pass
    
    # Final validation
    if _is_valid(best_params[0], best_params[1]):
        return best_params[0], best_params[1]
    
    return fallback_a, fallback_b
=======
def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients using gradient descent
    on symmetric parameters, with a second-order-accurate initial guess.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Fallback: Simple Strang splitting (symmetric)
    fallback_sym_a = np.array([1.0/12.0] * 6)
    fallback_sym_b = np.array([1.0/6.0, 0.0, 1.0/6.0, 0.0, 1.0/6.0, 0.0])
    
    fallback_a, fallback_b = _symmetric_to_full(fallback_sym_a, fallback_sym_b)
    if fallback_a is None:
        fallback_a = [1.0/12.0] * 12
        fallback_b = [0.0] * 12
        for i in range(0, 12, 2):
            fallback_b[i] = 1.0/6.0
    
    best_params = (list(fallback_a), list(fallback_b))
    best_error = _compute_error(fallback_a, fallback_b)
    
    # Define objective function
    def objective(params):
        sym_a = params[:6]
        sym_b = params[6:12]
        a, b = _symmetric_to_full(sym_a, sym_b)
        if a is None:
            return 1e10
        return _compute_error(a, b)
    
    # Use a better initial guess based on 4th-order Suzuki coefficients
    # For a 2-stage 4th-order formula: a1=a2=1/2, b1=b2=1/2 but with specific ratios
    # For 6 symmetric pairs, use a smooth distribution
    # Try: sym_a = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] (sum=0.6, normalized to 0.5)
    # sym_b = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] (sum=0.6, normalized to 0.5)
    # This gives uniform distribution which is the fallback.
    
    # Alternative: use coefficients that minimize the second-order error
    # The second-order error involves sum_k a_k b_k weighted by position
    # For a symmetric formula, try to make the "center of mass" of a and b match
    
    # Start with the fallback and run gradient descent
    x0 = np.concatenate([fallback_sym_a, fallback_sym_b])
    bounds = [(-10.0, 10.0)] * 12
    
    # Run gradient descent
    try:
        best_x, best_val = _gradient_descent_sym(objective, x0, bounds, max_iter=300, time_limit=130.0, lr_init=0.03)
        
        sym_a_best = best_x[:6]
        sym_b_best = best_x[6:12]
        a_best, b_best = _symmetric_to_full(sym_a_best, sym_b_best)
        
        if a_best is not None and best_val < best_error:
            best_error = best_val
            best_params = (list(a_best), list(b_best))
    except Exception as e:
        pass
    
    # Final validation
    if _is_valid(best_params[0], best_params[1]):
        return best_params[0], best_params[1]
    
    return fallback_a, fallback_b
>>>>>>> REPLACE
