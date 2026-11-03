<<<<<<< SEARCH
    # 3. Optimization: Vectorized Soft-Min Gradient Ascent
    # We use a soft-min of the minimum distance.
    # Objective: Maximize min(d_ij, b_i)
    # Soft Min: -1/k * log(sum(exp(-k * d_ij)))
    # Gradient of soft min w.r.t. positions gives us the direction to move.
    
    # Hyperparameters
    alpha = 0.05
    cooling = 0.999
    k_soft = 50.0  # Start with moderate sharpness
    k_polish = 200.0 # Sharper for final phase
    steps_per_k = 1000 # Increase sharpness every 1000 steps
    
    step_count = 0
    
    while time.time() - start_time < time_limit:
=======
    # 3. Optimization: Vectorized Soft-Min Gradient Ascent
    # We use a soft-min of the minimum distance.
    # Objective: Maximize min(d_ij, b_i)
    # Soft Min: -1/k * log(sum(exp(-k * d_ij)))
    # Gradient of soft min w.r.t. positions gives us the direction to move.
    
    # Hyperparameters
    alpha = 0.03
    cooling = 0.9995
    k_soft = 30.0  # Start with moderate sharpness
    k_polish = 500.0 # Sharper for final phase
    steps_per_k = 500 # Increase sharpness every 500 steps
    
    step_count = 0
    
    # Track best score more frequently
    eval_interval = 50
    
    while time.time() - start_time < time_limit:
>>>>>>> REPLACE

