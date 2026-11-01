<<<<<<< SEARCH
def _optimize_points(points, time_budget, beta=50.0, method='gradient'):
    """
    Optimize points using gradient descent on soft-max covering radius.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.05
    min_step = 1e-8
    decay = 0.92
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _compute_soft_max_radius_and_grads(current_points, beta=beta)
        
        # Also track the true radius
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-12:
            best_radius = true_radius
            best_points = current_points.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-14:
            break
        
        # Normalize gradient
        direction = grads / grad_norm
        
        # Try a step
        new_points = current_points + step_size * direction
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_true_radius = _compute_covering_radius(new_points)
        
        if new_true_radius < true_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.2)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius
=======
def _optimize_points(points, time_budget, beta=50.0, method='gradient'):
    """
    Optimize points using gradient descent on soft-max covering radius.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.05
    min_step = 1e-8
    decay = 0.92
    growth = 1.05
    stall_count = 0
    max_stall = 20
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _compute_soft_max_radius_and_grads(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-12:
            best_radius = true_radius
            best_points = current_points.copy()
            stall_count = 0
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-14:
            break
        
        direction = grads / grad_norm
        
        new_points = current_points + step_size * direction
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_true_radius = _compute_covering_radius(new_points)
        
        if new_true_radius < true_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.2)
        else:
            step_size *= decay
            stall_count += 1
            if step_size < min_step or stall_count > max_stall:
                break
    
    return best_points, best_radius
>>>>>>> REPLACE

