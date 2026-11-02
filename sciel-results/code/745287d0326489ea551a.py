def _objective(x_flat):
    """
    Objective function for differential_evolution.
    Minimizes negative minimum area.
    """
    points = x_flat.reshape(N, 2)
    points = _project_to_triangle(points)
    area = _min_area(points)
    return -area

def construct():
    """
    Place 11 points using Differential Evolution.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Define bounds: x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, S3 / 2.0))
        
    # Run Differential Evolution
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=150, 
            popsize=25, 
            tol=1e-8, 
            seed=42,
            polish=True,
            init='sobol'
        )
    except Exception:
        # Fallback to centroid if DE fails
        return np.full((N, 2), [0.5, S3/6])

    best_points_flat = res.x
    best_points = best_points_flat.reshape(N, 2)
    best_points = _project_to_triangle(best_points)
    
    return best_points
