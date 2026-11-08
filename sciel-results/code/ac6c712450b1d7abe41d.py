<<<<<<< SEARCH
def objective_flat(x):
    p = x.reshape(-1, 2)
    # Project
    p = np.array([project_to_triangle(pi) for pi in p])
    return -min_triangle_area(p)

def local_optimize_nm(points, deadline, max_iter=500):
    """Nelder-Mead optimization."""
    x0 = points.flatten()
    # Options for Nelder-Mead
    options = {'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
    
    # We need to wrap the objective to check deadline?
    # scipy doesn't support callbacks easily for stopping.
    # We can just run it and hope it finishes, or check time after.
    # To be safe, we can run it in a way that we can interrupt, but standard minimize is blocking.
    # Given 120s limit, and NM is fast for 22 vars, it should be fine.
    
    res = minimize(objective_flat, x0, method='Nelder-Mead', options=options)
    
    if res.success or res.fun < objective_flat(x0):
        best_points = res.x.reshape(-1, 2)
        best_points = np.array([project_to_triangle(p) for p in best_points])
        return best_points
    else:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_points = None
    best_value = -1
    
    # 1. Structured Initial
    init_structured = generate_structured_initial()
    # Surrogate warmup
    warmup_1 = surrogate_optimize_min_distance(init_structured, deadline, max_iter=200)
    # NM Polish
    opt_1 = local_optimize_nm(warmup_1, deadline, max_iter=1000)
    val_1 = min_triangle_area(opt_1)
    if val_1 > best_value:
        best_value = val_1
        best_points = opt_1
    
    # 2. Greedy Max-Min Initial
    if time.time() < deadline - 20:
        init_greedy = greedy_max_min(seed=42)
        warmup_2 = surrogate_optimize_min_distance(init_greedy, deadline, max_iter=200)
        opt_2 = local_optimize_nm(warmup_2, deadline, max_iter=1000)
        val_2 = min_triangle_area(opt_2)
        if val_2 > best_value:
            best_value = val_2
            best_points = opt_2
            
    # 3. Random Initials with Surrogate
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        init_rand = generate_random_points(seed * 100 + 1)
        warmup_r = surrogate_optimize_min_distance(init_rand, deadline, max_iter=200)
        opt_r = local_optimize_nm(warmup_r, deadline, max_iter=1000)
        val_r = min_triangle_area(opt_r)
        if val_r > best_value:
            best_value = val_r
            best_points = opt_r
            
    # 4. Final Polish on Best
    if best_points is not None and time.time() < deadline - 10:
        # Perturb slightly
        perturbed = best_points + np.random.randn(*best_points.shape) * 0.01
        perturbed = np.array([project_to_triangle(p) for p in perturbed])
        opt_final = local_optimize_nm(perturbed, deadline, max_iter=1000)
        val_final = min_triangle_area(opt_final)
        if val_final > best_value:
            best_value = val_final
            best_points = opt_final
            
    if best_points is None:
        best_points = init_structured
        
    # Final verification
    for i in range(N):
        if not is_inside_triangle(best_points[i]):
            best_points[i] = project_to_triangle(best_points[i])
            
    return best_points
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    best_points = None
    best_value = -1
    
    # 1. Structured Initial
    init_structured = generate_structured_initial()
    pts, val = simulated_annealing(init_structured, deadline)
    if val > best_value:
        best_value = val
        best_points = pts
    
    # 2. Greedy Max-Min Initial
    if time.time() < deadline - 10:
        init_greedy = greedy_max_min(seed=42)
        pts, val = simulated_annealing(init_greedy, deadline)
        if val > best_value:
            best_value = val
            best_points = pts
            
    # 3. Random Initials
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        init_rand = generate_random_points(seed * 100 + 1)
        pts, val = simulated_annealing(init_rand, deadline)
        if val > best_value:
            best_value = val
            best_points = pts
            
    if best_points is None:
        best_points = init_structured
        
    # Final verification
    for i in range(N):
        if not is_inside_triangle(best_points[i]):
            best_points[i] = project_to_triangle(best_points[i])
            
    return best_points
>>>>>>> REPLACE
