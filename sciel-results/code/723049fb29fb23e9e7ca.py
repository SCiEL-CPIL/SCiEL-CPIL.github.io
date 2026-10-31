<<<<<<< SEARCH
def construct():
    global DEADLINE
    t0 = time.time()
    DEADLINE = t0 + 138.0
    
    best = None
    best_val = -1.0
    
    # Aspect ratio candidates: W + H = 2
    # Try a range of aspects
    aspect_choices = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.15, 0.85]
    random.shuffle(aspect_choices)
    
    for W in aspect_choices:
        if time.time() > DEADLINE - 10:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        circles = _force_directed_pack(W, H, seed=int(W * 100))
        if circles is not None:
            # Scale to exact perimeter 4
            circles_scaled = _scale_to_perimeter(circles, PERIMETER)
            ok, per, s = _validity(circles_scaled, 2.0, 2.0)
            if ok and s > best_val:
                best_val = s
                best = circles_scaled
    
    if best is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = cells[:N]
        best = _scale_to_perimeter(best, PERIMETER)
    
    return np.array(best)
=======
def construct():
    global DEADLINE
    t0 = time.time()
    DEADLINE = t0 + 138.0
    
    best = None
    best_val = -1.0
    
    # Aspect ratio candidates: W + H = 2
    # Focus on more promising aspect ratios for 21 circles
    aspect_choices = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.25, 0.75, 1.3, 0.7, 1.4, 0.65]
    random.shuffle(aspect_choices)
    
    # Time budget per aspect ratio
    time_per_aspect = 12.0
    
    for W in aspect_choices:
        if time.time() > DEADLINE - 15:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        # Try multiple seeds for each aspect ratio
        for seed_offset in range(3):
            if time.time() > DEADLINE - 15:
                break
            
            seed = int(W * 100) + seed_offset * 1000
            time_limit = min(time_per_aspect, DEADLINE - time.time() - 10)
            if time_limit < 2.0:
                break
            
            circles = _simulated_annealing_grid(W, H, seed=seed, time_limit=time_limit)
            if circles is not None:
                # Scale to exact perimeter 4
                circles_scaled = _scale_to_perimeter(circles, PERIMETER)
                ok, per, s = _validity(circles_scaled, 2.0, 2.0)
                if ok and s > best_val:
                    best_val = s
                    best = circles_scaled
    
    if best is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = cells[:N]
        best = _scale_to_perimeter(best, PERIMETER)
    
    return np.array(best)
>>>>>>> REPLACE
