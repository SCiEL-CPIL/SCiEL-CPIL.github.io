def construct():
    start_time = time.time()
    time_limit = 85.0
    
    best_points = None
    best_score = 0
    
    seeds = [
        [(x - 2, y - 2) for x in range(6) for y in range(5)],
        [(x - 2, y - 3) for x in range(5) for y in range(6)],
        [(x - 4, y - 1) for x in range(10) for y in range(3)],
        [(x - 1, y - 4) for x in range(3) for y in range(10)]
    ]
    
    for seed in seeds:
        if len(set(seed)) < 30: continue
        score = evaluate(seed)
        if score > best_score:
            best_score = score
            best_points = seed[:]

    if best_points is None:
        return [(0,0)] * 30

    current_points = best_points[:]
    current_set = set(current_points)
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 200 # Lower to allow more restarts
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, 29)
        old_point = current_points[idx]
        
        # Generate candidate
        if random.random() < 0.8:
            nx = old_point[0] + random.randint(-3, 3)
            ny = old_point[1] + random.randint(-3, 3)
        else:
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            
        if -30 <= nx <= 30 and -30 <= ny <= 30:
            candidate = (nx, ny)
            if candidate not in current_set:
                # Apply move
                current_points[idx] = candidate
                current_set.remove(old_point)
                current_set.add(candidate)
                
                new_score = evaluate(current_points)
                
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points[:]
                    no_improve_count = 0
                    continue
                else:
                    # Revert
                    current_points[idx] = old_point
                    current_set.remove(candidate)
                    current_set.add(old_point)
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        if no_improve_count > max_no_improve:
            current_points = best_points[:]
            current_set = set(current_points)
            current_score = best_score
            no_improve_count = 0
            # Perturb
            for _ in range(random.randint(1, 3)):
                pidx = random.randint(0, 29)
                px, py = current_points[pidx]
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
                if (new_x, new_y) not in current_set:
                    current_points[pidx] = (new_x, new_y)
                    current_set.remove((px, py))
                    current_set.add((new_x, new_y))

    return best_points
