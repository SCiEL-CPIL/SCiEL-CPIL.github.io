<<<<<<< SEARCH
    # Local search: perturb the best configuration found so far
    if best_points is not None and elapsed() < DEADLINE - 5.0:
        random.seed(42)
        current = best_points[:]
        current_score = best_score
        no_improve = 0
        max_no_improve = 100
        current_set = set(current)
        
        while elapsed() < DEADLINE - 2.0 and no_improve < max_no_improve:
            # Pick a random point to move
            idx = random.randint(0, 29)
            ox, oy = current[idx]
            
            # Try small perturbations
            improved = False
            for dx in [-3, -2, -1, 0, 1, 2, 3]:
                for dy in [-3, -2, -1, 0, 1, 2, 3]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    # Check if point already exists
                    if (nx, ny) in current_set:
                        continue
                    
                    candidate = current[:]
                    candidate[idx] = (nx, ny)
                    
                    if len(set(candidate)) != 30:
                        continue
                    
                    s = score_of(candidate)
                    if s > current_score:
                        current = candidate
                        current_score = s
                        current_set = set(current)
                        no_improve = 0
                        improved = True
                        break
                    elif s == current_score:
                        # Accept equal score moves to explore
                        if random.random() < 0.4:
                            current = candidate
                            current_set = set(current)
                            no_improve += 1
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
        
        # Update best if local search improved
        if current_score > best_score:
            best_score = current_score
            best_points = current[:]
=======
    # Local search: perturb the best configuration found so far
    if best_points is not None and elapsed() < DEADLINE - 5.0:
        random.seed(42)
        current = best_points[:]
        current_score = best_score
        no_improve = 0
        max_no_improve = 200
        current_set = set(current)
        
        # Multi-start: try several different starting points
        start_configs = [current[:]]
        # Add some variations
        for variant_seed in [43, 44, 45]:
            if elapsed() > DEADLINE - 5.0:
                break
            random.seed(variant_seed)
            variant = current[:]
            for _ in range(5):
                idx = random.randint(0, 29)
                ox, oy = variant[idx]
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = ox + dx, oy + dy
                        if -30 <= nx <= 30 and -30 <= ny <= 30 and (nx, ny) not in set(variant):
                            variant[idx] = (nx, ny)
                            break
                    else:
                        continue
                    break
            if len(set(variant)) == 30:
                start_configs.append(variant[:])
        
        for start_cfg in start_configs:
            if elapsed() > DEADLINE - 2.0:
                break
            current = start_cfg[:]
            current_score = score_of(current)
            no_improve = 0
            current_set = set(current)
            
            while elapsed() < DEADLINE - 2.0 and no_improve < max_no_improve:
                # Pick a random point to move
                idx = random.randint(0, 29)
                ox, oy = current[idx]
                
                # Try small perturbations
                improved = False
                for dx in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
                    for dy in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = ox + dx, oy + dy
                        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                            continue
                        # Check if point already exists
                        if (nx, ny) in current_set:
                            continue
                        
                        candidate = current[:]
                        candidate[idx] = (nx, ny)
                        
                        if len(set(candidate)) != 30:
                            continue
                        
                        s = score_of(candidate)
                        if s > current_score:
                            current = candidate
                            current_score = s
                            current_set = set(current)
                            no_improve = 0
                            improved = True
                            break
                        elif s == current_score:
                            # Accept equal score moves to explore
                            if random.random() < 0.5:
                                current = candidate
                                current_set = set(current)
                                no_improve += 1
                                improved = True
                                break
                    if improved:
                        break
                
                if not improved:
                    no_improve += 1
            
            # Update best if local search improved
            if current_score > best_score:
                best_score = current_score
                best_points = current[:]
>>>>>>> REPLACE

