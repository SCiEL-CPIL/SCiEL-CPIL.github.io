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
    # Enhanced local search with larger perturbation range and multi-point moves
    if best_points is not None and elapsed() < DEADLINE - 3.0:
        random.seed(42)
        current = best_points[:]
        current_score = best_score
        no_improve = 0
        max_no_improve = 200
        current_set = set(current)
        
        # Phase 1: Single-point moves with larger perturbation
        while elapsed() < DEADLINE - 3.0 and no_improve < max_no_improve:
            idx = random.randint(0, 29)
            ox, oy = current[idx]
            
            improved = False
            # Larger perturbation range
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
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
                        if random.random() < 0.3:
                            current = candidate
                            current_set = set(current)
                            no_improve += 1
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
        
        # Phase 2: Two-point simultaneous moves
        if elapsed() < DEADLINE - 1.0:
            no_improve2 = 0
            max_no_improve2 = 50
            while elapsed() < DEADLINE - 1.0 and no_improve2 < max_no_improve2:
                i = random.randint(0, 29)
                j = random.randint(0, 29)
                if i == j:
                    continue
                
                improved = False
                for dx1 in range(-3, 4):
                    for dy1 in range(-3, 4):
                        for dx2 in range(-3, 4):
                            for dy2 in range(-3, 4):
                                if dx1 == 0 and dy1 == 0 and dx2 == 0 and dy2 == 0:
                                    continue
                                ox1, oy1 = current[i]
                                ox2, oy2 = current[j]
                                nx1, ny1 = ox1 + dx1, oy1 + dy1
                                nx2, ny2 = ox2 + dx2, oy2 + dy2
                                
                                if not (-30 <= nx1 <= 30 and -30 <= ny1 <= 30):
                                    continue
                                if not (-30 <= nx2 <= 30 and -30 <= ny2 <= 30):
                                    continue
                                if (nx1, ny1) in current_set or (nx2, ny2) in current_set:
                                    continue
                                if (nx1, ny1) == (nx2, ny2):
                                    continue
                                
                                candidate = current[:]
                                candidate[i] = (nx1, ny1)
                                candidate[j] = (nx2, ny2)
                                
                                if len(set(candidate)) != 30:
                                    continue
                                
                                s = score_of(candidate)
                                if s > current_score:
                                    current = candidate
                                    current_score = s
                                    current_set = set(current)
                                    no_improve2 = 0
                                    improved = True
                                    break
                                elif s == current_score:
                                    if random.random() < 0.2:
                                        current = candidate
                                        current_set = set(current)
                                        no_improve2 += 1
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if not improved:
                    no_improve2 += 1
        
        if current_score > best_score:
            best_score = current_score
            best_points = current[:]
>>>>>>> REPLACE
