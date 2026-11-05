<<<<<<< SEARCH
def _try_partition_line_counts(partition, max_iter=200000):
    """
    Given a partition of 12 points into lines (each line has >= 3 points),
    try to assign integer coordinates so that:
      - All points on the same line are collinear.
      - No three points from different lines are collinear (to avoid extra ordinary lines).
    We use a simple backtracking/relaxation approach.
    """
    # This is a placeholder for a more sophisticated search.
    # For now, we'll just return None to indicate this path isn't fully implemented.
    return None
=======
def _structured_search(points, max_rounds=30):
    """
    Structured local search: at each round, identify ordinary lines and try to
    complete them by moving points to positions that create 3-point lines.
    """
    n = len(points)
    current = [p[:] for p in points]
    current_score = _count_ordinary_lines(current)
    best_points = [p[:] for p in current]
    best_score = current_score
    
    for round_num in range(max_rounds):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Find all ordinary lines
        ordinary_lines = _find_ordinary_lines(current)
        
        if not ordinary_lines:
            break
        
        # Try to fix each ordinary line
        improved_this_round = False
        
        for key, (i, j) in ordinary_lines:
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            dx, dy, c = key
            xi, yi = current[i]
            xj, yj = current[j]
            
            # Try moving point i to complete a 3-point line with some other point k
            for k in range(n):
                if k == i or k == j:
                    continue
                
                xk, yk = current[k]
                
                # Check if i, j, k are already collinear (shouldn't be since it's ordinary)
                # Try to move i to a position on line jk that is integer
                dx_kj = xk - xj
                dy_kj = yk - yj
                g_kj = gcd(abs(dx_kj), abs(dy_kj))
                if g_kj == 0:
                    continue
                dx_kj //= g_kj
                dy_kj //= g_kj
                
                # Line through j and k: dy_kj * x - dx_kj * y = c_kj
                c_kj = dy_kj * xj - dx_kj * yj
                
                # We want to find integer point on this line that is close to current i
                # Parameterize: x = xj + t*dx_kj, y = yj + t*dy_kj
                # Find t such that point is near (xi, yi)
                # t ≈ ((xi - xj)*dx_kj + (yi - yj)*dy_kj) / (dx_kj^2 + dy_kj^2)
                denom = dx_kj * dx_kj + dy_kj * dy_kj
                if denom == 0:
                    continue
                t_approx = ((xi - xj) * dx_kj + (yi - yj) * dy_kj) / denom
                t_int = round(t_approx)
                
                # Try t_int and nearby values
                for t_try in [t_int - 2, t_int - 1, t_int, t_int + 1, t_int + 2]:
                    new_x = xj + t_try * dx_kj
                    new_y = yj + t_try * dy_kj
                    
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    # Check if this is a valid new position (not colliding, not same as old)
                    if new_x == xi and new_y == yi:
                        continue
                    
                    conflict = False
                    for m in range(n):
                        if m != i and current[m][0] == new_x and current[m][1] == new_y:
                            conflict = True
                            break
                    if conflict:
                        continue
                    
                    # Check that new position doesn't create a line with only 2 points that is worse
                    # Apply and evaluate
                    old = current[i][:]
                    current[i] = [new_x, new_y]
                    new_score = _count_ordinary_lines(current)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved_this_round = True
                        if new_score < best_score:
                            best_score = new_score
                            best_points = [p[:] for p in current]
                    else:
                        current[i] = old
            
            # Also try moving point j similarly
            for k in range(n):
                if k == i or k == j:
                    continue
                
                xk, yk = current[k]
                
                dx_ki = xk - xi
                dy_ki = yk - yi
                g_ki = gcd(abs(dx_ki), abs(dy_ki))
                if g_ki == 0:
                    continue
                dx_ki //= g_ki
                dy_ki //= g_ki
                
                c_ki = dy_ki * xi - dx_ki * yi
                
                denom = dx_ki * dx_ki + dy_ki * dy_ki
                if denom == 0:
                    continue
                t_approx = ((xj - xi) * dx_ki + (yj - yi) * dy_ki) / denom
                t_int = round(t_approx)
                
                for t_try in [t_int - 2, t_int - 1, t_int, t_int + 1, t_int + 2]:
                    new_x = xi + t_try * dx_ki
                    new_y = yi + t_try * dy_ki
                    
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    if new_x == xj and new_y == yj:
                        continue
                    
                    conflict = False
                    for m in range(n):
                        if m != j and current[m][0] == new_x and current[m][1] == new_y:
                            conflict = True
                            break
                    if conflict:
                        continue
                    
                    old = current[j][:]
                    current[j] = [new_x, new_y]
                    new_score = _count_ordinary_lines(current)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved_this_round = True
                        if new_score < best_score:
                            best_score = new_score
                            best_points = [p[:] for p in current]
                    else:
                        current[j] = old
        
        if not improved_this_round:
            # Try random restarts from best
            random.seed(round_num)
            current = [p[:] for p in best_points]
            current_score = best_score
            
            # Small random perturbations
            for _ in range(100):
                if time.time() > DEADLINE - SAFETY_MARGIN:
                    break
                idx = random.randint(0, 11)
                dx_try = random.randint(-5, 5)
                dy_try = random.randint(-5, 5)
                if dx_try == 0 and dy_try == 0:
                    continue
                new_x = current[idx][0] + dx_try
                new_y = current[idx][1] + dy_try
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                conflict = False
                for m in range(12):
                    if m != idx and current[m][0] == new_x and current[m][1] == new_y:
                        conflict = True
                        break
                if conflict:
                    continue
                old = current[idx][:]
                current[idx] = [new_x, new_y]
                new_score = _count_ordinary_lines(current)
                if new_score < current_score:
                    current_score = new_score
                    improved_this_round = True
                    if new_score < best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current]
                else:
                    current[idx] = old
            
            if not improved_this_round:
                break
    
    return best_points, best_score
>>>>>>> REPLACE

