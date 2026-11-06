import time
import random
import math

def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * p1[0] + b * p1[1]
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _score_lines(points):
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None: continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    score = 0
    valid = True
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            score += 1
        elif cnt >= 5:
            valid = False
    return score, valid

def _get_candidates_on_line(line_key, existing_indices, points, exclude_indices=None):
    """
    Get candidate integer points on the line defined by line_key,
    excluding points already in 'exclude_indices' (or existing_indices if exclude is None).
    """
    a, b, c = line_key
    if not existing_indices:
        return []
    ref_idx = list(existing_indices)[0]
    x_ref, y_ref = points[ref_idx]
    
    dx_step = b
    dy_step = -a
    
    k_min = 10**9
    k_max = -10**9
    
    if dx_step > 0:
        k_min = max(k_min, math.ceil((-1000 - x_ref) / dx_step))
        k_max = min(k_max, math.floor((1000 - x_ref) / dx_step))
    elif dx_step < 0:
        k_min = max(k_min, math.ceil((1000 - x_ref) / dx_step))
        k_max = min(k_max, math.floor((-1000 - x_ref) / dx_step))
    else:
        if not (-1000 <= x_ref <= 1000):
            return []
        
    if dy_step > 0:
        k_min = max(k_min, math.ceil((-1000 - y_ref) / dy_step))
        k_max = min(k_max, math.floor((1000 - y_ref) / dy_step))
    elif dy_step < 0:
        k_min = max(k_min, math.ceil((1000 - y_ref) / dy_step))
        k_max = min(k_max, math.floor((-1000 - y_ref) / dy_step))
    else:
        if not (-1000 <= y_ref <= 1000):
            return []
            
    if k_min > k_max:
        return []
        
    # Determine which points to exclude
    if exclude_indices is not None:
        exclude_set = set(tuple(points[i]) for i in exclude_indices)
    else:
        exclude_set = set(tuple(points[i]) for i in existing_indices)
        
    candidates = []
    for k in range(k_min, k_max + 1):
        x = x_ref + k * dx_step
        y = y_ref + k * dy_step
        pt = (x, y)
        if pt not in exclude_set:
            candidates.append(pt)
    return candidates

def _local_search(points, deadline, rng, max_iter=200):
    """
    Local search that first tries to fix 5-collinear violations,
    then tries to improve the score by completing 3-lines to 4-lines.
    """
    current = [p[:] for p in points]
    current_score, h5 = _score_lines(current)
    
    # If invalid, we must fix it. We treat validity as a hard constraint.
    # Score is only meaningful if valid.
    best = [p[:] for p in current]
    best_score = current_score if h5 else -1
    
    T = 1.0
    T_min = 0.1
    cooling = 0.99
    
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # 1. Check for 5-collinear violations
        line_pts = {}
        n = len(current)
        for i in range(n):
            for j in range(i + 1, n):
                key = _line_key(current[i], current[j])
                if key is None: continue
                if key not in line_pts:
                    line_pts[key] = set()
                line_pts[key].add(i)
                line_pts[key].add(j)
        
        invalid_lines = [k for k, v in line_pts.items() if len(v) >= 5]
        
        if invalid_lines:
            # Pick a random invalid line and move a point on it
            target_line = rng.choice(invalid_lines)
            pts_on_line = list(line_pts[target_line])
            idx_to_move = rng.choice(pts_on_line)
            
            # Try to move this point to a random nearby location
            old_pt = current[idx_to_move][:]
            found_valid_move = False
            for _ in range(10):
                step = rng.randint(1, 5)
                new_x = old_pt[0] + rng.randint(-step, step)
                new_y = old_pt[1] + rng.randint(-step, step)
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                if any([new_x, new_y] == p for p in current):
                    continue
                current[idx_to_move] = [new_x, new_y]
                new_score, new_h5 = _score_lines(current)
                if new_h5:
                    found_valid_move = True
                    break
                else:
                    current[idx_to_move] = old_pt
            
            if found_valid_move:
                current_score, h5 = _score_lines(current)
                if current_score > best_score:
                    best_score = current_score
                    best = [p[:] for p in current]
            else:
                # If we couldn't fix it easily, keep trying or accept a worse state?
                # For GA, we want valid individuals. If we can't fix it, this individual is dead.
                # But let's try a bigger jump.
                step = rng.randint(5, 20)
                new_x = old_pt[0] + rng.randint(-step, step)
                new_y = old_pt[1] + rng.randint(-step, step)
                if abs(new_x) <= 1000 and abs(new_y) <= 1000 and not any([new_x, new_y] == p for p in current):
                    current[idx_to_move] = [new_x, new_y]
                    new_score, new_h5 = _score_lines(current)
                    if new_h5:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best = [p[:] for p in current]
                    else:
                        current[idx_to_move] = old_pt
        else:
            # 2. No violations. Try to improve score.
            # Look for 3-lines
            three_lines = [k for k, v in line_pts.items() if len(v) == 3]
            
            if three_lines:
                target_line = rng.choice(three_lines)
                pts_on_line = line_pts[target_line]
                # Candidates should be on the line, not currently on the line
                candidates = _get_candidates_on_line(target_line, pts_on_line, current)
                if candidates:
                    new_pt = rng.choice(candidates)
                    # Replace a point NOT on this line to avoid breaking other structures too much?
                    # Or just replace a random point?
                    # Replacing a point on the line would break the 3-line.
                    # We want to ADD a point to the line. So we must replace a point NOT on the line.
                    replace_candidates = [i for i in range(n) if i not in pts_on_line]
                    if replace_candidates:
                        idx = rng.choice(replace_candidates)
                        old_pt = current[idx][:]
                        current[idx] = list(new_pt)
                        
                        new_score, new_h5 = _score_lines(current)
                        
                        if new_h5:
                            delta = new_score - current_score
                            # Accept if better, or worse with probability exp(delta/T)
                            if delta > 0 or rng.random() < math.exp(delta / T):
                                current_score = new_score
                                if current_score > best_score:
                                    best_score = current_score
                                    best = [p[:] for p in current]
                            else:
                                current[idx] = old_pt
                        else:
                            current[idx] = old_pt
            else:
                # No 3-lines. Try random perturbation to find new structures
                idx = rng.randint(0, 15)
                old_pt = current[idx][:]
                step = rng.randint(1, 10)
                new_x = old_pt[0] + rng.randint(-step, step)
                new_y = old_pt[1] + rng.randint(-step, step)
                if abs(new_x) <= 1000 and abs(new_y) <= 1000 and not any([new_x, new_y] == p for p in current):
                    current[idx] = [new_x, new_y]
                    new_score, new_h5 = _score_lines(current)
                    if new_h5:
                        delta = new_score - current_score
                        if delta > 0 or rng.random() < math.exp(delta / T):
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = [p[:] for p in current]
                        else:
                            current[idx] = old_pt
                    else:
                        current[idx] = old_pt
            
        T = max(T_min, T * cooling)

    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    rng = random.Random(42)
    
    best_points = None
    best_score = -1
    
    # Initialize population
    pop_size = 50
    population = []
    scores = []
    
    # Create some diverse initial individuals
    for i in range(pop_size):
        if i == 0:
            # Grid-like start
            points = [[x, y] for x in range(4) for y in range(4)]
            # Scale and shift to avoid 5-collinear if possible, or let repair handle it
            # Actually 4x4 grid has 4 collinear in rows/cols. No 5.
            # But we need distinct points.
            points = [[x*10, y*10] for x in range(4) for y in range(4)]
        else:
            points = []
            for _ in range(16):
                x = rng.randint(-1000, 1000)
                y = rng.randint(-1000, 1000)
                points.append([x, y])
        
        s, h5 = _score_lines(points)
        if h5:
            population.append(points)
            scores.append(s)
        else:
            # Try to repair immediately
            repaired, rep_score = _local_search(points, deadline, rng, max_iter=50)
            if rep_score > -1:
                population.append(repaired)
                scores.append(rep_score)
            else:
                # If repair failed, keep it invalid for now? No, GA needs valid individuals.
                # Discard or keep as low fitness.
                population.append(points)
                scores.append(-100) # Low fitness
        
        if time.time() > deadline - 30:
            break

    if not population:
        return [[0,0], [1,0], [2,0], [3,0], [0,1], [1,1], [2,1], [3,1], 
                [0,2], [1,2], [2,2], [3,2], [0,3], [1,3], [2,3], [3,3]]

    # Update global best
    max_idx = scores.index(max(scores))
    best_score = scores[max_idx]
    best_points = population[max_idx][:]
    
    # GA Loop
    while time.time() < deadline - 10:
        new_population = []
        new_scores = []
        
        # Elitism: Keep top 10%
        sorted_indices = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        elite_count = max(5, pop_size // 10)
        for i in range(elite_count):
            idx = sorted_indices[i]
            new_population.append(population[idx][:])
            new_scores.append(scores[idx])
            
            if scores[idx] > best_score:
                best_score = scores[idx]
                best_points = population[idx][:]
                
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10:
                break
                
            # Selection
            # Tournament selection
            p1_idx = rng.choice(sorted_indices[:pop_size//2]) # Pick from top half
            p2_idx = rng.choice(sorted_indices[:pop_size//2])
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover
            # Select random subset of indices from p1
            child = [None] * 16
            mask = [rng.random() < 0.5 for _ in range(16)]
            
            # Ensure we take at least a few from each?
            # Just random mix
            for i in range(16):
                if mask[i]:
                    child[i] = p1[i][:]
                else:
                    child[i] = p2[i][:]
            
            # Mutation
            for i in range(16):
                if rng.random() < 0.2:
                    step = rng.randint(1, 20)
                    new_x = child[i][0] + rng.randint(-step, step)
                    new_y = child[i][1] + rng.randint(-step, step)
                    if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                        # Check uniqueness
                        if not any([new_x, new_y] == child[j] for j in range(16) if j != i):
                            child[i] = [new_x, new_y]
            
            # Local Repair/Improvement
            improved, score = _local_search(child, deadline, rng, max_iter=100)
            
            new_population.append(improved)
            new_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_points = improved[:]
                
        population = new_population
        scores = new_scores

    return best_points
