"""Genetic algorithm for 16 integer points, maximize exact-4 lines."""
import time
import random
import math
from itertools import combinations

def _line_key(p1, p2):
    """Compute normalized line key (a, b, c) for line through p1, p2: a*x + b*y = c."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
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
    """
    Returns (num_exact_4_lines, has_5_collinear).
    points: list of 16 [x, y]
    """
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    exact_4 = 0
    has_5 = False
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _fix_collinearity(points, rng, max_iters=200):
    """
    Shift points to break any 5-collinear sets.
    Returns (fixed_points, success).
    """
    pts = [p[:] for p in points]
    for _ in range(max_iters):
        # Find any line with 5+ points
        line_pts = {}
        for i in range(16):
            for j in range(i + 1, 16):
                key = _line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in line_pts:
                    line_pts[key] = set()
                line_pts[key].add(i)
                line_pts[key].add(j)
        
        bad_line = None
        bad_indices = None
        for key, indices in line_pts.items():
            if len(indices) >= 5:
                bad_line = key
                bad_indices = indices
                break
        
        if bad_line is None:
            return pts, True
        
        # Pick a random point from the bad line and move it
        idx = rng.choice(list(bad_indices))
        a, b, c = bad_line
        # Move perpendicular to the line
        # Line direction: (b, -a) normalized
        # Perpendicular: (a, b)
        step = rng.randint(1, 5)
        new_x = pts[idx][0] + a * step
        new_y = pts[idx][1] + b * step
        
        # Check bounds and collision
        if abs(new_x) <= 1000 and abs(new_y) <= 1000:
            if not any([new_x, new_y] == p for p in pts):
                pts[idx] = [new_x, new_y]
        
        # If still bad, try another direction
        if any(len(v) >= 5 for v in line_pts.values()):
            # Try negative step
            new_x = pts[idx][0] - a * step
            new_y = pts[idx][1] - b * step
            if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                if not any([new_x, new_y] == p for p in pts):
                    pts[idx] = [new_x, new_y]
    
    # Final check
    line_pts = {}
    for i in range(16):
        for j in range(i + 1, 16):
            key = _line_key(pts[i], pts[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    has_5 = any(len(v) >= 5 for v in line_pts.values())
    return pts, not has_5

def _local_optimize(points, rng, deadline, max_iter=500):
    """
    Local search to improve the score.
    Uses targeted moves: complete 3-point lines, perturb points.
    """
    current = [p[:] for p in points]
    current_score, h5 = _score_lines(current)
    if h5:
        current, ok = _fix_collinearity(current, rng)
        if not ok:
            return current, 0
        current_score, h5 = _score_lines(current)
    
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 2.0
    T_min = 0.1
    cooling = 0.995
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Targeted: Complete a 3-point line
            line_pts = {}
            for i in range(16):
                for j in range(i + 1, 16):
                    key = _line_key(current[i], current[j])
                    if key is None:
                        continue
                    if key not in line_pts:
                        line_pts[key] = set()
                    line_pts[key].add(i)
                    line_pts[key].add(j)
            
            three_lines = [k for k, v in line_pts.items() if len(v) == 3]
            
            if three_lines:
                target_line = rng.choice(three_lines)
                a, b, c = target_line
                pts_on_line = line_pts[target_line]
                
                # Find integer points on this line
                # Direction: (b, -a)
                ref_idx = list(pts_on_line)[0]
                x_ref, y_ref = current[ref_idx]
                dx_step = b
                dy_step = -a
                
                candidates = []
                # Search along the line
                for k in range(-200, 201):
                    x = x_ref + k * dx_step
                    y = y_ref + k * dy_step
                    if abs(x) <= 1000 and abs(y) <= 1000:
                        pt = (x, y)
                        if not any(pt == tuple(p) for p in current):
                            candidates.append(pt)
                
                if candidates:
                    new_pt = rng.choice(candidates)
                    replace_candidates = [i for i in range(16) if i not in pts_on_line]
                    if replace_candidates:
                        idx = rng.choice(replace_candidates)
                        old_pt = current[idx][:]
                        current[idx] = list(new_pt)
                        
                        new_score, new_h5 = _score_lines(current)
                        
                        if not new_h5:
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
        
        elif move_type < 0.8:
            # Perturb a random point
            idx = rng.randint(0, 15)
            old_pt = current[idx][:]
            step = rng.randint(1, 15)
            new_x = old_pt[0] + rng.randint(-step, step)
            new_y = old_pt[1] + rng.randint(-step, step)
            
            if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                if not any([new_x, new_y] == p for p in current):
                    current[idx] = [new_x, new_y]
                    new_score, new_h5 = _score_lines(current)
                    
                    if not new_h5:
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
        
        else:
            # Swap two points
            i, j = rng.sample(range(16), 2)
            old_i, old_j = current[i][:], current[j][:]
            current[i], current[j] = old_j, old_i
            
            new_score, new_h5 = _score_lines(current)
            
            if not new_h5:
                delta = new_score - current_score
                if delta > 0 or rng.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = [p[:] for p in current]
                else:
                    current[i], current[j] = old_i, old_j
            else:
                current[i], current[j] = old_i, old_j
        
        T = max(T_min, T * cooling)
    
    return best, best_score

def _crossover(parent1, parent2, rng):
    """
    Crossover: select random subset of indices from parent1, rest from parent2.
    """
    child = []
    # Random subset of indices
    subset_size = rng.randint(4, 12)
    subset = set(rng.sample(range(16), subset_size))
    
    for i in range(16):
        if i in subset:
            child.append(parent1[i][:])
        else:
            child.append(parent2[i][:])
    
    return child

def _mutation(points, rng, rate=0.3):
    """
    Mutation: perturb some points.
    """
    child = [p[:] for p in points]
    for i in range(16):
        if rng.random() < rate:
            step = rng.randint(1, 10)
            new_x = child[i][0] + rng.randint(-step, step)
            new_y = child[i][1] + rng.randint(-step, step)
            if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                if not any([new_x, new_y] == p for p in child):
                    child[i] = [new_x, new_y]
    return child

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rng = random.Random(42)
    
    # Initialize population
    pop_size = 20
    
    # Base: 4x4 grid
    base = [[x, y] for x in range(4) for y in range(4)]
    base_score, _ = _score_lines(base)
    
    population = []
    scores = []
    
    # Add base
    population.append(base)
    scores.append(base_score)
    
    # Add perturbed versions
    for _ in range(pop_size - 1):
        current = [p[:] for p in base]
        for _ in range(rng.randint(5, 15)):
            idx = rng.randint(0, 15)
            new_x = current[idx][0] + rng.randint(-20, 20)
            new_y = current[idx][1] + rng.randint(-20, 20)
            new_pt = [new_x, new_y]
            if any(new_pt == p for p in current):
                continue
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            current[idx] = new_pt
        
        # Fix collinearity
        current, ok = _fix_collinearity(current, rng)
        if ok:
            s, h5 = _score_lines(current)
            if not h5:
                population.append(current)
                scores.append(s)
    
    # Ensure population size
    while len(population) < pop_size:
        current = [p[:] for p in base]
        for _ in range(rng.randint(10, 20)):
            idx = rng.randint(0, 15)
            new_x = current[idx][0] + rng.randint(-30, 30)
            new_y = current[idx][1] + rng.randint(-30, 30)
            new_pt = [new_x, new_y]
            if any(new_pt == p for p in current):
                continue
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            current[idx] = new_pt
        current, ok = _fix_collinearity(current, rng)
        if ok:
            s, h5 = _score_lines(current)
            if not h5:
                population.append(current)
                scores.append(s)
        else:
            population.append(base)
            scores.append(base_score)
    
    best_points = [p[:] for p in population[0]]
    best_score = max(scores)
    
    # Evolutionary loop
    gen = 0
    while time.time() < deadline - 10:
        gen += 1
        
        # Sort by score
        indices = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        
        # Elitism: keep top half
        elite = [population[i][:] for i in indices[:len(population)//2]]
        elite_scores = [scores[i] for i in indices[:len(population)//2]]
        
        # Generate new population
        new_pop = []
        new_scores = []
        
        # Add elites
        for i in range(len(elite)):
            new_pop.append(elite[i])
            new_scores.append(elite_scores[i])
        
        # Crossover and mutation
        while len(new_pop) < pop_size:
            if time.time() > deadline - 5:
                break
            
            # Select parents (tournament selection)
            p1_idx = indices[rng.randint(0, min(5, len(indices)-1))]
            p2_idx = indices[rng.randint(0, min(5, len(indices)-1))]
            
            child = _crossover(population[p1_idx], population[p2_idx], rng)
            child = _mutation(child, rng)
            
            # Fix collinearity
            child, ok = _fix_collinearity(child, rng)
            if ok:
                s, h5 = _score_lines(child)
                if not h5:
                    new_pop.append(child)
                    new_scores.append(s)
                else:
                    new_pop.append(child)
                    new_scores.append(0)
            else:
                new_pop.append(child)
                new_scores.append(0)
        
        # Local optimize the best few
        for i in range(min(5, len(new_pop))):
            if time.time() > deadline - 5:
                break
            idx = sorted(range(len(new_pop)), key=lambda i: new_scores[i], reverse=True)[i]
            if new_scores[idx] > 0:
                optimized, opt_score = _local_optimize(new_pop[idx], rng, deadline, max_iter=100)
                new_pop[idx] = optimized
                new_scores[idx] = opt_score
        
        population = new_pop
        scores = new_scores
        
        # Update best
        max_idx = scores.index(max(scores))
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_points = [p[:] for p in population[max_idx]]
    
    # Final verification
    final_score, final_h5 = _score_lines(best_points)
    if final_h5:
        # Fallback to base
        best_points = [p[:] for p in base]
        final_score, _ = _score_lines(best_points)
    
    return best_points
