"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import itertools
import time
from collections import defaultdict

def compute_line_map(points):
    """Compute line_to_points mapping for 16 points. Returns (line_to_points, max_collinear)."""
    n = len(points)
    line_to_points = defaultdict(set)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            a = dy
            b = -dx
            c = a * x1 + b * y1
            key = (a, b, c)
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    max_collinear = 2
    for key, pts in line_to_points.items():
        if len(pts) > max_collinear:
            max_collinear = len(pts)
    
    return line_to_points, max_collinear

def score_fast(points):
    """Fast score: returns (valid, count_4). Uses precomputed line map."""
    n = len(points)
    if n != 16:
        return (False, 0)
    
    # Check bounds and distinctness
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return (False, 0)
        if (x, y) in seen:
            return (False, 0)
        seen.add((x, y))
    
    line_to_points, max_collinear = compute_line_map(points)
    
    if max_collinear >= 5:
        return (False, 0)
    
    count_4 = 0
    for key, pts in line_to_points.items():
        if len(pts) == 4:
            count_4 += 1
    
    return (True, count_4)

def score(points):
    """Return score: number of exactly-4 lines if valid, else 0."""
    valid, count_4 = score_fast(points)
    return count_4 if valid else 0

def is_valid(points):
    """Check if points are valid."""
    valid, _ = score_fast(points)
    return valid

def local_search(points, time_limit, rng):
    """Perform local search starting from initial points. Returns best found."""
    start_time = time.time()
    best = [p[:] for p in points]
    best_valid, best_score = score_fast(best)
    if not best_valid:
        return None, 0
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        # Pick a random point to move
        idx = rng.randint(0, 15)
        
        # Try several random moves
        found_improvement = False
        for attempt in range(20):
            dx = rng.randint(-50, 50)
            dy = rng.randint(-50, 50)
            if dx == 0 and dy == 0:
                continue
            
            new_pts = [p[:] for p in best]
            new_pts[idx][0] += dx
            new_pts[idx][1] += dy
            
            valid, new_score = score_fast(new_pts)
            if valid and new_score > best_score:
                best = new_pts
                best_score = new_score
                no_improve_count = 0
                found_improvement = True
                break
            elif valid and new_score == best_score:
                # Accept with small probability for diversity
                if rng.random() < 0.1:
                    best = new_pts
                    no_improve_count = 0
                    found_improvement = True
                    break
        
        if not found_improvement:
            no_improve_count += 1
        
        # Occasionally try larger jumps
        if no_improve_count % 100 == 0 and no_improve_count > 0:
            idx = rng.randint(0, 15)
            dx = rng.randint(-200, 200)
            dy = rng.randint(-200, 200)
            new_pts = [p[:] for p in best]
            new_pts[idx][0] += dx
            new_pts[idx][1] += dy
            valid, new_score = score_fast(new_pts)
            if valid and new_score >= best_score:
                best = new_pts
                if new_score > best_score:
                    best_score = new_score
                    no_improve_count = 0
    
    return best, best_score

def simulated_annealing(points, time_limit, rng):
    """Simulated annealing to escape local optima."""
    start_time = time.time()
    current = [p[:] for p in points]
    cur_valid, cur_score = score_fast(current)
    if not cur_valid:
        return None, 0
    
    best = [p[:] for p in current]
    best_score = cur_score
    
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() - start_time < time_limit and T > T_min:
        idx = rng.randint(0, 15)
        dx = rng.randint(-30, 30)
        dy = rng.randint(-30, 30)
        if dx == 0 and dy == 0:
            T *= cooling
            continue
        
        new_pts = [p[:] for p in current]
        new_pts[idx][0] += dx
        new_pts[idx][1] += dy
        
        new_valid, new_score = score_fast(new_pts)
        
        if new_valid:
            if new_score >= cur_score:
                current = new_pts
                cur_score = new_score
                if new_score > best_score:
                    best = new_pts
                    best_score = new_score
            else:
                # Accept with probability exp((new_score - cur_score) / T)
                prob = math.exp((new_score - cur_score) / T)
                if rng.random() < prob:
                    current = new_pts
                    cur_score = new_score
        
        T *= cooling
    
    return best, best_score

def generate_random_points(rng):
    """Generate a random valid starting configuration."""
    points = []
    seen = set()
    attempts = 0
    while len(points) < 16 and attempts < 10000:
        x = rng.randint(-500, 500)
        y = rng.randint(-500, 500)
        if (x, y) not in seen:
            seen.add((x, y))
            points.append([x, y])
        attempts += 1
    
    if len(points) < 16:
        return None
    
    # Check validity
    valid, _ = score_fast(points)
    return points if valid else None

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    random.seed(42)
    rng = random.Random(42)
    
    best = None
    best_score = 0
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    strategies.append(grid)
    
    # Strategy 2: Random points
    for i in range(5):
        pts = generate_random_points(rng)
        if pts:
            strategies.append(pts)
    
    # Strategy 3: Structured random (grid-like with perturbations)
    for i in range(3):
        base_x = [0, 10, 20, 30]
        base_y = [0, 10, 20, 30]
        pts = []
        for x in base_x:
            for y in base_y:
                px = x + rng.randint(-5, 5)
                py = y + rng.randint(-5, 5)
                pts.append([px, py])
        if len(set(tuple(p) for p in pts)) == 16:
            valid, _ = score_fast(pts)
            if valid:
                strategies.append(pts)
    
    # Try each strategy with local search
    per_strategy_time = 10.0
    for strategy_pts in strategies:
        if time.time() > deadline - 5:
            break
        
        # First check if this is already good
        valid, s = score_fast(strategy_pts)
        if valid and s > best_score:
            best = [p[:] for p in strategy_pts]
            best_score = s
        
        # Run local search
        remaining = min(per_strategy_time, deadline - time.time())
        if remaining > 1:
            ls_best, ls_score = local_search(strategy_pts, remaining, rng)
            if ls_best and ls_score > best_score:
                best = ls_best
                best_score = ls_score
    
    # If we have time, run simulated annealing on the best
    if best and time.time() < deadline - 10:
        sa_time = min(20.0, deadline - time.time() - 5)
        if sa_time > 2:
            sa_best, sa_score = simulated_annealing(best, sa_time, rng)
            if sa_best and sa_score > best_score:
                best = sa_best
                best_score = sa_score
    
    # Final safety: if we didn't find anything valid, use the grid
    if best is None:
        best = [[x, y] for x in range(4) for y in range(4)]
    
    # Verify final result
    valid, final_score = score_fast(best)
    if not valid:
        best = [[x, y] for x in range(4) for y in range(4)]
    
    return best
# EVOLVE-BLOCK-END
