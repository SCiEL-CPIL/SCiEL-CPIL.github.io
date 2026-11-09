"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _count_ordinary_lines(pts):
    """Count lines containing exactly two of the points."""
    n = len(pts)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                key = (0, 0, pts[i][0])
            elif dy == 0:
                key = (0, 1, pts[i][1])
            else:
                if dy < 0:
                    dx, dy = -dx, -dy
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                c = dy * pts[i][0] - dx * pts[i][1]
                key = (1, dx, dy, c)
            lines[key] = lines.get(key, 0) + 1
    return sum(1 for c in lines.values() if c == 1)

def _not_all_collinear(points):
    n = len(points)
    if n < 3:
        return True
    for j in range(1, n):
        for k in range(j + 1, n):
            x0, y0 = points[0]
            x1, y1 = points[j]
            x2, y2 = points[k]
            if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
                return True
    return False

def _get_line_key(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 0, x1)
    elif dy == 0:
        return (0, 1, y1)
    else:
        if dy < 0:
            dx, dy = -dx, -dy
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        c = dy * x1 - dx * y1
        return (1, dx, dy, c)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 12
    rng = random.Random(42)
    
    # Initial seeds
    seeds = []
    
    # 1. 3x4 Grid
    grid_34 = [[x, y] for x in range(4) for y in range(3)]
    seeds.append(grid_34)
    
    # 2. 4x3 Grid
    grid_43 = [[x, y] for x in range(3) for y in range(4)]
    seeds.append(grid_43)
    
    # 3. 2x6 Grid
    grid_26 = [[x, y] for x in range(6) for y in range(2)]
    seeds.append(grid_26)
    
    # 4. Hexagon + Center + Extensions
    hex_pts = [
        [2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1],
        [0, 0],
        [3, 0], [0, 2], [-3, 0], [0, -2], [2, 1]
    ]
    # Check distinct
    seen = set()
    hex_unique = []
    for p in hex_pts:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            hex_unique.append(p)
    if len(hex_unique) == 12:
        seeds.append(hex_unique)
        
    # 5. Random
    rand_pts = [[rng.randint(-10, 10), rng.randint(-10, 10)] for _ in range(12)]
    # Ensure distinct
    seen = set()
    rand_unique = []
    for p in rand_pts:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            rand_unique.append(p)
    while len(rand_unique) < 12:
        p = [rng.randint(-10, 10), rng.randint(-10, 10)]
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            rand_unique.append(p)
    seeds.append(rand_unique)

    best_points = None
    best_score = float('inf')
    
    # Evaluate seeds
    for seed in seeds:
        if not _not_all_collinear(seed):
            continue
        score = _count_ordinary_lines(seed)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in seed]
            
    if best_points is None:
        best_points = [[i, i*i] for i in range(12)]
        best_score = _count_ordinary_lines(best_points)

    # Simulated Annealing with Line-Extension Heuristic
    def _generate_candidates(points, idx, max_candidates=20):
        """Generate candidate positions for points[idx] to create collinearities."""
        candidates = []
        old_x, old_y = points[idx]
        other_pts = [points[i] for i in range(n) if i != idx]
        occupied = set((p[0], p[1]) for p in other_pts)
        
        # 1. Line extension candidates
        # For each pair of other points, find integer points on the line
        for i in range(len(other_pts)):
            for j in range(i+1, len(other_pts)):
                x1, y1 = other_pts[i]
                x2, y2 = other_pts[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                
                # Normalize direction
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                g = gcd(abs(dx), abs(dy))
                if g > 0:
                    dx_norm = dx // g
                    dy_norm = dy // g
                else:
                    continue
                
                # Try points on the line near the midpoint or extending
                # Parameter t: point = p1 + t * (dx_norm, dy_norm)
                # We want points that are not p1 or p2
                # Try t values that keep coordinates reasonable
                # Midpoint is t=0.5. 
                # Let's just try a few integer steps from p1
                # Actually, we want points (x,y) such that (x,y) is collinear with p1,p2
                # And (x,y) != p1, p2
                
                # Simple approach: try a few random t values
                for t in range(-3, 4):
                    if t == 0 or t == 1: # Skip p1 and p2 (if p1+p2 vector is normalized, t=1 is p2 only if g=1? No, p1 + 1*(dx/g, dy/g) is not necessarily p2)
                        # Wait, if we use normalized vector, p2 = p1 + k*(dx_norm, dy_norm) where k = g.
                        # So we should avoid t = 0 and t = g.
                        pass
                    
                    new_x = x1 + t * dx_norm
                    new_y = y1 + t * dy_norm
                    
                    if abs(new_x) > 10**8 or abs(new_y) > 10**8:
                        continue
                    
                    if (new_x, new_y) not in occupied:
                        # Check if it's actually on the segment or extension? 
                        # Doesn't matter, just collinear.
                        candidates.append([new_x, new_y])
                        
        # 2. Random nearby candidates
        for _ in range(10):
            nx = old_x + rng.randint(-5, 5)
            ny = old_y + rng.randint(-5, 5)
            if (nx, ny) not in occupied:
                candidates.append([nx, ny])
                
        # 3. Original position (to allow "stay" move implicitly or explicitly)
        candidates.append([old_x, old_y])
        
        # Shuffle and limit
        rng.shuffle(candidates)
        return candidates[:max_candidates]

    def _sa_search(initial_points, time_limit):
        current_points = [p[:] for p in initial_points]
        current_score = _count_ordinary_lines(current_points)
        best_local_points = [p[:] for p in current_points]
        best_local_score = current_score
        
        T = 50.0
        cooling = 0.9999
        min_T = 0.1
        
        while time.time() < time_limit:
            # Pick a random point to move
            idx = rng.randrange(n)
            
            # Generate candidates
            candidates = _generate_candidates(current_points, idx, max_candidates=30)
            
            # Pick a candidate
            new_x, new_y = rng.choice(candidates)
            
            # Check validity
            is_valid = True
            for i in range(n):
                if i != idx:
                    if current_points[i][0] == new_x and current_points[i][1] == new_y:
                        is_valid = False
                        break
            
            if not is_valid:
                continue
                
            # Tentatively move
            old_x, old_y = current_points[idx]
            current_points[idx] = [new_x, new_y]
            
            if not _not_all_collinear(current_points):
                current_points[idx] = [old_x, old_y]
                continue
            
            new_score = _count_ordinary_lines(current_points)
            delta = new_score - current_score
            
            if delta < 0:
                # Accept
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_points = [p[:] for p in current_points]
            else:
                # Accept with probability
                if T > 0:
                    prob = 1.0 / (1.0 + (1.0 - 1.0)) # Placeholder, need exp
                    # P = exp(-delta / T)
                    import math
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        current_score = new_score
                # else reject
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return best_local_points, best_local_score

    # Run SA for multiple seeds
    # Allocate time per seed
    num_sa_runs = 5
    time_per_run = 20
    
    for i in range(num_sa_runs):
        if time.time() > deadline:
            break
            
        seed_idx = i % len(seeds)
        seed = seeds[seed_idx]
        
        # Perturb seed slightly for diversity
        perturbed_seed = [p[:] for p in seed]
        for _ in range(rng.randint(1, 3)):
            idx = rng.randrange(n)
            perturbed_seed[idx] = [perturbed_seed[idx][0] + rng.randint(-2, 2), 
                                    perturbed_seed[idx][1] + rng.randint(-2, 2)]
        
        # Ensure distinct
        seen = set()
        valid_seed = []
        for p in perturbed_seed:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                valid_seed.append(p)
        while len(valid_seed) < 12:
            p = [rng.randint(-10, 10), rng.randint(-10, 10)]
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                valid_seed.append(p)
        
        run_deadline = time.time() + time_per_run
        local_best_pts, local_best_score = _sa_search(valid_seed, run_deadline)
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_points = local_best_pts
            
        if best_score <= 6:
            break

    # Final validation
    assert len(best_points) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    assert _not_all_collinear(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
