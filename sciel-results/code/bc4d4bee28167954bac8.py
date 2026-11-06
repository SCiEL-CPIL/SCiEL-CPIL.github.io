"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def _count_exact_four_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    # Map each line (represented by normalized (a,b,c) for ax+by+c=0) to count
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2): (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
            a = y2 - y1
            b = -(x2 - x1)
            c = x2 * y1 - x1 * y2
            # Normalize
            from math import gcd
            g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a or b or c) else 1
            if g == 0:
                g = 1
            a //= g
            b //= g
            c //= g
            # Ensure canonical sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            line_counts[(a, b, c)] += 1
    
    count = 0
    for key, cnt in line_counts.items():
        if cnt == 6:  # 4 points -> C(4,2)=6 pairs
            count += 1
    return count

def _has_five_collinear(points):
    """Check if any 5 points are collinear."""
    n = len(points)
    if n < 5:
        return False
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            a = y2 - y1
            b = -(x2 - x1)
            c = x2 * y1 - x1 * y2
            from math import gcd
            g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a or b or c) else 1
            if g == 0:
                g = 1
            a //= g
            b //= g
            c //= g
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            line_counts[(a, b, c)] += 1
    for key, cnt in line_counts.items():
        if cnt >= 10:  # C(5,2) = 10
            return True
    return False

def _validity(points):
    """Check all constraints: 16 distinct points, |x|,|y|<=1000, no five collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if not (-1000 <= x <= 1000 and -1000 <= y <= 1000):
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if _has_five_collinear(points):
        return False
    return True

def _score(points):
    """Score: number of exact 4-point lines, 0 if invalid."""
    if not _validity(points):
        return 0
    return _count_exact_four_lines(points)

def _initial_population(pop_size, rng):
    """Generate initial population."""
    pop = []
    # Start with the 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    pop.append(list(grid))
    for _ in range(pop_size - 1):
        pts = []
        used = set()
        while len(pts) < 16:
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        pop.append(pts)
    return pop

def _crossover(p1, p2, rng):
    """Crossover: select random subset from p1, complement from p2."""
    indices = list(range(16))
    rng.shuffle(indices)
    k = rng.randint(4, 12)
    from_p1 = set(indices[:k])
    child = []
    for i in range(16):
        if i in from_p1:
            child.append(list(p1[i]))
        else:
            child.append(list(p2[i]))
    return child

def _mutate(points, rng, strength=3):
    """Mutate: perturb some points slightly."""
    pts = [list(p) for p in points]
    n_mut = rng.randint(1, 3)
    for _ in range(n_mut):
        i = rng.randint(0, 15)
        dx = rng.randint(-strength, strength)
        dy = rng.randint(-strength, strength)
        pts[i][0] += dx
        pts[i][1] += dy
    return pts

def _local_repair(points, rng, max_iter=200):
    """Repair: break 5-collinear sets, try to improve 4-line count."""
    pts = [list(p) for p in points]
    
    for iteration in range(max_iter):
        # Check for 5-collinear
        if _has_five_collinear(pts):
            # Find a 5-collinear set and move one point
            line_counts = defaultdict(list)
            for i in range(16):
                for j in range(i+1, 16):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    a = y2 - y1
                    b = -(x2 - x1)
                    c = x2 * y1 - x1 * y2
                    from math import gcd
                    g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a or b or c) else 1
                    if g == 0:
                        g = 1
                    a //= g
                    b //= g
                    c //= g
                    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                        a, b, c = -a, -b, -c
                    line_counts[(a, b, c)].append((i, j))
            
            bad_line = None
            for key, pairs in line_counts.items():
                if len(pairs) >= 10:
                    bad_line = key
                    break
            
            if bad_line:
                # Get all points on this line
                a, b, c = bad_line
                on_line = []
                for i in range(16):
                    x, y = pts[i]
                    if a*x + b*y + c == 0:
                        on_line.append(i)
                # Move a random point on the line
                if on_line:
                    idx = rng.choice(on_line)
                    dx = rng.randint(-5, 5)
                    dy = rng.randint(-5, 5)
                    if dx == 0 and dy == 0:
                        dy = 1
                    new_x = pts[idx][0] + dx
                    new_y = pts[idx][1] + dy
                    if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                        # Check no duplicate
                        new_pt = (new_x, new_y)
                        if new_pt not in [(p[0], p[1]) for i, p in enumerate(pts) if i != idx]:
                            pts[idx] = [new_x, new_y]
                    else:
                        pts[idx] = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
                    continue
        
        # Try to improve: find a pair that could form a new 4-line
        current_score = _count_exact_four_lines(pts)
        improved = False
        # Try moving one point to create a new 4-collinear
        for attempt in range(20):
            i = rng.randint(0, 15)
            # Try to find 3 other points that are nearly collinear with pts[i]
            best_move = None
            best_gain = 0
            for j in range(16):
                if j == i:
                    continue
                for k in range(16):
                    if k == i or k == j:
                        continue
                    x1, y1 = pts[j]
                    x2, y2 = pts[k]
                    # Line through j and k
                    a = y2 - y1
                    b = -(x2 - x1)
                    c = x2 * y1 - x1 * y2
                    if a == 0 and b == 0:
                        continue
                    # Check how many other points are on this line
                    cnt = 0
                    for m in range(16):
                        if m == j or m == k:
                            continue
                        x, y = pts[m]
                        if a*x + b*y + c == 0:
                            cnt += 1
                    # If cnt == 2, then adding pts[i] to this line makes 4
                    if cnt == 2:
                        # Check if pts[i] is close to this line
                        dist = abs(a*pts[i][0] + b*pts[i][1] + c)
                        # Check if moving pts[i] onto this line helps
                        # Project pts[i] onto line
                        # For simplicity, just check if it's a valid position
                        # Find a point on the line near pts[i]
                        # Line: ax + by + c = 0
                        # Nearest integer point on line to (x0,y0)
                        x0, y0 = pts[i]
                        # If a != 0: y = (-ax - c)/b if b != 0
                        # Just try a few nearby positions
                        for dx in range(-3, 4):
                            for dy in range(-3, 4):
                                nx, ny = x0 + dx, y0 + dy
                                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                                    if a*nx + b*ny + c == 0:
                                        # Check no duplicate
                                        if (nx, ny) not in [(p[0], p[1]) for m, p in enumerate(pts) if m != i]:
                                            # Check no 5-collinear after move
                                            test_pts = [list(p) for p in pts]
                                            test_pts[i] = [nx, ny]
                                            if not _has_five_collinear(test_pts):
                                                test_score = _count_exact_four_lines(test_pts)
                                                if test_score > current_score:
                                                    best_move = (nx, ny)
                                                    best_gain = test_score - current_score
                                                    break
                            if best_move:
                                break
                        if best_move:
                            break
                    if best_move:
                        break
                if best_move:
                    break
            if best_move:
                pts[i] = list(best_move)
                improved = True
                break
        
        if not improved and iteration > 50:
            break
    
    return pts

def construct():
    """Main entry point: run genetic algorithm to find 16 points maximizing exact 4-point lines."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s
    
    rng = random.Random(42)
    
    # Initialize population
    pop_size = 30
    population = _initial_population(pop_size, rng)
    
    # Evaluate initial population
    best_pts = None
    best_score = 0
    for pts in population:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = [list(p) for p in pts]
    
    # Genetic algorithm
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Selection: tournament
        def tournament():
            candidates = []
            for _ in range(5):
                idx = rng.randint(0, len(population) - 1)
                candidates.append((population[idx], _score(population[idx])))
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        # Create new generation
        new_pop = []
        
        # Keep best individual
        new_pop.append([list(p) for p in best_pts])
        
        while len(new_pop) < pop_size:
            if time.time() > deadline - 5:
                break
            
            p1 = tournament()
            p2 = tournament()
            child = _crossover(p1, p2, rng)
            
            # Mutation
            if rng.random() < 0.7:
                child = _mutate(child, rng, strength=4)
            
            # Local repair
            if time.time() < deadline - 10:
                child = _local_repair(child, rng, max_iter=100)
            
            # Ensure validity
            if not _validity(child):
                # Try to fix
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    child = _mutate(child, rng, strength=5)
                    if _validity(child):
                        break
            
            if _validity(child):
                new_pop.append(child)
            else:
                new_pop.append([list(p) for p in best_pts])
        
        # Update population
        population = new_pop[:pop_size]
        
        # Update best
        for pts in population:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = [list(p) for p in pts]
        
        # Elitism: ensure best is in population
        population[0] = [list(p) for p in best_pts]
    
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
