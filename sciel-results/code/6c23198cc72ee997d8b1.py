import time
import random
import math
from itertools import combinations
from collections import defaultdict

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g != 0:
                a //= g
                b //= g
                c //= g
            
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    return sum(1 for count in line_counts.values() if count == 2)


def _all_collinear(points):
    """Check if all points are collinear."""
    n = len(points)
    if n < 3:
        return False
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx = x1 - x0
    dy = y1 - y0
    for i in range(2, n):
        xi, yi = points[i]
        if dx * (yi - y0) != dy * (xi - x0):
            return False
    return True


def _get_lines_with_points(points):
    """Return dict mapping line key -> list of point indices on that line."""
    n = len(points)
    line_points = defaultdict(list)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g != 0:
                a //= g
                b //= g
                c //= g
            
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            line_points[key].append(i)
            line_points[key].append(j)
    
    # Deduplicate
    for key in line_points:
        line_points[key] = list(set(line_points[key]))
    
    return line_points


def _candidate_positions(points, idx, max_range=100):
    """Generate candidate positions for moving point idx to create 3+ collinearities."""
    n = len(points)
    candidates = set()
    
    # For every pair of other points, find integer positions on their line
    for i1 in range(n):
        if i1 == idx:
            continue
        for i2 in range(n):
            if i2 == idx or i2 <= i1:
                continue
            
            x1, y1 = points[i1]
            x2, y2 = points[i2]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            # Try to place point at integer positions on this line
            # Point = p1 + t * (dx/g, dy/g) where g = gcd(dx,dy)
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            step_x = dx // g
            step_y = dy // g
            
            # Try t values that give integer coordinates within range
            # x = x1 + t*step_x, y = y1 + t*step_y
            for t in range(-max_range, max_range + 1):
                nx = x1 + t * step_x
                ny = y1 + t * step_y
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                # Check not already occupied by another point
                occupied = False
                for k in range(n):
                    if k != idx and points[k][0] == nx and points[k][1] == ny:
                        occupied = True
                        break
                if not occupied:
                    candidates.add((nx, ny))
    
    return candidates


def _simulated_annealing(points, deadline, max_iters=20000):
    """Simulated annealing to minimize ordinary lines."""
    current = [p[:] for p in points]
    current_count = count_ordinary_lines(current)
    n = len(current)
    
    best = [p[:] for p in current]
    best_count = current_count
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Choose move strategy
        idx = random.randint(0, n - 1)
        
        if random.random() < 0.7:
            # Targeted move: try to place on line through two other points
            others = [i for i in range(n) if i != idx]
            i1, i2 = random.sample(others, 2)
            x1, y1 = current[i1]
            x2, y2 = current[i2]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            step_x = dx // g
            step_y = dy // g
            
            # Try a few t values
            t_candidates = list(range(-10, 11))
            random.shuffle(t_candidates)
            moved = False
            for t in t_candidates:
                nx = x1 + t * step_x
                ny = y1 + t * step_y
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                # Check not occupied
                occupied = False
                for k in range(n):
                    if k != idx and current[k][0] == nx and current[k][1] == ny:
                        occupied = True
                        break
                if occupied:
                    continue
                
                test = [p[:] for p in current]
                test[idx] = [nx, ny]
                if _all_collinear(test):
                    continue
                
                new_count = count_ordinary_lines(test)
                delta = new_count - current_count
                
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current = test
                    current_count = new_count
                    if new_count < best_count:
                        best = [p[:] for p in current]
                        best_count = new_count
                    moved = True
                    break
        else:
            # Random small perturbation
            for _ in range(3):
                nx = current[idx][0] + random.randint(-3, 3)
                ny = current[idx][1] + random.randint(-3, 3)
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                occupied = False
                for k in range(n):
                    if k != idx and current[k][0] == nx and current[k][1] == ny:
                        occupied = True
                        break
                if occupied:
                    continue
                
                test = [p[:] for p in current]
                test[idx] = [nx, ny]
                if _all_collinear(test):
                    continue
                
                new_count = count_ordinary_lines(test)
                delta = new_count - current_count
                
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current = test
                    current_count = new_count
                    if new_count < best_count:
                        best = [p[:] for p in current]
                        best_count = new_count
                    break
        
        T *= cooling
    
    return best, best_count


def _initial_configurations():
    """Generate diverse initial configurations."""
    configs = []
    
    # 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3x4 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Two lines of 6
    configs.append([[i, 0] for i in range(6)] + [[i, 1] for i in range(6)])
    
    # Three lines of 4
    configs.append([[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(4)])
    
    # Four lines of 3
    configs.append([[i, 0] for i in range(3)] + [[i, 1] for i in range(3)] + 
                   [[i, 2] for i in range(3)] + [[i, 3] for i in range(3)])
    
    # Hexagonal-like arrangement
    hex_pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2]
    ]
    configs.append(hex_pts)
    
    # Diamond/rhombus pattern
    diamond = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2]
    ]
    # Shift to create more collinearities
    configs.append([[x + (y % 2), y] for x in range(4) for y in range(3)])
    
    # Pappus-like configuration
    pappus = [
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [3, 0], [3, 1], [3, 2]
    ]
    configs.append(pappus)
    
    # Random configurations
    random.seed(123)
    for _ in range(5):
        pts = []
        used = set()
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        if not _all_collinear(pts):
            configs.append(pts)
    
    return configs


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    random.seed(42)
    
    best_points = None
    best_count = float('inf')
    
    # Try each initial configuration with simulated annealing
    for trial, init_pts in enumerate(_initial_configurations()):
        if time.time() > deadline - 10:
            break
        
        # Quick check
        init_count = count_ordinary_lines(init_pts)
        if init_count < best_count:
            best_count = init_count
            best_points = [p[:] for p in init_pts]
        
        # Run simulated annealing
        if time.time() > deadline - 15:
            break
        
        result_pts, result_count = _simulated_annealing(init_pts, deadline, max_iters=15000)
        
        if result_count < best_count:
            best_count = result_count
            best_points = result_pts
        
        # If we hit theoretical minimum, stop
        if best_count <= 6:
            break
    
    # Final polish with more iterations if time permits
    if best_count > 6 and time.time() < deadline - 20:
        result_pts, result_count = _simulated_annealing(best_points, deadline, max_iters=30000)
        if result_count < best_count:
            best_count = result_count
            best_points = result_pts
    
    return best_points


if __name__ == "__main__":
    pts = construct()
    print(f"Number of points: {len(pts)}")
    print(f"Ordinary lines: {count_ordinary_lines(pts)}")
    print(f"Points: {pts}")
