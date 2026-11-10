import time
import math
import random
from collections import defaultdict

def _count_ordinary_lines(points):
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(dx, dy)
            if g == 0:
                continue
                
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            
            line_counts[key] += 1
            
    ordinary_count = 0
    for count in line_counts.values():
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def _generate_initial_populations():
    """Generate diverse initial populations."""
    populations = []
    
    # 1. 3x4 Grid
    p1 = [[x, y] for x in range(4) for y in range(3)]
    populations.append(p1)
    
    # 2. 4x3 Grid
    p2 = [[x, y] for x in range(3) for y in range(4)]
    populations.append(p2)
    
    # 3. Points on two parallel lines (6 each)
    # Try to make many 3-point lines within the rows
    p3 = []
    for i in range(6):
        p3.append([i, 0])
        p3.append([i, 1])
    populations.append(p3)
    
    # 4. Points on three concurrent lines (4 each)
    p4 = []
    # Line y=0
    p4 += [[0,0], [1,0], [2,0], [3,0]]
    # Line y=x
    p4 += [[0,1], [1,2], [2,3], [3,4]] # Shifted to avoid duplicates with y=0? (0,0) is shared.
    # We need distinct points.
    # Let's use a star pattern.
    p4 = []
    center = [0,0]
    p4.append(center)
    # 3 lines through center, 3 other points each? Total 1 + 3*3 = 10. Need 2 more.
    # This is getting complex. Let's stick to grids and perturbations.
    
    # 5. Random in box [-5, 5]
    for _ in range(5):
        p5 = []
        seen = set()
        while len(p5) < 12:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if (x,y) not in seen:
                seen.add((x,y))
                p5.append([x,y])
        populations.append(p5)
        
    # 6. "Orchard" known good configs?
    # Try a configuration with many 3-point lines.
    # e.g. 3 rows of 4, but shifted.
    p6 = []
    for i in range(4):
        p6.append([i, 0])
    for i in range(4):
        p6.append([i+1, 1])
    for i in range(4):
        p6.append([i+2, 2])
    if len(set(map(tuple, p6))) == 12:
        populations.append(p6)
        
    return populations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    initial_pops = _generate_initial_populations()
    
    best_points = None
    best_score = float('inf')
    
    for init_points in initial_pops:
        if time.time() > deadline:
            break
            
        current_points = [p[:] for p in init_points]
        current_score = _count_ordinary_lines(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        if best_score == 6:
            break
            
        # Local Search: Simulated Annealing
        temp = 10.0
        cooling = 0.99
        
        # Run SA for a fixed number of steps or until time
        steps = 0
        while time.time() < deadline - 5 and steps < 5000:
            steps += 1
            idx = random.randint(0, 11)
            old_x, old_y = current_points[idx]
            
            # Move to random nearby point
            # Use a slightly larger jump size to escape local minima
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
            
            if not (-10**9 <= new_x <= 10**9 and -10**9 <= new_y <= 10**9):
                continue
                
            new_points = [p[:] for p in current_points]
            new_points[idx] = [new_x, new_y]
            
            # Check distinctness
            is_distinct = True
            for k, p in enumerate(new_points):
                if k == idx:
                    continue
                if p[0] == new_x and p[1] == new_y:
                    is_distinct = False
                    break
                    
            if not is_distinct:
                continue
                
            new_score = _count_ordinary_lines(new_points)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_points = new_points
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    
                    if best_score == 6:
                        break
            else:
                pass # Reject
            
            temp *= cooling
            
        if best_score == 6:
            break

    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
