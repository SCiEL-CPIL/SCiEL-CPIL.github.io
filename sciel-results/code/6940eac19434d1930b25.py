import math
import random
import time

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Check if all collinear
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        # Not collinear
        pass
    else:
        # Check rest
        collinear = True
        for i in range(3, n):
            xi, yi = points[i]
            if (x2 - x1) * (yi - y1) != (y2 - y1) * (xi - x1):
                collinear = False
                break
        if collinear:
            return 1000  # Penalty
            
    line_pairs = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            g = math.gcd(math.gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            elif A == 0 and B == 0 and C < 0:
                C = -C
            
            key = (A, B, C)
            line_pairs[key] = line_pairs.get(key, 0) + 1
            
    ordinary = 0
    for pairs in line_pairs.values():
        disc = 1 + 8 * pairs
        sqrt_disc = int(math.isqrt(disc))
        if sqrt_disc * sqrt_disc == disc:
            k = (1 + sqrt_disc) // 2
            if k == 2:
                ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_points = None
    best_cost = float('inf')
    
    # Initial guess: 3x4 grid
    init_points = [[x, y] for x in range(4) for y in range(3)]
    init_cost = count_ordinary_lines(init_points)
    best_points = init_points[:]
    best_cost = init_cost
    
    # Simulated Annealing
    max_restarts = 50
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            points = init_points[:]
        else:
            points = []
            used = set()
            while len(points) < 12:
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                if (x, y) not in used:
                    points.append([x, y])
                    used.add((x, y))
            
        current_cost = count_ordinary_lines(points)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_points = points[:]
            
        T = 10.0
        steps_per_restart = 20000
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            i = random.randint(0, 11)
            old_x, old_y = points[i]
            
            if random.random() < 0.5:
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                if dx == 0 and dy == 0:
                    continue
                new_x = old_x + dx
                new_y = old_y + dy
            else:
                new_x = random.randint(-20, 20)
                new_y = random.randint(-20, 20)
                
            valid = True
            for k in range(12):
                if k != i:
                    if points[k][0] == new_x and points[k][1] == new_y:
                        valid = False
                        break
            if not valid:
                continue
                
            points[i] = [new_x, new_y]
            new_cost = count_ordinary_lines(points)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_points = points[:]
            else:
                points[i] = [old_x, old_y]
                
            T *= 0.9999
            
        if best_cost <= 6:
            break

    return best_points
