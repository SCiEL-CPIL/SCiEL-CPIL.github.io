import random
import math
import time

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Helper to check if a number is a perfect square
    def is_perfect_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    # Check if 3 points are collinear
    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        # Area of triangle * 2
        area2 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        return area2 == 0

    # Check if 4 points are concyclic
    # | x^2+y^2  x  y  1 |
    # | x^2+y^2  x  y  1 | = 0
    # | x^2+y^2  x  y  1 |
    # | x^2+y^2  x  y  1 |
    def is_concyclic(p1, p2, p3, p4):
        rows = []
        for p in (p1, p2, p3, p4):
            x, y = p
            rows.append([x*x + y*y, x, y, 1])
        # Calculate determinant of 4x4 matrix
        # For simplicity and speed with small ints, we can use a generic det or specific logic
        # Since numbers might get large, we rely on Python's big ints
        def det4(m):
            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - 
                             m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1]) -
                    m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - 
                             m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0]) +
                    m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0]) -
                    m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
        return det4(rows) == 0

    def is_valid_set(points):
        n = len(points)
        # Check distinctness
        if len(set(points)) != n:
            return False
        # Check pairwise distances are rational (integers in this search)
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        return False
        # Check no four concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def fitness(points):
        # Count number of integer distance pairs
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if is_perfect_square(d2):
                    count += 1
        return count

    # Seed solution (5 points)
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6-point solution
    # Use multiple strategies: random restarts + hill climbing from seed
    max_restarts = 30
    found_solution = None
    
    def hill_climb(points, max_steps=500, step_size_range=(1, 5)):
        """Hill climbing with variable step sizes"""
        current_fitness = fitness(points)
        improved = True
        steps = 0
        while improved and steps < max_steps and time.time() < deadline - 3.0:
            improved = False
            steps += 1
            # Pick a point to move
            idx = random.randint(0, len(points) - 1)
            old_x, old_y = points[idx]
            
            # Variable step size - larger early, smaller later
            step_scale = max(1, int(step_size_range[1] * (1 - steps / max_steps)))
            dx = random.randint(-step_scale, step_scale)
            dy = random.randint(-step_scale, step_scale)
            if dx == 0 and dy == 0:
                continue
            new_x = old_x + dx
            new_y = old_y + dy
            
            points[idx] = (new_x, new_y)
            new_fitness = fitness(points)
            
            if new_fitness > current_fitness:
                current_fitness = new_fitness
                improved = True
            else:
                points[idx] = (old_x, old_y)
            
            if current_fitness == 15:
                if is_valid_set(points):
                    return points, True
        
        return points, False
    
    # Strategy 1: Start from seed + random 6th point, then optimize
    for r in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if r < 10:
            # Start from seed with random 6th point
            box = 80
            new_pt = (random.randint(-box, box), random.randint(-box, box))
            while new_pt in seed_points:
                new_pt = (random.randint(-box, box), random.randint(-box, box))
            points = seed_points + [new_pt]
        else:
            # Pure random initialization
            box = 60
            points = [(random.randint(-box, box), random.randint(-box, box)) for _ in range(6)]
            while len(set(points)) < 6:
                idx = random.randint(0, 5)
                points[idx] = (random.randint(-box, box), random.randint(-box, box))
        
        result, success = hill_climb(points, max_steps=300, step_size_range=(1, 8))
        if success:
            found_solution = result
            break
    
    # Strategy 2: If not found, try larger search with simulated annealing-like approach
    if not found_solution and time.time() < deadline - 10.0:
        for r in range(10):
            if time.time() > deadline - 5.0:
                break
            
            box = 120
            points = [(random.randint(-box, box), random.randint(-box, box)) for _ in range(6)]
            while len(set(points)) < 6:
                idx = random.randint(0, 5)
                points[idx] = (random.randint(-box, box), random.randint(-box, box))
            
            current_fitness = fitness(points)
            best_points = list(points)
            best_fitness = current_fitness
            temp = 2.0  # Temperature for occasional worse moves
            
            for step in range(200):
                if time.time() > deadline - 3.0:
                    break
                idx = random.randint(0, 5)
                old_x, old_y = points[idx]
                
                # Larger steps for exploration
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                new_x = old_x + dx
                new_y = old_y + dy
                
                points[idx] = (new_x, new_y)
                new_fitness = fitness(points)
                
                delta = new_fitness - current_fitness
                if delta > 0 or random.random() < math.exp(delta / max(temp, 0.1)):
                    current_fitness = new_fitness
                    if new_fitness > best_fitness:
                        best_fitness = new_fitness
                        best_points = list(points)
                    if new_fitness == 15 and is_valid_set(points):
                        found_solution = points
                        break
                else:
                    points[idx] = (old_x, old_y)
                
                temp = max(0.1, temp * 0.98)
            
            if found_solution:
                break
            # Keep best found so far
            points = best_points
            current_fitness = best_fitness
    
    if found_solution:
        pts = found_solution
    else:
        # Fallback to seed
        pts = seed_points
    
    # Final validation check
    if not is_valid_set(pts):
        # If something went wrong, use the known valid seed
        pts = seed_points
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
# EVOLVE-BLOCK-END
