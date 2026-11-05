import random
import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Helper to check if a number is a perfect square
    def is_perfect_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    # Check if 3 points are collinear (using integer coords)
    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        # Area of triangle * 2
        area2 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        return area2 == 0

    # Check if 4 points are concyclic (using integer coords)
    def is_concyclic(p1, p2, p3, p4):
        rows = []
        for p in (p1, p2, p3, p4):
            x, y = p
            rows.append([x*x + y*y, x, y, 1])
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
        if len(set(points)) != n:
            return False
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def fitness(points):
        # Count number of integer distance pairs, penalize collinearities
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if is_perfect_square(d2):
                    count += 1
        # Penalize collinear triples
        penalty = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        penalty += 10
        return count - penalty

    # Seed solution (5 points)
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6-point solution using multiple strategies
    max_restarts = 100
    found_solution = None
    
    for r in range(max_restarts):
        if time.time() > deadline - 3.0:
            break
            
        # Strategy 1: Start from seed and add points
        if r % 3 == 0:
            # Use seed points as base, add 1-2 random points
            base = seed_points[:]
            num_add = 1
            points = base + [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(num_add)]
            while len(set(points)) < 6:
                idx = random.randint(0, 5)
                points[idx] = (random.randint(-100, 100), random.randint(-100, 100))
        else:
            # Strategy 2: Random initialization with larger box
            box = 150
            points = [(random.randint(-box, box), random.randint(-box, box)) for _ in range(6)]
            while len(set(points)) < 6:
                idx = random.randint(0, 5)
                points[idx] = (random.randint(-box, box), random.randint(-box, box))
        
        current_fitness = fitness(points)
        
        # Hill Climbing with adaptive step size
        improved = True
        steps = 0
        step_size = 5
        while improved and steps < 2000:
            if time.time() > deadline - 1.0:
                break
            improved = False
            steps += 1
            # Pick a point to move
            idx = random.randint(0, 5)
            old_x, old_y = points[idx]
            
            # Adaptive step: larger early, smaller later
            dx = random.randint(-step_size, step_size)
            dy = random.randint(-step_size, step_size)
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Keep in reasonable bounds
            if abs(new_x) > 300 or abs(new_y) > 300:
                continue
                
            points[idx] = (new_x, new_y)
            new_fitness = fitness(points)
            
            if new_fitness > current_fitness:
                current_fitness = new_fitness
                improved = True
                step_size = max(1, step_size - 1)
            else:
                points[idx] = (old_x, old_y)
            
            # If we reached full fitness (15 pairs), check validity
            if current_fitness == 15:
                if is_valid_set(points):
                    found_solution = points
                    break
        
        if found_solution:
            break
    
    if found_solution:
        pts = found_solution
    else:
        # Fallback to seed
        pts = seed_points
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
# EVOLVE-BLOCK-END
