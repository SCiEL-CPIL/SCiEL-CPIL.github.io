import math
import random
import time
from collections import defaultdict

def count_four_lines(points):
    """
    Counts the number of lines containing exactly 4 points.
    Returns -1 if any line contains 5 or more points.
    """
    n = len(points)
    line_counts = defaultdict(int)
    
    # Precompute point coordinates for speed
    pts = [(p[0], p[1]) for p in points]
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            
            # Normalize the line representation
            # A line is defined by (A, B, C) such that Ax + By + C = 0
            # To make it unique, we normalize (A, B, C).
            # The direction vector is (dx, dy). The normal is (dy, -dx).
            # So A = dy, B = -dx.
            A = dy
            B = -dx
            C = -(A * x1 + B * y1)
            
            # Normalize (A, B, C) by GCD
            g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            # Ensure a canonical sign (e.g., first non-zero of A,B is positive)
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            line_id = (A, B, C)
            line_counts[line_id] += 1
            
            # Early termination if we find a 5-point line?
            # We can't just return -1 immediately because we need to know if there are OTHER 4-point lines?
            # The task says "no five collinear". If there are 5 collinear, the configuration is INVALID.
            # So if we ever see a count reach 5, the whole config is invalid (score -1 or 0).
            if line_counts[line_id] == 5:
                return -1

    four_count = 0
    for count in line_counts.values():
        if count == 4:
            four_count += 1
            
    return four_count

def construct():
    start_time = time.time()
    time_limit = 110  # Leave some buffer
    
    random.seed(42)
    
    # Population size
    pop_size = 50
    max_generations = 500
    
    # Initial population: Mix of random and structured
    population = []
    
    # 1. Random points
    for _ in range(pop_size // 2):
        pts = []
        used = set()
        while len(pts) < 16:
            x = random.randint(-1000, 1000)
            y = random.randint(-1000, 1000)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        population.append(pts)
        
    # 2. Grid-based perturbations (4x4 grid + noise)
    for _ in range(pop_size // 4):
        base = [[x, y] for x in range(4) for y in range(4)]
        # Scale up to use space
        scale = random.randint(10, 50)
        pts = [[x * scale + random.randint(-5, 5), y * scale + random.randint(-5, 5)] for x, y in base]
        # Ensure distinct
        valid = True
        seen = set()
        for p in pts:
            if tuple(p) in seen:
                valid = False
                break
            seen.add(tuple(p))
        if valid:
            population.append(pts)
            
    # 3. Spiral / Parametric curves
    for _ in range(pop_size // 4):
        pts = []
        seen = set()
        for k in range(16):
            # Try a few random parametric curves
            t = k * random.uniform(0.5, 1.5)
            # Lissajous-like
            x = int(1000 * math.sin(random.uniform(1, 3) * t + random.uniform(0, 6)))
            y = int(1000 * math.sin(random.uniform(1, 3) * t + random.uniform(0, 6)))
            if (x, y) not in seen and -1000 <= x <= 1000 and -1000 <= y <= 1000:
                seen.add((x, y))
                pts.append([x, y])
        if len(pts) == 16:
             population.append(pts)
        else:
             # Fill with random
             while len(pts) < 16:
                 x = random.randint(-1000, 1000)
                 y = random.randint(-1000, 1000)
                 if (x, y) not in seen:
                     seen.add((x, y))
                     pts.append([x, y])
             population.append(pts)

    best_score = -1
    best_points = None
    
    # Evaluate initial population
    for i, pts in enumerate(population):
        score = count_four_lines(pts)
        if score > best_score:
            best_score = score
            best_points = [list(p) for p in pts]
            
    # GA Loop
    gen = 0
    while time.time() - start_time < time_limit:
        gen += 1
        
        # Evaluate current population scores
        scores = []
        for pts in population:
            scores.append(count_four_lines(pts))
            
        # Find current best
        max_score = max(scores)
        if max_score > best_score:
            best_score = max_score
            best_idx = scores.index(max_score)
            best_points = [list(p) for p in population[best_idx]]
            
        # Selection: Tournament selection
        def select_tournament():
            # Pick 3 random individuals, return the best
            candidates = [random.randint(0, len(population)-1) for _ in range(3)]
            best_candidate = max(candidates, key=lambda i: scores[i])
            return [list(p) for p in population[best_candidate]]
            
        new_population = []
        
        # Elitism: Keep the best individual
        best_idx = scores.index(max_score)
        new_population.append([list(p) for p in population[best_idx]])
        
        while len(new_population) < pop_size:
            # Crossover
            parent1 = select_tournament()
            parent2 = select_tournament()
            
            # Uniform crossover: For each point index, pick from parent1 or parent2
            child = []
            for i in range(16):
                if random.random() < 0.5:
                    child.append(list(parent1[i]))
                else:
                    child.append(list(parent2[i]))
                    
            # Mutation
            # Move 1-3 points to new random locations or nearby
            num_mutations = random.randint(1, 3)
            for _ in range(num_mutations):
                idx = random.randint(0, 15)
                # Decide mutation type: small step or jump
                if random.random() < 0.5:
                    # Small step
                    dx = random.randint(-10, 10)
                    dy = random.randint(-10, 10)
                    nx = child[idx][0] + dx
                    ny = child[idx][1] + dy
                    if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                        # Check collision
                        valid = True
                        for k in range(16):
                            if k != idx and child[k][0] == nx and child[k][1] == ny:
                                valid = False
                                break
                        if valid:
                            child[idx] = [nx, ny]
                else:
                    # Jump
                    nx = random.randint(-1000, 1000)
                    ny = random.randint(-1000, 1000)
                    valid = True
                    for k in range(16):
                        if k != idx and child[k][0] == nx and child[k][1] == ny:
                            valid = False
                            break
                    if valid:
                        child[idx] = [nx, ny]
                        
            new_population.append(child)
            
        population = new_population
        
    return best_points
