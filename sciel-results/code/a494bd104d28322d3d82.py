import time
import random
import math
import sys

def _score(points):
    """
    Calculate the number of lines containing exactly 4 points.
    Also returns 0 if any line has 5 or more points (invalid).
    """
    n = len(points)
    # Map from line key to count of points
    line_counts = {}
    
    # Iterate over all pairs to define lines
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line equation: ax + by + c = 0
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            # Normalize the line equation
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g != 0:
                a //= g
                b //= g
                c //= g
            else:
                continue
                
            # Ensure consistent sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    four_lines = 0
    for count in line_counts.values():
        # Find k such that k(k-1)/2 == count
        val = 1 + 8 * count
        root = int(val ** 0.5)
        if root * root != val:
            continue
        k = (1 + root) // 2
        if k == 4:
            four_lines += 1
        elif k >= 5:
            return 0 # Invalid configuration
            
    return four_lines

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin before 150s kill
    
    n = 16
    
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    current_score = _score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Simulated annealing parameters
    temperature = 50.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # Try to improve via local search
        improved = False
        order = list(range(n))
        random.shuffle(order)
        
        for i in order:
            old_x, old_y = points[i]
            best_candidate = None
            best_candidate_score = current_score
            
            # Generate candidates: small moves + some medium moves
            candidates = set()
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = old_x + dx, old_y + dy
                    if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                        candidates.add((nx, ny))
                        
            # Occasionally add random far moves to escape local optima
            if random.random() < 0.15:
                for _ in range(10):
                    nx = random.randint(-100, 100)
                    ny = random.randint(-100, 100)
                    candidates.add((nx, ny))

            # Evaluate candidates
            for cx, cy in candidates:
                # Check distinctness
                valid = True
                for j in range(n):
                    if i == j:
                        continue
                    if points[j][0] == cx and points[j][1] == cy:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Temporarily move
                points[i] = [cx, cy]
                score = _score(points)
                
                # Accept if better, or with probability based on temperature (simulated annealing)
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = (cx, cy)
                elif best_candidate is None and score >= current_score:
                    # Accept non-worsening moves
                    best_candidate = (cx, cy)
                    best_candidate_score = score
                elif best_candidate is None and random.random() < math.exp((score - current_score) / max(temperature, 0.1)):
                    # Accept worsening moves with probability based on temperature
                    best_candidate = (cx, cy)
                    best_candidate_score = score

            # Restore original position
            points[i] = [old_x, old_y]
            
            # Apply best candidate if found
            if best_candidate is not None:
                points[i] = [best_candidate[0], best_candidate[1]]
                current_score = best_candidate_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
                    
        # Cool down
        temperature *= cooling_rate
        
        if not improved:
            # Local optimum reached. Try a random restart/kick
            # Move 2-3 random points to random locations
            for _ in range(random.randint(2, 3)):
                i = random.randint(0, n - 1)
                old_x, old_y = points[i]
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                
                # Ensure distinct
                valid = True
                for j in range(n):
                    if i == j: continue
                    if points[j][0] == new_x and points[j][1] == new_y:
                        valid = False
                        break
                
                if valid:
                    points[i] = [new_x, new_y]
            
            current_score = _score(points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
            else:
                # Revert to best if kick didn't help, but keep some randomness
                # by perturbing best points slightly
                points = [p[:] for p in best_points]
                current_score = best_score
                i = random.randint(0, n - 1)
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                nx, ny = points[i][0] + dx, points[i][1] + dy
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    valid = True
                    for j in range(n):
                        if i == j: continue
                        if points[j][0] == nx and points[j][1] == ny:
                            valid = False
                            break
                    if valid:
                        points[i] = [nx, ny]
                        current_score = _score(points)
                        if current_score > best_score:
                            best_score = current_score
                            best_points = [p[:] for p in points]

    return best_points
