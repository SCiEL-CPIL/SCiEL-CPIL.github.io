import itertools
import random
import time
import math

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for safety
    
    # Initial configuration: 5x6 grid
    # Centered to allow movement
    # x in [-2, 2], y in [-3, 2] -> 5 * 6 = 30 points
    def initial_grid():
        pts = []
        for x in range(-2, 3):
            for y in range(-3, 3):
                pts.append((x, y))
        return pts

    current_points = initial_grid()
    best_points = current_points[:]
    
    def calculate_score(points):
        counts = {}
        # Optimize: use local variables for speed
        p = points
        for i in range(29):
            xi, yi = p[i]
            for j in range(i + 1, 29):
                xj, yj = p[j]
                dx1 = xj - xi
                dy1 = yj - yi
                for k in range(j + 1, 30):
                    xk, yk = p[k]
                    dx2 = xk - xi
                    dy2 = yk - yi
                    det = abs(dx1 * dy2 - dy1 * dx2)
                    if det > 0:
                        counts[det] = counts.get(det, 0) + 1
        if not counts:
            return 0
        return max(counts.values())

    current_score = calculate_score(current_points)
    best_score = current_score
    
    # Search parameters
    stuck_counter = 0
    max_stuck = 2000
    temp = 1.0 # Not using SA temperature directly, using logic for moves
    
    while time.time() - start_time < time_limit:
        # Generate a move
        # 1. Move one point
        # 2. Swap two points
        move_type = random.random()
        
        new_points = current_points[:]
        
        if move_type < 0.7:
            # Move one point
            idx = random.randint(0, 29)
            x, y = new_points[idx]
            # Random step in [-3, 3]
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_x = x + dx
            new_y = y + dy
            
            # Bounds check
            if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                # Distinctness check
                # Check if (new_x, new_y) is already in new_points (excluding idx)
                valid = True
                for j in range(30):
                    if j != idx and new_points[j] == (new_x, new_y):
                        valid = False
                        break
                if valid:
                    new_points[idx] = (new_x, new_y)
                else:
                    continue
            else:
                continue
        else:
            # Swap two points
            i, j = random.sample(range(30), 2)
            new_points[i], new_points[j] = new_points[j], new_points[i]

        new_score = calculate_score(new_points)
        
        # Acceptance criteria
        if new_score > best_score:
            best_score = new_score
            best_points = new_points
            current_points = new_points
            current_score = new_score
            stuck_counter = 0
        elif new_score > current_score:
            # Accept improvement over current, even if not global best
            current_points = new_points
            current_score = new_score
            stuck_counter = 0
        else:
            # Accept with some probability to escape local minima?
            # Since the objective is discrete and we want MAX, 
            # accepting worse moves is risky but can help.
            # However, "max multiplicity" is a very sharp peak.
            # Let's try a simple hill climbing with random restarts if stuck.
            stuck_counter += 1
            
            if stuck_counter > max_stuck:
                # Restart from best with perturbation
                current_points = best_points[:]
                current_score = best_score
                stuck_counter = 0
                
                # Perturb best points
                for _ in range(5):
                    idx = random.randint(0, 29)
                    x, y = current_points[idx]
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    nx, ny = x + dx, y + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        # Check distinctness
                        valid = True
                        for j in range(30):
                            if j != idx and current_points[j] == (nx, ny):
                                valid = False
                                break
                        if valid:
                            current_points[idx] = (nx, ny)
                
                current_score = calculate_score(current_points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points[:]

    return best_points
