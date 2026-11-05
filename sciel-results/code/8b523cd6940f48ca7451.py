import math
import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0
    
    def evaluate(points):
        """Fast evaluation using numpy. Returns (valid, score)."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        
        # Check bounds and distinctness
        if np.any(pts[:, 0] < -1000) or np.any(pts[:, 0] > 1000):
            return False, 0
        if np.any(pts[:, 1] < -1000) or np.any(pts[:, 1] > 1000):
            return False, 0
        if len(set(map(tuple, pts))) != n:
            return False, 0
        
        # Count points on each line
        line_counts = {}
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(int(dx)), abs(int(dy)))
                if g == 0:
                    continue
                A = int(dy) // g
                B = -int(dx) // g
                C = -(A * x1 + B * y1)
                
                # Normalize
                g2 = math.gcd(abs(A), abs(B))
                if g2 > 0:
                    A //= g2
                    B //= g2
                    C //= g2
                
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                    
                key = (A, B, C)
                if key not in line_counts:
                    line_counts[key] = set()
                line_counts[key].add(i)
                line_counts[key].add(j)
        
        score = 0
        for key, idxs in line_counts.items():
            c = len(idxs)
            if c >= 5:
                return False, 0
            if c == 4:
                score += 1
        
        return True, score

    def get_three_point_lines(points):
        """Get all lines with exactly 3 points."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        line_pts = {}
        
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(int(dx)), abs(int(dy)))
                if g == 0:
                    continue
                A = int(dy) // g
                B = -int(dx) // g
                C = -(A * x1 + B * y1)
                
                g2 = math.gcd(abs(A), abs(B))
                if g2 > 0:
                    A //= g2
                    B //= g2
                    C //= g2
                
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                    
                key = (A, B, C)
                if key not in line_pts:
                    line_pts[key] = set()
                line_pts[key].add(i)
                line_pts[key].add(j)
        
        three_lines = []
        for key, idxs in line_pts.items():
            if len(idxs) == 3:
                three_lines.append((key, list(idxs)))
        
        return three_lines

    def generate_candidates_for_line(points, line_key, occupied_set):
        """Generate candidate points on a specific line to extend it to 4 points."""
        A, B, C = line_key
        candidates = []
        
        # Find two points on the line to determine direction
        pts = [p for p in points if A * p[0] + B * p[1] + C == 0]
        if len(pts) < 2:
            return candidates
        
        p1, p2 = pts[0], pts[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            return candidates
        dx //= g
        dy //= g
        
        # Try various t values to find new points on the line
        for t in range(-20, 21):
            nx = p1[0] + t * dx
            ny = p1[1] + t * dy
            if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                if (nx, ny) not in occupied_set:
                    candidates.append([nx, ny])
        
        return candidates

    def simulated_annealing(initial_points, max_time):
        """Simulated annealing to maximize 4-point lines."""
        best_score = -1
        best_points = None
        current_points = [p[:] for p in initial_points]
        
        current_valid, current_score = evaluate(current_points)
        if current_valid:
            best_score = current_score
            best_points = [p[:] for p in current_points]
        
        temperature = 50.0
        cooling_rate = 0.9995
        min_temperature = 0.5
        no_improve = 0
        
        while time.time() < max_time:
            if temperature < min_temperature:
                break
            
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.3:
                # Move one point to a random position
                idx = random.randint(0, 15)
                old_x, old_y = current_points[idx]
                new_x = random.randint(-1000, 1000)
                new_y = random.randint(-1000, 1000)
                current_points[idx] = [new_x, new_y]
                
                new_valid, new_score = evaluate(current_points)
                
                if not new_valid:
                    current_points[idx] = [old_x, old_y]
                    no_improve += 1
                    continue
                
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    current_score = new_score
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                else:
                    current_points[idx] = [old_x, old_y]
                    no_improve += 1
                    
            elif move_type < 0.6:
                # Try to extend a 3-point line
                three_lines = get_three_point_lines(current_points)
                if three_lines:
                    line_key, line_pts = random.choice(three_lines)
                    occupied = set(map(tuple, current_points))
                    candidates = generate_candidates_for_line(current_points, line_key, occupied)
                    if candidates:
                        new_point = random.choice(candidates)
                        # Find a point to move that is NOT on this line
                        movable = [i for i in range(16) if i not in line_pts]
                        if movable:
                            idx = random.choice(movable)
                            old_x, old_y = current_points[idx]
                            current_points[idx] = new_point
                            
                            new_valid, new_score = evaluate(current_points)
                            
                            if not new_valid:
                                current_points[idx] = [old_x, old_y]
                                no_improve += 1
                                continue
                            
                            delta = new_score - current_score
                            if delta > 0 or random.random() < math.exp(delta / temperature):
                                current_score = new_score
                                no_improve = 0
                                if current_score > best_score:
                                    best_score = current_score
                                    best_points = [p[:] for p in current_points]
                            else:
                                current_points[idx] = [old_x, old_y]
                                no_improve += 1
            
            else:
                # Swap two points
                idx1, idx2 = random.sample(range(16), 2)
                p1 = current_points[idx1][:]
                p2 = current_points[idx2][:]
                current_points[idx1] = p2
                current_points[idx2] = p1
                
                new_valid, new_score = evaluate(current_points)
                
                if not new_valid:
                    current_points[idx1] = p1
                    current_points[idx2] = p2
                    no_improve += 1
                    continue
                
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    current_score = new_score
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                else:
                    current_points[idx1] = p1
                    current_points[idx2] = p2
                    no_improve += 1
            
            temperature *= cooling_rate
            
            # Restart if stuck
            if no_improve > 200:
                no_improve = 0
                temperature = max(temperature, 20.0)
                # Random kick
                for _ in range(random.randint(1, 3)):
                    k = random.randint(0, 15)
                    current_points[k] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
                current_valid, current_score = evaluate(current_points)
                if not current_valid:
                    current_points = [p[:] for p in best_points] if best_points else [[x, y] for x in range(4) for y in range(4)]
                    current_valid, current_score = evaluate(current_points)
        
        return best_points, best_score

    def final_polish(points, max_time):
        """Final polish with targeted moves."""
        best_points = [p[:] for p in points]
        best_score, _ = evaluate(points)
        current_points = [p[:] for p in points]
        
        while time.time() < max_time:
            improved = False
            three_lines = get_three_point_lines(current_points)
            
            for line_key, line_pts in three_lines:
                occupied = set(map(tuple, current_points))
                candidates = generate_candidates_for_line(current_points, line_key, occupied)
                
                for new_point in candidates:
                    for idx in range(16):
                        if idx in line_pts:
                            continue
                        old_x, old_y = current_points[idx]
                        current_points[idx] = new_point
                        
                        new_valid, new_score = evaluate(current_points)
                        
                        if new_valid and new_score > best_score:
                            best_score = new_score
                            best_points = [p[:] for p in current_points]
                            improved = True
                        else:
                            current_points[idx] = [old_x, old_y]
                        
                        if time.time() > max_time:
                            return best_points, best_score
            
            if not improved:
                # Try random moves
                for _ in range(10):
                    if time.time() > max_time:
                        return best_points, best_score
                    idx = random.randint(0, 15)
                    old_x, old_y = current_points[idx]
                    new_x = random.randint(-1000, 1000)
                    new_y = random.randint(-1000, 1000)
                    current_points[idx] = [new_x, new_y]
                    
                    new_valid, new_score = evaluate(current_points)
                    
                    if new_valid and new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                    else:
                        current_points[idx] = [old_x, old_y]
        
        return best_points, best_score

    # Initial configurations to try
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Various structured configurations
    for offset in range(-2, 3):
        config = [[x + offset, y + offset] for x in range(4) for y in range(4)]
        initial_configs.append(config)
    
    # Random initial configurations
    for _ in range(5):
        config = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)]
        initial_configs.append(config)
    
    best_overall = None
    best_overall_score = -1
    
    remaining_time = deadline - time.time()
    time_per_config = max(1.0, remaining_time / len(initial_configs))
    
    for config in initial_configs:
        if time.time() > deadline - 5.0:
            break
        
        # Evaluate initial config
        valid, score = evaluate(config)
        if not valid:
            continue
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall = [p[:] for p in config]
        
        # Run simulated annealing
        sa_points, sa_score = simulated_annealing(config, deadline - 5.0 - time.time() * 0.3)
        
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall = sa_points
    
    # Final polish
    if best_overall and time.time() < deadline - 2.0:
        polished, polished_score = final_polish(best_overall, deadline - 1.0)
        if polished_score > best_overall_score:
            best_overall = polished
            best_overall_score = polished_score
    
    # Ensure we have a valid result
    if best_overall is None:
        best_overall = [[x, y] for x in range(4) for y in range(4)]
    
    # Final validation
    valid, _ = evaluate(best_overall)
    if not valid:
        best_overall = [[x, y] for x in range(4) for y in range(4)]
    
    return best_overall
