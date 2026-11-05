import math
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0
    
    best_score = -1
    best_points = None
    
    def evaluate(points):
        """Fast evaluation using numpy for line counting."""
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        
        # Check bounds and duplicates
        if np.any(np.abs(pts) > 1000):
            return False, 0, []
        if len(set(map(tuple, pts))) != n:
            return False, 0, []
        
        lines = {}
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                A = dy // g
                B = -(dx // g)
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
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
                
        score = 0
        lines_3 = []
        for key, pts_set in lines.items():
            c = len(pts_set)
            if c >= 5:
                return False, 0, []
            if c == 4:
                score += 1
            elif c == 3:
                lines_3.append((key, pts_set))
                
        return True, score, lines_3

    def get_smart_candidates(points, lines_3, occupied_set):
        """Generate candidates by extending 3-point lines and finding good positions."""
        candidates = []
        
        for key, pts_set in lines_3:
            A, B, C = key
            # Get the two points to parameterize the line
            pts_list = list(pts_set)
            p1 = points[pts_list[0]]
            p2 = points[pts_list[1]]
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            
            # Try extending in both directions
            for t in range(-8, 12):
                if t in (0, 1):  # Skip existing points
                    continue
                nx = x1 + t * dx
                ny = y1 + t * dy
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    if (nx, ny) not in occupied_set:
                        candidates.append([nx, ny])
        
        # Also add some random candidates for diversity
        for _ in range(5):
            rx = random.randint(-1000, 1000)
            ry = random.randint(-1000, 1000)
            if (rx, ry) not in occupied_set:
                candidates.append([rx, ry])
        
        return candidates

    # Try multiple initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Offset grids
    for ox in [-2, -1, 0, 1, 2]:
        for oy in [-2, -1, 0, 1, 2]:
            initial_configs.append([[x + ox, y + oy] for x in range(4) for y in range(4)])
    
    # Random configurations
    for _ in range(10):
        initial_configs.append([[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(16)])
    
    # Evaluate all initial configs and pick the best
    for init_pts in initial_configs:
        if time.time() > deadline:
            break
        valid, score, _ = evaluate(init_pts)
        if valid and score > best_score:
            best_score = score
            best_points = [p[:] for p in init_pts]
    
    if best_points is None:
        # Fallback
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = 0
    
    # Local search from best initial configuration
    current_points = [p[:] for p in best_points]
    current_valid, current_score, current_lines_3 = evaluate(current_points)
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        if not current_valid:
            # Random restart
            current_points = [[random.randint(-200, 200), random.randint(-200, 200)] for _ in range(16)]
            current_valid, current_score, current_lines_3 = evaluate(current_points)
            no_improve_count = 0
            continue
        
        occupied_set = set((p[0], p[1]) for p in current_points)
        candidates = get_smart_candidates(current_points, current_lines_3, occupied_set)
        
        if not candidates:
            # Fallback to random move
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            current_points[idx] = [random.randint(-200, 200), random.randint(-200, 200)]
            valid, score, lines_3 = evaluate(current_points)
            if valid and score > current_score:
                current_score = score
                current_lines_3 = lines_3
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[idx] = [old_x, old_y]
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random kick
                for _ in range(3):
                    k = random.randint(0, 15)
                    current_points[k] = [random.randint(-200, 200), random.randint(-200, 200)]
                current_valid, current_score, current_lines_3 = evaluate(current_points)
                no_improve_count = 0
            continue
        
        # Try multiple candidates in a batch
        tried = False
        for new_point in random.sample(candidates, min(len(candidates), 10)):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            if old_x == new_point[0] and old_y == new_point[1]:
                continue
                
            current_points[idx] = new_point
            valid, score, lines_3 = evaluate(current_points)
            
            if valid and score > current_score:
                current_score = score
                current_lines_3 = lines_3
                no_improve_count = 0
                tried = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[idx] = [old_x, old_y]
        
        if not tried:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random kick
                for _ in range(3):
                    k = random.randint(0, 15)
                    current_points[k] = [random.randint(-200, 200), random.randint(-200, 200)]
                current_valid, current_score, current_lines_3 = evaluate(current_points)
                no_improve_count = 0
    
    # Final polish: try more iterations on the best configuration
    if best_points and time.time() < deadline - 10:
        current_points = [p[:] for p in best_points]
        current_valid, current_score, current_lines_3 = evaluate(current_points)
        
        polish_deadline = min(deadline, time.time() + 10)
        polish_no_improve = 0
        
        while time.time() < polish_deadline:
            if not current_valid:
                break
                
            occupied_set = set((p[0], p[1]) for p in current_points)
            candidates = get_smart_candidates(current_points, current_lines_3, occupied_set)
            
            if not candidates:
                break
                
            improved = False
            for new_point in random.sample(candidates, min(len(candidates), 20)):
                if time.time() > polish_deadline:
                    break
                    
                idx = random.randint(0, 15)
                old_x, old_y = current_points[idx]
                
                if old_x == new_point[0] and old_y == new_point[1]:
                    continue
                    
                current_points[idx] = new_point
                valid, score, lines_3 = evaluate(current_points)
                
                if valid and score > current_score:
                    current_score = score
                    current_lines_3 = lines_3
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                else:
                    current_points[idx] = [old_x, old_y]
            
            if not improved:
                polish_no_improve += 1
                if polish_no_improve > 30:
                    break
    
    return best_points
