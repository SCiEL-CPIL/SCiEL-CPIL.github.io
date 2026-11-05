import math
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    best_score = -1
    best_points = None
    
    def evaluate(points):
        """Fast evaluation: check validity and count 4-point lines."""
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
        """Generate candidates by extending 3-point lines."""
        candidates = []
        
        for key, pts_set in lines_3:
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
            
            for t in range(-6, 8):
                if t in (0, 1):
                    continue
                nx = x1 + t * dx
                ny = y1 + t * dy
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    if (nx, ny) not in occupied_set:
                        candidates.append([nx, ny])
        
        # Add random candidates for diversity
        for _ in range(3):
            rx = random.randint(-500, 500)
            ry = random.randint(-500, 500)
            if (rx, ry) not in occupied_set:
                candidates.append([rx, ry])
        
        return candidates
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 4x4 grid (gives 10 four-point lines)
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Slightly perturbed grids
    for ox in [-1, 0, 1]:
        for oy in [-1, 0, 1]:
            if ox == 0 and oy == 0:
                continue
            initial_configs.append([[x + ox, y + oy] for x in range(4) for y in range(4)])
    
    # Random configurations
    for _ in range(5):
        initial_configs.append([[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)])
    
    # Evaluate all initial configs and pick the best
    for init_pts in initial_configs:
        if time.time() > deadline:
            break
        valid, score, _ = evaluate(init_pts)
        if valid and score > best_score:
            best_score = score
            best_points = [p[:] for p in init_pts]
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = 0
    
    # Simulated annealing from best initial configuration
    current_points = [p[:] for p in best_points]
    current_valid, current_score, current_lines_3 = evaluate(current_points)
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline and T > T_min:
        if not current_valid:
            # Random restart
            current_points = [[random.randint(-300, 300), random.randint(-300, 300)] for _ in range(16)]
            current_valid, current_score, current_lines_3 = evaluate(current_points)
            no_improve_count = 0
            T = max(T, 2.0)
            continue
        
        occupied_set = set((p[0], p[1]) for p in current_points)
        
        # Generate candidates: mix of smart and random
        smart_candidates = get_smart_candidates(current_points, current_lines_3, occupied_set)
        
        # Also add some purely random candidates
        random_candidates = []
        for _ in range(5):
            rx = random.randint(-500, 500)
            ry = random.randint(-500, 500)
            if (rx, ry) not in occupied_set:
                random_candidates.append([rx, ry])
        
        all_candidates = smart_candidates + random_candidates
        
        if not all_candidates:
            # Fallback: random move
            idx = random.randint(0, 15)
            current_points[idx] = [random.randint(-300, 300), random.randint(-300, 300)]
            valid, score, lines_3 = evaluate(current_points)
            if valid:
                delta = score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = score
                    current_lines_3 = lines_3
                    current_valid = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Revert
                    pass
            else:
                no_improve_count += 1
            T *= cooling_rate
            continue
        
        # Try a few random candidates from the pool
        tried_count = min(len(all_candidates), 5)
        tried_indices = random.sample(range(len(all_candidates)), tried_count)
        
        improved_this_iter = False
        for ci in tried_indices:
            if time.time() > deadline:
                break
            
            new_point = all_candidates[ci]
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            if old_x == new_point[0] and old_y == new_point[1]:
                continue
                
            current_points[idx] = new_point
            valid, score, lines_3 = evaluate(current_points)
            
            if valid:
                delta = score - current_score
                # Accept if better, or with probability exp(delta/T) if worse
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = score
                    current_lines_3 = lines_3
                    current_valid = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                        no_improve_count = 0
                        improved_this_iter = True
                    else:
                        no_improve_count += 1
                else:
                    current_points[idx] = [old_x, old_y]
            else:
                current_points[idx] = [old_x, old_y]
                no_improve_count += 1
        
        if not improved_this_iter:
            no_improve_count += 1
        
        # Occasional random kick to escape deep local minima
        if no_improve_count > max_no_improve:
            for _ in range(3):
                k = random.randint(0, 15)
                current_points[k] = [random.randint(-300, 300), random.randint(-300, 300)]
            current_valid, current_score, current_lines_3 = evaluate(current_points)
            no_improve_count = 0
            T = max(T, 3.0)  # Reheat
        
        T *= cooling_rate
    
    # Final polish: greedy local search on best configuration
    if best_points and time.time() < deadline - 5:
        current_points = [p[:] for p in best_points]
        current_valid, current_score, current_lines_3 = evaluate(current_points)
        
        polish_deadline = min(deadline, time.time() + 5)
        polish_no_improve = 0
        
        while time.time() < polish_deadline:
            if not current_valid:
                break
                
            occupied_set = set((p[0], p[1]) for p in current_points)
            candidates = get_smart_candidates(current_points, current_lines_3, occupied_set)
            
            if not candidates:
                break
                
            improved = False
            for new_point in random.sample(candidates, min(len(candidates), 10)):
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
                if polish_no_improve > 20:
                    break
    
    return best_points
