import time
import random
import math
from collections import defaultdict

def _score_points(points):
    """
    Returns (num_exact_4_lines, has_5_collinear, three_point_lines).
    points: list of 16 [x, y]
    """
    n = len(points)
    lines = defaultdict(set)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction vector
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            # Ensure consistent direction (e.g., dx > 0 or dx == 0 and dy > 0)
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            # Line equation: A*x + B*y + C = 0
            # A = dy, B = -dx, C = -(A*x1 + B*y1)
            A = dy_n
            B = -dx_n
            C = -(A * x1 + B * y1)
            
            # Normalize (A, B, C) by gcd
            g2 = math.gcd(abs(A), abs(B))
            if g2 > 0:
                A //= g2
                B //= g2
                C //= g2
                
            # Final sign normalization
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            lines[key].add(i)
            lines[key].add(j)
            
    exact_4 = 0
    has_5 = False
    three_point_lines = []
    
    for key, pts in lines.items():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
        elif cnt == 3:
            three_point_lines.append((key, list(pts)))
            
    return exact_4, has_5, three_point_lines

def construct():
    deadline = time.time() + 115
    
    # Initial configuration: 4x4 grid scaled by 10
    base = [[10 * i, 10 * j] for i in range(4) for j in range(4)]
    
    best_points = [p[:] for p in base]
    best_score, _, _ = _score_points(base)
    
    rng = random.Random(42)
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.999
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        if restart == 0:
            current = [p[:] for p in base]
        else:
            # Start from best known or random perturbation of best
            current = [p[:] for p in best_points]
            for _ in range(rng.randint(10, 30)):
                idx = rng.randint(0, 15)
                current[idx][0] += rng.randint(-20, 20)
                current[idx][1] += rng.randint(-20, 20)
                if abs(current[idx][0]) > 1000 or abs(current[idx][1]) > 1000:
                    current[idx][0] = max(-1000, min(1000, current[idx][0]))
                    current[idx][1] = max(-1000, min(1000, current[idx][1]))
                # Ensure distinct
                for k in range(16):
                    if k != idx and current[k] == current[idx]:
                        current[idx][0] += 1
                        break
        
        s, h5, _ = _score_points(current)
        if h5:
            # Try to fix initial state
            for _ in range(200):
                # Find a 5-collinear line
                lines = defaultdict(set)
                for i in range(16):
                    x1, y1 = current[i]
                    for j in range(i + 1, 16):
                        x2, y2 = current[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        g = math.gcd(abs(dx), abs(dy))
                        if g == 0: continue
                        dx_n = dx // g
                        dy_n = dy // g
                        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                            dx_n = -dx_n
                            dy_n = -dy_n
                        A = dy_n
                        B = -dx_n
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
                        lines[key].add(i)
                
                found_line = None
                for key, pts in lines.items():
                    if len(pts) >= 5:
                        found_line = pts
                        break
                
                if not found_line:
                    break
                
                # Move one point from this line
                idx_to_move = rng.choice(list(found_line))
                old = current[idx_to_move][:]
                current[idx_to_move][0] += rng.randint(-5, 5)
                current[idx_to_move][1] += rng.randint(-5, 5)
                if abs(current[idx_to_move][0]) > 1000 or abs(current[idx_to_move][1]) > 1000:
                    current[idx_to_move] = old
                    continue
                if any(current[k] == current[idx_to_move] for k in range(16) if k != idx_to_move):
                    current[idx_to_move] = old
                    continue
                s, h5, _ = _score_points(current)
                if not h5:
                    break
            
            if h5:
                continue

        T = T_start
        current_score = s
        
        while time.time() < deadline - 2 and T > T_end:
            # Generate a move
            move_type = rng.random()
            
            if move_type < 0.7:
                # Move one point
                idx = rng.randint(0, 15)
                old = current[idx][:]
                step = rng.randint(1, 15)
                dx = rng.randint(-step, step)
                dy = rng.randint(-step, step)
                if dx == 0 and dy == 0:
                    dx = 1
                new_x = old[0] + dx
                new_y = old[1] + dy
                
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    T *= cooling_rate
                    continue
                    
                new_pt = [new_x, new_y]
                
                if any(current[k] == new_pt for k in range(16) if k != idx):
                    T *= cooling_rate
                    continue
                    
                current[idx] = new_pt
                s_new, h5_new, _ = _score_points(current)
                
                if h5_new:
                    current[idx] = old
                    T *= cooling_rate
                    continue
            else:
                # Swap two points
                i, j = rng.sample(range(16), 2)
                old_i = current[i][:]
                old_j = current[j][:]
                current[i], current[j] = current[j], current[i]
                
                s_new, h5_new, _ = _score_points(current)
                
                if h5_new:
                    current[i], current[j] = old_i, old_j
                    T *= cooling_rate
                    continue
            
            delta = s_new - current_score
            
            if delta > 0:
                # Better, accept
                current_score = s_new
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current]
            else:
                # Worse, accept with probability exp(delta / T)
                if rng.random() < math.exp(delta / T):
                    current_score = s_new
                else:
                    # Revert
                    if move_type < 0.7:
                        current[idx] = old
                    else:
                        current[i], current[j] = old_i, old_j
            
            T *= cooling_rate

    return best_points
