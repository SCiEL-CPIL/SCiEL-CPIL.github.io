import time
import random
from math import gcd

def count_4_lines(points):
    n = len(points)
    if n < 4:
        return 0, True
    
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
                
            g = gcd(abs(dx), abs(dy))
            dx /= g
            dy /= g
            
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
                
            c = dx * y1 - dy * x1
            key = (dy, -dx, c)
            
            if key not in line_counts:
                line_counts[key] = 1
            line_counts[key] += 1
            
            if line_counts[key] >= 5:
                return 0, False

    count_4 = 0
    for cnt in line_counts.values():
        if cnt == 4:
            count_4 += 1
             
    return count_4, True

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = count_4_lines(best_points)
    
    # Local Search with 1-point and 2-point moves
    current_points = [pt[:] for pt in best_points]
    current_score = best_score
    
    # Helper to check validity and score
    def evaluate(pts):
        return count_4_lines(pts)
    
    # Main loop
    while time.time() < deadline:
        improved = False
        
        # 1-point moves
        for i in range(16):
            if time.time() > deadline: break
            ox, oy = current_points[i]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0: continue
                    nx, ny = ox + dx, oy + dy
                    if nx < -1000 or nx > 1000 or ny < -1000 or ny > 1000: continue
                    if [nx, ny] in current_points: continue
                    
                    old = current_points[i]
                    current_points[i] = [nx, ny]
                    ns, valid = evaluate(current_points)
                    if valid and ns > current_score:
                        current_score = ns
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_points = [pt[:] for pt in current_points]
                    else:
                        current_points[i] = old
                        
        # 2-point moves (large jumps)
        if not improved:
            for i in range(16):
                if time.time() > deadline: break
                for j in range(i+1, 16):
                    if time.time() > deadline: break
                    ox1, oy1 = current_points[i]
                    ox2, oy2 = current_points[j]
                    
                    # Random large jumps
                    for _ in range(10):
                        nx1 = random.randint(-50, 50)
                        ny1 = random.randint(-50, 50)
                        nx2 = random.randint(-50, 50)
                        ny2 = random.randint(-50, 50)
                        
                        if [nx1, ny1] in current_points or [nx2, ny2] in current_points:
                            continue
                        if [nx1, ny1] == [nx2, ny2]:
                            continue
                        
                        old1 = current_points[i]
                        old2 = current_points[j]
                        current_points[i] = [nx1, ny1]
                        current_points[j] = [nx2, ny2]
                        
                        ns, valid = evaluate(current_points)
                        if valid and ns > current_score:
                            current_score = ns
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_points = [pt[:] for pt in current_points]
                        else:
                            current_points[i] = old1
                            current_points[j] = old2

    # Final verification
    final_score, valid = count_4_lines(best_points)
    if not valid:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

# EVOLVE-BLOCK-END
