import time
import random
from math import gcd

def evaluate_fast(points):
    n = len(points)
    if n != 16:
        return -1
        
    # Check distinctness and bounds
    seen = set()
    for x, y in points:
        if (x, y) in seen:
            return -1
        seen.add((x, y))
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
            
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
                
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    score = 0
    for count in line_counts.values():
        if count >= 10: # Corresponds to 5 points (C(5,2)=10)
            return -1
        if count == 6: # Corresponds to 4 points (C(4,2)=6)
            score += 1
            
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = evaluate_fast(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Hill Climbing
    while time.time() < deadline:
        improved = False
        
        # Try moving each point to nearby locations
        # To avoid bias, shuffle point order
        indices = list(range(16))
        random.shuffle(indices)
        
        for i in indices:
            # Try a few random nearby moves
            # Adaptive step size? Start small, increase if stuck.
            # Let's try offsets in [-10, 10]
            for _ in range(20):
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                if dx == 0 and dy == 0:
                    continue
                    
                new_x = current_points[i][0] + dx
                new_y = current_points[i][1] + dy
                
                if abs(new_x) > 1000 or abs(y) > 1000:
                    continue
                
                new_points = [p[:] for p in current_points]
                new_points[i] = [new_x, new_y]
                
                # Quick distinctness check
                if any(new_points[i] == p for j, p in enumerate(new_points) if j != i):
                    continue
                
                score = evaluate_fast(new_points)
                
                if score > current_score:
                    current_points = new_points
                    current_score = score
                    improved = True
                    if score > best_score:
                        best_points = [p[:] for p in current_points]
                        best_score = score
                        if best_score == 15:
                            return best_points
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
        if not improved:
            # If stuck, try a random restart from best or a larger jump
            # Let's try a random perturbation of one point with larger range
            i = random.randint(0, 15)
            for _ in range(5):
                new_x = random.randint(-1000, 1000)
                new_y = random.randint(-1000, 1000)
                
                new_points = [p[:] for p in current_points]
                new_points[i] = [new_x, new_y]
                
                if any(new_points[i] == p for j, p in enumerate(new_points) if j != i):
                    continue
                
                score = evaluate_fast(new_points)
                if score > current_score:
                    current_points = new_points
                    current_score = score
                    improved = True
                    if score > best_score:
                        best_points = [p[:] for p in current_points]
                        best_score = score
                        if best_score == 15:
                            return best_points
                    break

    return best_points
