import random
import math
from collections import defaultdict
import time

def construct():
    # Initial configuration: 4x4 grid
    base = [-2, -1, 0, 1]
    points = [[x, y] for x in base for y in base]
    
    best_points = points[:]
    best_score = -1
    cur_score = -1
    cur_valid = False
    
    def evaluate(pts):
        line_counts = defaultdict(int)
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                else:
                    # This shouldn't happen for distinct points
                    continue
                
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                
                line_counts[(A, B, C)] += 1
        
        score = 0
        for key, pairs in line_counts.items():
            if pairs == 6:
                score += 1
            elif pairs > 6:
                return -1000, False # Penalty for 5+ collinear
        
        return score, True

    cur_score, cur_valid = evaluate(points)
    if cur_valid:
        best_score = cur_score
        best_points = points[:]
    else:
        best_score = 0
        best_points = points[:]

    start_time = time.time()
    timeout = 110
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    
    # We will do many iterations
    while time.time() - start_time < timeout:
        if T < 0.01:
            T = 10.0 # Reheat
            # Optional: Random restart if stuck
            if random.random() < 0.1:
                 points = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(16)]
                 # Ensure distinct
                 for i in range(16):
                     while any(points[i] == points[j] for j in range(16) if j != i):
                         points[i] = [random.randint(-10, 10), random.randint(-10, 10)]
                 cur_score, cur_valid = evaluate(points)
                 if cur_valid and cur_score > best_score:
                     best_score = cur_score
                     best_points = points[:]
        
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Random move
        # Mix of small and large steps
        if random.random() < 0.8:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        else:
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
            
        if dx == 0 and dy == 0:
            continue
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        is_dup = False
        for i in range(16):
            if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                is_dup = True
                break
        if is_dup:
            continue
            
        points[idx] = [new_x, new_y]
        
        new_score, new_valid = evaluate(points)
        
        if new_valid:
            delta = new_score - cur_score
            if delta > 0:
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = points[:]
            else:
                # Accept worse move with probability exp(delta/T)
                if random.random() < math.exp(delta / T):
                    cur_score = new_score
                else:
                    points[idx] = [old_x, old_y]
        else:
            # Invalid move (5+ collinear)
            # Reject with high probability, or accept with very low prob to escape?
            # Usually reject invalid states in this type of search unless we are stuck.
            points[idx] = [old_x, old_y]
            
        T *= cooling_rate

    return best_points
