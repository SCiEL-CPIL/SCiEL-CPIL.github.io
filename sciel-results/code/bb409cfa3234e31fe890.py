import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def make_grid():
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        return pts
    
    def make_random():
        pts = []
        while len(pts) < 16:
            x = random.randint(-1000, 1000)
            y = random.randint(-1000, 1000)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        return pts

    def score(points):
        line_counts = {}
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                g = math.gcd(A, B)
                g = math.gcd(g, C)
                if g == 0:
                    continue
                A //= g
                B //= g
                C //= g
                
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                    
                key = (A, B, C)
                if key in line_counts:
                    line_counts[key] += 1
                else:
                    line_counts[key] = 1
        
        num_4 = 0
        num_5_plus = 0
        for count in line_counts.values():
            if count == 6: # 4 points -> 6 pairs
                num_4 += 1
            elif count >= 10: # 5 points -> 10 pairs
                num_5_plus += 1
                
        return num_4, num_5_plus

    def fitness(points):
        n4, n5 = score(points)
        return n4 - 10000 * n5

    best_points = make_grid()
    best_score = fitness(best_points)
    
    # Try a few random starts
    for _ in range(20):
        pts = make_random()
        s = fitness(pts)
        if s > best_score:
            best_score = s
            best_points = pts

    # Simulated Annealing
    current_points = [p[:] for p in best_points]
    current_score = best_score
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        new_points = [p[:] for p in current_points]
        idx = random.randint(0, 15)
        
        if random.random() < 0.8:
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
            new_points[idx][0] += dx
            new_points[idx][1] += dy
        else:
            new_points[idx][0] = random.randint(-1000, 1000)
            new_points[idx][1] = random.randint(-1000, 1000)
            
        if abs(new_points[idx][0]) > 1000 or abs(new_points[idx][1]) > 1000:
            continue
            
        if len(set(map(tuple, new_points))) < 16:
            continue
            
        new_score = fitness(new_points)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                
        T *= cooling
        
    return best_points
