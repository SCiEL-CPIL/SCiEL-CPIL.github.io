import time
import random
import math
from collections import defaultdict

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _score(points):
    n = len(points)
    if n != 16:
        return 0, False
    
    pts_set = set()
    for p in points:
        if p in pts_set:
            return 0, False
        pts_set.add(p)
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return 0, False

    line_counts = defaultdict(int)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            A = dy
            B = -dx
            C = A*x1 + B*y1
            g2 = _gcd(_gcd(abs(A), abs(B)), abs(C))
            if g2 > 0:
                A //= g2
                B //= g2
                C //= g2
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_counts[key] += 1
            
    exact4 = 0
    for key, count in line_counts.items():
        k = 2
        found = False
        while k <= 16:
            if k*(k-1)//2 == count:
                found = True
                break
            k += 1
        if not found:
            continue
        if k >= 5:
            return 0, False
        if k == 4:
            exact4 += 1
            
    return exact4, True

def _sa_search(points, time_limit):
    start_time = time.time()
    current_score, is_valid = _score(points)
    if not is_valid:
        return points, current_score
    
    best_score = current_score
    best_points = [p[:] for p in points]
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() - start_time < time_limit and T > T_min:
        # Generate a move
        i = random.randint(0, 15)
        
        # Strategy: 50% random, 50% line-based
        if random.random() < 0.5:
            x = random.randint(-1000, 1000)
            y = random.randint(-1000, 1000)
        else:
            # Line-based
            # Pick a random pair of other points
            others = [j for j in range(16) if j != i]
            j1 = random.choice(others)
            j2 = random.choice([j for j in others if j != j1])
            
            p1 = points[j1]
            p2 = points[j2]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            
            # Pick a random t
            t = random.randint(-10, 10)
            x = p1[0] + t * dx
            y = p1[1] + t * dy
            
            if x < -1000 or x > 1000 or y < -1000 or y > 1000:
                # Fallback to random
                x = random.randint(-1000, 1000)
                y = random.randint(-1000, 1000)
                
        new_pt = [x, y]
        
        # Check distinctness
        if any(p == new_pt for j, p in enumerate(points) if j != i):
            continue
            
        new_points = [p[:] for p in points]
        new_points[i] = new_pt
        
        new_score, is_valid = _score(new_points)
        
        if not is_valid:
            # Penalize invalid moves heavily
            delta = -100
        else:
            delta = new_score - current_score
            
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
                
        T *= cooling
        
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_score = -1
    best_points = None
    
    # Seed 1: 4x4 Grid
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    if time.time() < deadline:
        pts, score = _sa_search(initial_points, deadline - time.time() - 5)
        if score > best_score:
            best_score = score
            best_points = pts
            
    # Random Restarts
    random.seed(42)
    for restart in range(100):
        if time.time() > deadline - 10:
            break
            
        random_points = []
        for _ in range(16):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in set(tuple(p) for p in random_points):
                    random_points.append([x, y])
                    break
        
        score, valid = _score(random_points)
        if not valid:
            continue
            
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in random_points]
            
        if time.time() < deadline - 5:
            pts, score = _sa_search(random_points, deadline - time.time() - 5)
            if score > best_score:
                best_score = score
                best_points = pts
                
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
