import random
import math
import time

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def count_lines(points):
    n = len(points)
    count_4 = 0
    count_5plus = 0
    
    lines = set()
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            lines.add((A, B, C))
            
    for line in lines:
        A, B, C = line
        cnt = 0
        for p in points:
            x, y = p
            if A*x + B*y + C == 0:
                cnt += 1
        if cnt == 4:
            count_4 += 1
        elif cnt >= 5:
            count_5plus += 1
            
    return count_4, count_5plus

def score(points):
    c4, c5 = count_lines(points)
    return c4 - 1000 * c5

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = score(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    T = 10.0
    T_min = 0.1
    alpha = 0.9995
    
    while time.time() < deadline:
        idx = random.randint(0, 15)
        x, y = current_points[idx]
        
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        if dx == 0 and dy == 0:
            continue
            
        new_x = x + dx
        new_y = y + dy
        
        if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
            continue
            
        new_point = [new_x, new_y]
        is_distinct = True
        for i, p in enumerate(current_points):
            if i == idx:
                continue
            if p[0] == new_x and p[1] == new_y:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        old_point = current_points[idx][:]
        current_points[idx] = new_point
        
        new_score = score(current_points)
        
        delta = new_score - current_score
        
        if delta > 0 or (T > T_min and random.random() < math.exp(delta / T)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[idx] = old_point
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_points
