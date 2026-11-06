import random
import math

N = 64

def get_forbidden_points(a, b):
    x1, y1 = a
    x2, y2 = b
    dx = x2 - x1
    dy = y2 - y1
    d_sq = dx * dx + dy * dy
    
    forbidden = set()
    
    if dx == 0 and dy == 0:
        return forbidden
        
    C = (x2 * x2 - x1 * x1) + (y2 * y2 - y1 * y1)
    
    if dx == 0:
        if C % (2 * dy) == 0:
            y = C // (2 * dy)
            if 0 <= y < N:
                for x in range(N):
                    forbidden.add((x, y))
    elif dy == 0:
        if C % (2 * dx) == 0:
            x = C // (2 * dx)
            if 0 <= x < N:
                for y in range(N):
                    forbidden.add((x, y))
    else:
        for x in range(N):
            rem = C - 2 * dx * x
            if rem % (2 * dy) == 0:
                y = rem // (2 * dy)
                if 0 <= y < N:
                    forbidden.add((x, y))

    max_dx = int(math.isqrt(d_sq))
    
    for x in range(max(0, x1 - max_dx), min(N, x1 + max_dx + 1)):
        dy_sq = d_sq - (x - x1) * (x - x1)
        if dy_sq < 0:
            continue
        dy = int(math.isqrt(dy_sq))
        if dy * dy == dy_sq:
            y1_val = y1 - dy
            y2_val = y1 + dy
            if 0 <= y1_val < N:
                forbidden.add((x, y1_val))
            if 0 <= y2_val < N:
                forbidden.add((x, y2_val))

    for x in range(max(0, x2 - max_dx), min(N, x2 + max_dx + 1)):
        dy_sq = d_sq - (x - x2) * (x - x2)
        if dy_sq < 0:
            continue
        dy = int(math.isqrt(dy_sq))
        if dy * dy == dy_sq:
            y1_val = y2 - dy
            y2_val = y2 + dy
            if 0 <= y1_val < N:
                forbidden.add((x, y1_val))
            if 0 <= y2_val < N:
                forbidden.add((x, y2_val))

    return forbidden

def construct():
    best_points = []
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Try multiple random orders and some structured orders
    num_trials = 500
    
    for trial in range(num_trials):
        if trial < 10:
            # Some structured orders
            if trial == 0:
                order = all_points
            elif trial == 1:
                order = sorted(all_points, key=lambda p: (p[0], p[1]))
            elif trial == 2:
                order = sorted(all_points, key=lambda p: (p[1], p[0]))
            elif trial == 3:
                order = sorted(all_points, key=lambda p: p[0] + p[1])
            elif trial == 4:
                order = sorted(all_points, key=lambda p: abs(p[0] - N//2) + abs(p[1] - N//2))
            elif trial == 5:
                order = sorted(all_points, key=lambda p: abs(p[0] - N//2) - abs(p[1] - N//2))
            elif trial == 6:
                order = sorted(all_points, key=lambda p: (p[0] - N//2)**2 + (p[1] - N//2)**2)
            elif trial == 7:
                order = sorted(all_points, key=lambda p: (p[0] + p[1]) % N)
            elif trial == 8:
                order = sorted(all_points, key=lambda p: (p[0] - p[1]) % N)
            else:
                order = sorted(all_points, key=lambda p: random.random())
        else:
            order = all_points[:]
            random.shuffle(order)
            
        points = []
        forbidden = set()
        
        for p in order:
            if p not in forbidden:
                points.append(p)
                # Update forbidden set
                for q in points[:-1]:
                    forbidden.update(get_forbidden_points(p, q))
                    
        if len(points) > len(best_points):
            best_points = points
            
    return best_points
