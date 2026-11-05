import random
import time
import math

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
            
            # Normalize
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            elif A == 0 and B == 0:
                pass
            else:
                # GCD normalization to ensure unique keys for parallel lines with same ratio
                # Actually, for collinearity check, (A,B,C) and (kA,kB,kC) are same line.
                # Since we use integer points, A,B,C are integers.
                # We should divide by GCD to normalize.
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                    # Ensure consistent sign
                    if A < 0 or (A == 0 and B < 0):
                        A, B, C = -A, -B, -C

            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_score = float('inf')
    best_points = None
    
    def try_update(pts):
        nonlocal best_score, best_points
        if len(pts) != 12:
            return
        if len(set(map(tuple, pts))) != 12:
            return
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts):
            return
        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]

    # Seeds
    # 1. 3x4 Grid
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    try_update(grid_3x4)
    
    # 2. Star (3 concurrent lines)
    star = [
        (0,0),
        (1,0), (-1,0), (2,0),
        (1,1), (-1,-1), (2,2),
        (1,-1), (-1,1), (2,-2),
        (0,1), (0,-1) # Added 2 more to make 12?
    ]
    # Let's make a proper 12-point star
    # 3 lines through origin. 3 points on each ray?
    # L1: (1,0), (2,0), (3,0)
    # L2: (0,1), (0,2), (0,3)
    # L3: (1,1), (2,2), (3,3)
    # Origin (0,0)
    # That's 10 points.
    # Add (1,-1), (-1,1) ?
    star2 = [
        (0,0),
        (1,0), (2,0), (3,0),
        (0,1), (0,2), (0,3),
        (1,1), (2,2), (3,3),
        (1,-1), (-1,1)
    ]
    try_update(star2)

    if best_points is None:
        best_points = grid_3x4
        best_score = count_ordinary_lines(best_points)
        
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.99
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        idx = random.randint(0, 11)
        old_x, old_y = current_points[idx]
        
        # Generate new point
        # 80% chance: Random nearby
        # 20% chance: Intersection of two ordinary lines
        if random.random() < 0.8:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            # Find ordinary lines
            n = 12
            lines = {}
            for a in range(n):
                for b in range(a + 1, n):
                    x1, y1 = current_points[a]
                    x2, y2 = current_points[b]
                    A = y1 - y2
                    B = x2 - x1
                    C = x1 * y2 - x2 * y1
                    if A == 0 and B == 0: continue
                    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                    if g != 0:
                        A //= g; B //= g; C //= g
                        if A < 0 or (A == 0 and B < 0):
                            A, B, C = -A, -B, -C
                    key = (A, B, C)
                    if key not in lines:
                        lines[key] = set()
                    lines[key].add(a)
                    lines[key].add(b)
                    
            ordinary_lines = [k for k, v in lines.items() if len(v) == 2]
            if len(ordinary_lines) < 2:
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
            else:
                l1, l2 = random.sample(ordinary_lines, 2)
                A1, B1, C1 = l1
                A2, B2, C2 = l2
                det = A1 * B2 - A2 * B1
                if det == 0:
                    new_x = old_x + random.randint(-5, 5)
                    new_y = old_y + random.randint(-5, 5)
                else:
                    x_num = B1 * C2 - B2 * C1
                    y_num = A2 * C1 - A1 * C2
                    if x_num % det != 0 or y_num % det != 0:
                        new_x = old_x + random.randint(-5, 5)
                        new_y = old_y + random.randint(-5, 5)
                    else:
                        new_x = x_num // det
                        new_y = y_num // det
                        
        # Validate
        if (new_x, new_y) == (old_x, old_y):
            continue
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(current_points) if j != idx):
            continue
            
        current_points[idx] = [new_x, new_y]
        new_score = count_ordinary_lines(current_points)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [list(p) for p in current_points]
        else:
            current_points[idx] = [old_x, old_y]
            
        T *= cooling
        
    return best_points
