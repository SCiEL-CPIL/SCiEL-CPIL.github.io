import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return None
            
        g = math.gcd(abs(dx), abs(dy))
        dx /= g
        dy /= g
        
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
            
        A = dy
        B = -dx
        C = A * x1 + B * y1
        
        g_abc = math.gcd(abs(int(A)), abs(int(B)))
        if g_abc == 0:
             g_abc = 1
        C = int(C)
        
        if A < 0 or (A == 0 and B < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    def score(points):
        pair_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                key = get_line_key(points[i], points[j])
                if key is not None:
                    pair_counts[key] += 1
        
        score_val = 0
        for count in pair_counts.values():
            disc = 1 + 8 * count
            sqrt_disc = math.isqrt(disc)
            if sqrt_disc * sqrt_disc == disc:
                k = (1 + sqrt_disc) / 2
                if k == int(k):
                    k = int(k)
                    if k == 4:
                        score_val += 1
                    elif k > 4:
                        return -1
        return score_val

    def random_config():
        points = []
        seen = set()
        while len(points) < 16:
            x = random.randint(-1000, 1000)
            y = random.randint(-1000, 1000)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        return points

    # Initial configuration: 4x4 grid
    current = [[x, y] for x in range(4) for y in range(4)]
    current_score = score(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    no_improve_count = 0
    restart_threshold = 2000
    
    while time.time() < deadline:
        # Propose move
        move_type = random.random()
        new_points = [p[:] for p in current]
        
        if move_type < 0.5:
            # Small shift
            i = random.randint(0, 15)
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            new_x = current[i][0] + dx
            new_y = current[i][1] + dy
            if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                new_points[i] = [new_x, new_y]
            else:
                continue
        elif move_type < 0.8:
            # Medium shift
            i = random.randint(0, 15)
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            new_x = current[i][0] + dx
            new_y = current[i][1] + dy
            if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                new_points[i] = [new_x, new_y]
            else:
                continue
        else:
            # Swap
            i, j = random.sample(range(16), 2)
            new_points[i], new_points[j] = new_points[j][:], new_points[i][:]
            
        # Check distinctness
        valid = True
        seen = set()
        for p in new_points:
            if tuple(p) in seen:
                valid = False
                break
            seen.add(tuple(p))
        
        if not valid:
            continue
            
        new_score = score(new_points)
        
        if new_score == -1:
            continue
            
        if new_score > current_score:
            current = new_points
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best = [p[:] for p in current]
        else:
            delta = new_score - current_score
            if delta == 0:
                if random.random() < 0.5:
                    current = new_points
                    current_score = new_score
            else:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current = new_points
                    current_score = new_score
            
            no_improve_count += 1
        
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Restart if stuck
        if no_improve_count > restart_threshold:
            current = random_config()
            current_score = score(current)
            if current_score > best_score:
                best_score = current_score
                best = [p[:] for p in current]
            no_improve_count = 0
            T = 1.0
            
    return best
