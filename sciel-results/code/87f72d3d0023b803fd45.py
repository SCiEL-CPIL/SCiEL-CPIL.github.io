import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = math.gcd(dx, dy)
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        C = dy * x1 - dx * y1
        return (dx, dy, C)

    def score_config(points):
        line_pairs = {}
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is not None:
                    line_pairs[key] = line_pairs.get(key, 0) + 1
        
        exact_4 = 0
        invalid = 0
        for key, count in line_pairs.items():
            D = 1 + 8 * count
            sqrt_D = int(math.isqrt(D))
            if sqrt_D * sqrt_D == D:
                n_pts = (1 + sqrt_D) // 2
                if n_pts == 4:
                    exact_4 += 1
                elif n_pts >= 5:
                    invalid += 1
        return exact_4, invalid

    def total_score(points):
        e4, inv = score_config(points)
        return e4 - 1000 * inv

    def is_valid(points):
        # Check distinctness and bounds
        seen = set()
        for p in points:
            if p[0] < -1000 or p[0] > 1000 or p[1] < -1000 or p[1] > 1000:
                return False
            if tuple(p) in seen:
                return False
            seen.add(tuple(p))
        return True

    # Seed 1: 4x4 Grid
    seeds = []
    base = [[x, y] for x in range(4) for y in range(4)]
    seeds.append([p[:] for p in base])
    
    # Seed 2: Random
    for _ in range(5):
        pts = [[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(16)]
        if is_valid(pts):
            seeds.append(pts)

    best_points = None
    best_score = -10**9
    
    for seed in seeds:
        current_points = [p[:] for p in seed]
        current_score = total_score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        
        # Limit time per seed
        seed_deadline = time.time() + 20.0 # 20s per seed? 
        # If we have 6 seeds, 120s.
        
        while time.time() < seed_deadline and T > min_T:
            # Choose move type
            if random.random() < 0.5:
                # Move one point
                idx = random.randint(0, 15)
                orig_x, orig_y = current_points[idx]
                # Random offset
                # Larger steps later? No, SA handles it.
                # Use a Gaussian-like distribution? Or uniform.
                # Uniform in [-10, 10]
                new_x = orig_x + random.randint(-10, 10)
                new_y = orig_y + random.randint(-10, 10)
                
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                    
                # Check collision
                collides = False
                for j in range(16):
                    if j == idx: continue
                    if current_points[j][0] == new_x and current_points[j][1] == new_y:
                        collides = True
                        break
                if collides:
                    continue
                    
                old_p = current_points[idx][:]
                current_points[idx] = [new_x, new_y]
                new_score = total_score(current_points)
                
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                else:
                    current_points[idx] = old_p
            else:
                # Swap two points
                i, j = random.sample(range(16), 2)
                p_i = current_points[i][:]
                p_j = current_points[j][:]
                current_points[i] = p_j
                current_points[j] = p_i
                
                new_score = total_score(current_points)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                else:
                    current_points[i] = p_i
                    current_points[j] = p_j
                    
            T *= cooling
            
        if time.time() > deadline:
            break

    return best_points
