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
        dx //= g
        dy //= g
        
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
            
        A = dy
        B = -dx
        C = A * x1 + B * y1
        
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
            # k points on a line -> C(k,2) pairs
            # count = C(k,2) = k*(k-1)/2
            # k = (1 + sqrt(1+8*count))/2
            disc = 1 + 8 * count
            sqrt_disc = math.isqrt(disc)
            if sqrt_disc * sqrt_disc == disc:
                k = (1 + sqrt_disc) // 2
                if k == 4:
                    score_val += 1
                elif k > 4:
                    return -1
        return score_val

    def fast_local_search(points, max_iters=2000):
        """Greedy local search to improve exact-4 line count."""
        current = [p[:] for p in points]
        current_score = score(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try moving each point in small steps
            for i in range(16):
                cx, cy = current[i]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                            # Check distinctness
                            valid = True
                            for j in range(16):
                                if j != i and current[j][0] == nx and current[j][1] == ny:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            old = current[i][:]
                            current[i] = [nx, ny]
                            new_score = score(current)
                            current[i] = old
                            
                            if new_score > current_score:
                                current[i] = [nx, ny]
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current, current_score

    def structured_seed():
        """Generate a structured seed with many 4-point lines."""
        # Try several structured configurations
        seeds = []
        
        # 4x4 grid
        seeds.append([[x, y] for x in range(4) for y in range(4)])
        
        # 4x4 grid shifted
        seeds.append([[x+1, y+1] for x in range(4) for y in range(4)])
        
        # Parabolic-ish
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, i*i + j])
        seeds.append(pts)
        
        # Two 4x4 grids offset
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
                pts.append([i+10, j+10])
        # Take first 16
        seeds.append(pts[:16])
        
        # Random structured
        pts = []
        seen = set()
        for _ in range(16):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        seeds.append(pts)
        
        best_seed = seeds[0]
        best_seed_score = score(seeds[0])
        
        for s in seeds[1:]:
            s_score = score(s)
            if s_score > best_seed_score:
                best_seed_score = s_score
                best_seed = s
        
        return best_seed, best_seed_score

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

    # Start with best structured seed, then improve with local search
    current, current_score = structured_seed()
    current, current_score = fast_local_search(current)
    
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    
    no_improve_count = 0
    restart_threshold = 1500
    
    while time.time() < deadline:
        # Propose move
        move_type = random.random()
        new_points = [p[:] for p in current]
        
        if move_type < 0.4:
            # Small shift
            i = random.randint(0, 15)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            new_x = current[i][0] + dx
            new_y = current[i][1] + dy
            if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                new_points[i] = [new_x, new_y]
            else:
                continue
        elif move_type < 0.7:
            # Medium shift
            i = random.randint(0, 15)
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
            new_x = current[i][0] + dx
            new_y = current[i][1] + dy
            if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                new_points[i] = [new_x, new_y]
            else:
                continue
        elif move_type < 0.85:
            # Swap
            i, j = random.sample(range(16), 2)
            new_points[i], new_points[j] = new_points[j][:], new_points[i][:]
        else:
            # Large jump
            i = random.randint(0, 15)
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
            new_x = current[i][0] + dx
            new_y = current[i][1] + dy
            if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                new_points[i] = [new_x, new_y]
            else:
                continue
            
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
                # Run quick local search after improvement
                improved, improved_score = fast_local_search(current, max_iters=500)
                if improved_score > best_score:
                    current = improved
                    current_score = improved_score
                    best_score = improved_score
                    best = [p[:] for p in current]
        else:
            delta = new_score - current_score
            if delta == 0:
                if random.random() < 0.3:
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
            # Try structured seed + local search
            new_current, new_score = structured_seed()
            new_current, new_score = fast_local_search(new_current)
            if new_score > best_score:
                best_score = new_score
                best = [p[:] for p in new_current]
            current = new_current
            current_score = new_score
            no_improve_count = 0
            T = 1.0
            
    return best
