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
        """
        Returns the number of lines with exactly 4 points.
        Returns -1 if any line has 5 or more points.
        """
        pair_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                key = get_line_key((x1, y1), (x2, y2))
                if key is not None:
                    pair_counts[key] += 1
        
        score_val = 0
        for count in pair_counts.values():
            # k points on a line -> C(k,2) pairs
            # count = k*(k-1)/2
            # We need to check if count corresponds to an integer k >= 5
            # k = (1 + sqrt(1+8*count))/2
            disc = 1 + 8 * count
            sqrt_disc = math.isqrt(disc)
            if sqrt_disc * sqrt_disc == disc:
                k = (1 + sqrt_disc) // 2
                if k >= 5:
                    return -1
                elif k == 4:
                    score_val += 1
        return score_val

    def fast_local_search(points, max_iters=5000):
        """
        Hill climbing local search.
        """
        current = [p[:] for p in points]
        current_score = score(current)
        if current_score == -1:
            return current, current_score

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try moving a random point
            i = random.randint(0, 15)
            cx, cy = current[i]
            
            # Random small step
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
                
            nx, ny = cx + dx, cy + dy
            if not (-1000 <= nx <= 1000 and -1000 <= ny <= 1000):
                continue
            
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
                
            if not improved and random.random() < 0.01:
                # Occasional larger jump to escape local minima
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                nx, ny = cx + dx, cy + dy
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    valid = True
                    for j in range(16):
                        if j != i and current[j][0] == nx and current[j][1] == ny:
                            valid = False
                            break
                    if valid:
                        old = current[i][:]
                        current[i] = [nx, ny]
                        new_score = score(current)
                        current[i] = old
                        if new_score > current_score:
                            current[i] = [nx, ny]
                            current_score = new_score
                            improved = True

            if not improved:
                # If we didn't improve in this iteration, continue
                pass

        return current, current_score

    def generate_seed():
        """
        Generate a seed configuration.
        """
        seed_type = random.random()
        
        if seed_type < 0.5:
            # 4x4 Grid with random small perturbations to break 5-lines
            pts = [[x, y] for x in range(4) for y in range(4)]
            # Perturb 2-4 points
            for _ in range(random.randint(2, 4)):
                i = random.randint(0, 15)
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                if dx == 0 and dy == 0:
                    continue
                pts[i] = [pts[i][0] + dx, pts[i][1] + dy]
            return pts
            
        elif seed_type < 0.8:
            # Random points in a small box
            pts = []
            seen = set()
            while len(pts) < 16:
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                if (x, y) not in seen:
                    seen.add((x, y))
                    pts.append([x, y])
            return pts
            
        else:
            # Structured: 3x3 grid + 7 random
            pts = [[x, y] for x in range(3) for y in range(3)]
            seen = set(tuple(p) for p in pts)
            while len(pts) < 16:
                x = random.randint(-5, 5)
                y = random.randint(-5, 5)
                if (x, y) not in seen:
                    seen.add((x, y))
                    pts.append([x, y])
            return pts

    best_points = None
    best_score = -1
    
    # Try a few initial seeds
    for _ in range(5):
        seed = generate_seed()
        s = score(seed)
        if s > best_score:
            best_score = s
            best_points = seed[:]
            
    # Iterated Local Search
    current = best_points[:]
    current_score = best_score
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
            
        # Local search
        current, current_score = fast_local_search(current, max_iters=1000)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current[:]
            
        # Mutate for next iteration
        # Pick a random point and move it significantly
        i = random.randint(0, 15)
        dx = random.randint(-20, 20)
        dy = random.randint(-20, 20)
        new_x = current[i][0] + dx
        new_y = current[i][1] + dy
        
        if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
            valid = True
            for j in range(16):
                if j != i and current[j][0] == new_x and current[j][1] == new_y:
                    valid = False
                    break
            if valid:
                current[i] = [new_x, new_y]
            else:
                current = best_points[:] # Revert if invalid move
        else:
            current = best_points[:]

    return best_points
