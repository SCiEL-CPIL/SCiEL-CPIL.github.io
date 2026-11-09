import time
import random
import math
from math import gcd

def count_ordinary(points):
    """Count lines containing exactly 2 of the given points using pair counting."""
    n = len(points)
    if n < 3:
        return 0
    
    # Count pairs per line: a line with k points has C(k,2) pairs
    # Ordinary lines have exactly 2 points -> exactly 1 pair
    line_pair_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            # Normalize sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
    
    # Lines with exactly 1 pair are ordinary (exactly 2 points)
    return sum(1 for c in line_pair_counts.values() if c == 1)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_points = None
    best_score = float('inf')
    
    # Generate diverse initial configurations
    def make_grid(cols, rows):
        return [[x, y] for x in range(cols) for y in range(rows)][:12]
    
    def random_distinct_points(seed_val, lo=-15, hi=15):
        rng = random.Random(seed_val)
        pts = set()
        while len(pts) < 12:
            x = rng.randint(lo, hi)
            y = rng.randint(lo, hi)
            pts.add((x, y))
        return [list(p) for p in pts]
    
    # Known good starting points
    initial_configs = [
        make_grid(4, 3),    # 4x3 grid
        make_grid(3, 4),    # 3x4 grid
        random_distinct_points(42),
        random_distinct_points(123),
        random_distinct_points(999),
        # A configuration close to known optimal structures
        [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[3,1],[3,2]],
    ]
    
    # Simulated Annealing with multiple restarts
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Ensure valid: 12 distinct points, within bounds
        if len(set(tuple(p) for p in init_pts)) != 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in init_pts):
            continue
        
        curr_pts = [list(p) for p in init_pts]
        curr_score = count_ordinary(curr_pts)
        
        if curr_score < best_score:
            best_score = curr_score
            best_points = [list(p) for p in curr_pts]
        
        # Simulated Annealing phase
        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.05
        steps = 0
        max_sa_steps = 50000
        
        while time.time() < deadline and steps < max_sa_steps:
            steps += 1
            
            # Pick random point
            i = random.randint(0, 11)
            old_x, old_y = curr_pts[i]
            
            # Generate new position: mix of small and medium moves
            r = random.random()
            if r < 0.5:
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
            elif r < 0.8:
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
            else:
                dx = random.randint(-15, 15)
                dy = random.randint(-15, 15)
            
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Bounds check
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            
            # Distinctness check
            if [new_x, new_y] in curr_pts:
                continue
            
            # Try the move
            curr_pts[i] = [new_x, new_y]
            new_score = count_ordinary(curr_pts)
            
            delta = new_score - curr_score
            
            # Accept if better, or with probability exp(-delta/T)
            if delta <= 0:
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_points = [list(p) for p in curr_pts]
            else:
                if random.random() < math.exp(-delta / T):
                    curr_score = new_score
                else:
                    curr_pts[i] = [old_x, old_y]
            
            T = max(min_T, T * cooling_rate)
        
        # Local search polish: try moving each point to nearby positions
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(12):
                    old_x, old_y = curr_pts[i]
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if dx == 0 and dy == 0:
                                continue
                            new_x = old_x + dx
                            new_y = old_y + dy
                            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                                continue
                            if [new_x, new_y] in curr_pts:
                                continue
                            
                            curr_pts[i] = [new_x, new_y]
                            new_score = count_ordinary(curr_pts)
                            if new_score < curr_score:
                                curr_score = new_score
                                if curr_score < best_score:
                                    best_score = curr_score
                                    best_points = [list(p) for p in curr_pts]
                                improved = True
                            else:
                                curr_pts[i] = [old_x, old_y]
                    if time.time() > deadline:
                        break
    
    # Final check: ensure we have a valid result
    if best_points is None:
        # Fallback: return a valid configuration
        best_points = make_grid(4, 3)
    
    return best_points
