import time
import random
from math import gcd
from itertools import combinations

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            lines[key] = lines.get(key, 0) + 1
            
    ordinary = 0
    for key, pair_count in lines.items():
        if pair_count == 1:
            ordinary += 1
            
    return ordinary

def _fast_local_search(points, deadline, max_iter=200):
    """
    Fast local search with incremental evaluation.
    """
    n = len(points)
    best_points = [p[:] for p in points]
    best_score = count_ordinary_lines(best_points)
    
    # Candidate moves: small and medium steps
    steps = [(dx, dy) for dx in range(-8, 9) for dy in range(-8, 9) if (dx, dy) != (0, 0)]
    random.shuffle(steps)
    
    iters = 0
    while iters < max_iter and time.time() < deadline:
        iters += 1
        improved = False
        
        # Try moving each point
        for i in range(n):
            x, y = best_points[i]
            
            for dx, dy in steps:
                nx, ny = x + dx, y + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_pt = [nx, ny]
                if any(new_pt == p for p in best_points):
                    continue
                
                new_points = best_points[:]
                new_points[i] = new_pt
                new_score = count_ordinary_lines(new_points)
                
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Random restart within neighborhood
            if random.random() < 0.7:
                i = random.randint(0, n-1)
                x, y = best_points[i]
                nx = x + random.randint(-30, 30)
                ny = y + random.randint(-30, 30)
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    new_pt = [nx, ny]
                    if not any(new_pt == p for p in best_points):
                        new_points = best_points[:]
                        new_points[i] = new_pt
                        new_score = count_ordinary_lines(new_points)
                        if new_score <= best_score:
                            best_score = new_score
                            best_points = new_points
            else:
                break
                
    return best_points, best_score

def _try_line_construction(deadline):
    """
    Try to construct a configuration with many 3-point lines.
    Strategy: Start with a base of collinear triples and add points to create more.
    """
    configs = []
    
    # Config A: Two parallel lines of 4 points each + 4 more
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=0,1,2,3
    # Add points to create more 3-point lines
    base_a = [[0,0],[1,0],[2,0],[3,0],
              [0,1],[1,1],[2,1],[3,1],
              [0,2],[3,2],[1,3],[2,-1]]
    configs.append(base_a)
    
    # Config B: Three lines of 4 points forming a triangle-like structure
    base_b = [[0,0],[1,0],[2,0],[3,0],   # horizontal
              [0,1],[0,2],[0,3],[0,4],   # vertical  
              [1,1],[2,2],[3,3],[4,4]]   # diagonal
    configs.append(base_b)
    
    # Config C: Based on known optimal structures for 12 points
    # Try a configuration inspired by Böröczky-type constructions
    base_c = [[0,0],[1,0],[2,0],[3,0],
              [0,1],[1,1],[2,1],[3,1],
              [0,2],[1,2],[2,2],[3,2],
              [4,0],[4,2]]
    configs.append(base_c)
    
    # Config D: More aggressive collinearity
    base_d = [[0,0],[1,0],[2,0],[3,0],
              [0,1],[1,1],[2,1],[3,1],
              [0,2],[3,2],[1,3],[2,-1],
              [4,1]]
    configs.append(base_d)
    
    # Config E: Symmetric cross-like
    base_e = [[-2,0],[-1,0],[0,0],[1,0],[2,0],
              [0,-2],[0,-1],[0,1],[0,2],
              [-1,-1],[1,1],[-1,1],[1,-1]]
    configs.append(base_e)
    
    best_points = None
    best_score = float('inf')
    
    for cfg in configs:
        if time.time() > deadline - 10:
            break
        if len(set(map(tuple, cfg))) != 12:
            continue
        
        pts, score = _fast_local_search(cfg, deadline, max_iter=150)
        if score < best_score:
            best_score = score
            best_points = pts
        if best_score <= 6:
            break
    
    return best_points, best_score

def _simulated_annealing(points, deadline, max_iter=500):
    """
    Simulated annealing to escape local minima.
    """
    n = len(points)
    current_points = [p[:] for p in points]
    current_score = count_ordinary_lines(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    steps = [(dx, dy) for dx in range(-10, 11) for dy in range(-10, 11) if (dx, dy) != (0, 0)]
    
    temp = 5.0
    cooling = 0.995
    
    iters = 0
    while iters < max_iter and time.time() < deadline:
        iters += 1
        
        i = random.randint(0, n-1)
        x, y = current_points[i]
        dx, dy = random.choice(steps)
        nx, ny = x + dx, y + dy
        
        if abs(nx) > 10**9 or abs(ny) > 10**9:
            continue
        
        new_pt = [nx, ny]
        if any(new_pt == p for p in current_points):
            continue
        
        new_points = current_points[:]
        new_points[i] = new_pt
        new_score = count_ordinary_lines(new_points)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(temp, 0.1)):
            current_points = new_points
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        temp *= cooling
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Phase 1: Try direct constructions
    pts1, score1 = _try_line_construction(deadline)
    if pts1 is not None:
        best_points = pts1
        best_score = score1
    
    if best_score <= 6:
        return best_points
    
    # Phase 2: Run simulated annealing from best so far
    if best_points is not None and time.time() < deadline - 20:
        pts2, score2 = _simulated_annealing(best_points, deadline, max_iter=300)
        if score2 < best_score:
            best_score = score2
            best_points = pts2
    
    if best_score <= 6:
        return best_points
    
    # Phase 3: Try additional random restarts
    base_configs = [
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]],
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[3,2],[1,3],[2,-1]],
        [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1],[2,0],[0,2],[0,-2]],
    ]
    
    for cfg in base_configs:
        if time.time() > deadline - 5:
            break
        if len(set(map(tuple, cfg))) != 12:
            continue
        
        pts, score = _fast_local_search(cfg, deadline, max_iter=100)
        if score < best_score:
            best_score = score
            best_points = pts
        if best_score <= 6:
            break
    
    # Final fallback
    if best_points is None:
        best_points = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
    
    return best_points
