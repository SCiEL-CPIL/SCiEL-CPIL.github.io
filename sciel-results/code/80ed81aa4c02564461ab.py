import time
from itertools import combinations
from math import gcd

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

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
        
        g = _gcd(_gcd(abs(A), abs(B)), abs(C))
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

def _generate_line_candidates(points):
    """
    Generate candidate positions for a new point by finding intersections of lines
    through pairs of existing points, and extending lines.
    """
    candidates = set()
    n = len(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Direction vector
            dx = x2 - x1
            dy = y2 - y1
            
            # Try extending the line by small integer multiples
            for t in range(-3, 4):
                if t == 0:
                    continue
                nx = x1 + t * dx
                ny = y1 + t * dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    candidates.add((nx, ny))
            
            # Also try perpendicular directions for variety
            if dx != 0 or dy != 0:
                # Normalize direction
                g = _gcd(abs(dx), abs(dy))
                if g != 0:
                    dx_n = dx // g
                    dy_n = dy // g
                    # Perpendicular direction
                    px = -dy_n
                    py = dx_n
                    for t in range(-2, 3):
                        if t == 0:
                            continue
                        nx = x1 + t * px
                        ny = y1 + t * py
                        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                            candidates.add((nx, ny))
    
    return list(candidates)

def _local_search(points, deadline, max_iter=300):
    """Local search: try moving individual points to reduce ordinary lines."""
    n = len(points)
    current_score = count_ordinary_lines(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    iters = 0
    
    while iters < max_iter and time.time() < deadline:
        iters += 1
        improved = False
        
        # Try moving each point
        for i in range(n):
            x, y = best_points[i]
            
            # Generate candidates: nearby positions + line-based candidates
            candidates = set()
            
            # Nearby grid
            for dx in range(-12, 13):
                for dy in range(-12, 13):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        candidates.add((nx, ny))
            
            # Line-based candidates from other points
            other_points = [best_points[j] for j in range(n) if j != i]
            line_cands = _generate_line_candidates(other_points)
            for cand in line_cands[:50]:  # Limit to avoid too many
                if cand not in candidates:
                    candidates.add(cand)
            
            # Try each candidate
            for nx, ny in candidates:
                new_pt = [nx, ny]
                # Check distinctness
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
            # Try larger jumps
            for i in range(n):
                x, y = best_points[i]
                for dx in range(-30, 31, 5):
                    for dy in range(-30, 31, 5):
                        if dx == 0 and dy == 0:
                            continue
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
                if improved:
                    break
            if not improved:
                break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Known good configurations to try
    configs = []
    
    # Config 1: 3x3 grid + 3 extensions
    base33 = [[x, y] for x in range(3) for y in range(3)]
    ext1 = [[3, 0], [0, 3], [3, 3]]
    configs.append(base33 + ext1)
    
    # Config 2: Different extensions
    ext2 = [[-1, 1], [1, -1], [4, 2]]
    configs.append(base33 + ext2)
    
    # Config 3: Another set
    ext3 = [[2, 3], [3, 2], [4, 1]]
    configs.append(base33 + ext3)
    
    # Config 4: Try points that create more collinearities
    ext4 = [[1, 3], [3, 1], [4, 0]]
    configs.append(base33 + ext4)
    
    # Config 5: Use a different base structure
    base2 = [[0,0],[1,0],[2,0],[3,0],
             [0,1],[0,2],[0,3],
             [1,1],[2,2],
             [4,1],[1,4]]
    configs.append(base2)
    
    # Config 6: Symmetric configuration
    base3 = [[-1,-1],[0,-1],[1,-1],
             [-1,0],[0,0],[1,0],
             [-1,1],[0,1],[1,1],
             [2,0],[0,2],[0,-2]]
    configs.append(base3)
    
    # Config 7: Another symmetric
    base4 = [[0,0],[1,0],[2,0],[3,0],
             [0,1],[1,1],[2,1],[3,1],
             [0,2],[3,2],[1,3],[2,-1]]
    configs.append(base4)
    
    # Config 8: 4-point lines
    configs.append([[0,0],[1,0],[2,0],[3,0],
                   [0,1],[1,1],[2,1],[3,1],
                   [0,2],[1,2],[2,2],[3,2]])
    
    # Config 9: Try to get 3-point lines
    configs.append([[0,0],[1,0],[2,0],
                   [0,1],[1,1],[2,1],
                   [0,2],[1,2],[2,2],
                   [3,0],[0,3],[3,3]])
    
    # Config 10: Fano-like structure
    configs.append([[0,0],[1,0],[2,0],[4,0],
                   [0,1],[1,1],[2,1],[4,1],
                   [0,2],[1,2],[2,2],[4,2]])
    
    # Evaluate all initial configs
    best_points = None
    best_score = float('inf')
    
    for cfg in configs:
        if len(set(map(tuple, cfg))) != 12:
            continue
        score = count_ordinary_lines(cfg)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
    
    # Local search from the best initial config
    if best_points is not None:
        best_points, best_score = _local_search(best_points, deadline)
    
    # If we still have time, try local search from other configs
    if time.time() < deadline - 10:
        for cfg in configs:
            if len(set(map(tuple, cfg))) != 12:
                continue
            if cfg == best_points:
                continue
            score = count_ordinary_lines(cfg)
            if score < best_score + 2:  # Only if close to best
                pts, sc = _local_search([p[:] for p in cfg], deadline, max_iter=200)
                if sc < best_score:
                    best_score = sc
                    best_points = pts
    
    # Final check
    if best_points is None:
        best_points = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
    
    return best_points
