"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from fractions import Fraction

def _line_key(p, q):
    """Return a canonical key for the line through p and q."""
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (x1, None)
    if dy == 0:
        return (None, y1)
    # slope = dy/dx
    slope = Fraction(dy, dx)
    intercept = Fraction(y1 * dx - x1 * dy, dx)
    return (slope, intercept)

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = {i, j}
            else:
                line_points[key].add(i)
                line_points[key].add(j)
    count = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            count += 1
    return count

def _not_all_collinear(points):
    """Check that not all points lie on a single line."""
    if len(points) < 3:
        return False
    for i in range(len(points) - 2):
        for j in range(i + 1, len(points) - 1):
            for k in range(j + 1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) != 0:
                    return True
    return False

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Start with some hand-crafted configurations
    candidates = []
    
    # 3x4 grid (baseline: has several ordinary lines)
    c1 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(c1)
    
    # 4x3 grid
    c2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(c2)
    
    # Square with diagonals
    c3 = [[0,0],[1,0],[2,0],[3,0],
          [0,1],[1,1],[2,1],[3,1],
          [0,2],[1,2],[2,2],[3,2]]
    # This is the same as c1, skip
    
    # Try configurations based on known low ordinary line sets
    # The Böröczky-type configurations
    
    # A configuration with 6 points on a conic + 6 more
    # Let's try: points on two parallel lines with shared structure
    
    # Configuration inspired by affine plane
    # 9 points of AG(2,3) + 3 additional
    
    # AG(2,3): {(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)}
    base_ag = [[i, j] for i in range(3) for j in range(3)]
    
    # Add (3,0), (3,1), (3,2) - makes 4x3
    cand_a = base_ag + [[3,0],[3,1],[3,2]]
    candidates.append(cand_a)
    
    # Add (0,3), (1,3), (2,3) - makes 3x4
    cand_b = base_ag + [[0,3],[1,3],[2,3]]
    candidates.append(cand_b)
    
    # Add diagonal points: (3,0), (0,3), (3,3)
    cand_c = base_ag + [[3,0],[0,3],[3,3]]
    candidates.append(cand_c)
    
    # Add (3,1), (1,3), (2,4)
    cand_d = base_ag + [[3,1],[1,3],[2,4]]
    candidates.append(cand_d)
    
    # Try a configuration with 3 points per row in a 4-row arrangement
    # with staggered positions to create more 3-point lines
    
    # Configuration: two interleaved triangles
    # Triangle 1: (0,0), (4,0), (2,4)
    # Triangle 2: (1,1), (5,1), (3,5) - but need 12 points
    # Let me try: 6 points on two parallel lines + 6 on two more
    
    # Better approach: use known configurations
    # The Hesse configuration has 9 points, 12 lines of 3
    # Hesse: cube roots of unity structure
    
    # Let's try: 12 points as vertices of a regular hexagon plus center-related
    # Actually, let's try: 3x4 grid but with one point moved
    
    # For now, evaluate what we have and do a local search
    
    best = None
    best_count = float('inf')
    
    for cand in candidates:
        if len(set(tuple(p) for p in cand)) != 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in cand):
            continue
        if not _not_all_collinear(cand):
            continue
        cnt = _count_ordinary_lines(cand)
        if cnt < best_count:
            best_count = cnt
            best = [list(p) for p in cand]
    
    if best is not None and best_count <= 6:
        return best
    
    # Local search: try moving one point at a time
    if best is None:
        best = c1
        best_count = _count_ordinary_lines(best)
    
    if time.time() > deadline - 30:
        return best
    
    # Try perturbation: for each point, try small moves
    import random
    random.seed(42)
    
    current = best
    current_count = best_count
    no_improve = 0
    
    while time.time() < deadline - 20 and no_improve < 200:
        # Pick a random point to move
        idx = random.randint(0, 11)
        old = list(current[idx])
        
        # Try some random nearby positions
        found = False
        for _ in range(50):
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            new_pt = [old[0] + dx, old[1] + dy]
            if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                continue
            
            # Check distinctness
            trial = [list(p) for p in current]
            trial[idx] = new_pt
            if len(set(tuple(p) for p in trial)) != 12:
                continue
            if not _not_all_collinear(trial):
                continue
            
            cnt = _count_ordinary_lines(trial)
            if cnt < current_count:
                current = trial
                current_count = cnt
                best = [list(p) for p in current]
                best_count = current_count
                no_improve = 0
                found = True
                break
            elif cnt == current_count:
                # Accept with some probability (simulated annealing-like)
                if random.random() < 0.3:
                    current = trial
                    no_improve += 1
                    break
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        if found:
            no_improve = 0
        else:
            no_improve += 1
    
    return best
# EVOLVE-BLOCK-END
