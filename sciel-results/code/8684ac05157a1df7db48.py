import time
import sys
from math import gcd
from collections import defaultdict

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    
    ordinary = 0
    for line, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _try_config(points):
    """Validate and count ordinary lines for a configuration."""
    if len(points) != 12:
        return None
    if len(set(tuple(p) for p in points)) != 12:
        return None
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return None
    collinear = True
    for i in range(1, 12):
        dx = points[i][0] - points[0][0]
        dy = points[i][1] - points[0][1]
        for j in range(i + 1, 12):
            dx2 = points[j][0] - points[0][0]
            dy2 = points[j][1] - points[0][1]
            if dx * dy2 != dy * dx2:
                collinear = False
                break
        if not collinear:
            break
    if collinear:
        return None
    return _count_ordinary_lines(points)

def _hexagon_points(scale=1):
    """Generate 12 points on a hexagonal lattice pattern.
    
    Uses vertices of regular hexagons at multiple scales to create
    multiple concurrent lines and 3-point collinearities.
    """
    # Hexagon vertices: (±s, 0), (±s/2, ±s*√3/2) - but we need integers
    # Use hexagonal lattice: points (i + j/2, j*√3/2) for integer i,j
    # To keep integers, use basis vectors (2,0) and (1,√3) -> scale by 2
    # So lattice points: (2i + j, j*√3) -> not integer
    # Alternative: use (3i + j, 2j) as a skewed lattice
    # Or use the standard hexagonal tiling with integer coords:
    # Points at (x, y) where x + y ≡ 0 mod 2 (checkerboard on hex lattice)
    
    # Simpler: use points on 3 lines through origin with 4 points each
    # Line 1: y=0 -> (s,0) for s in {1,2,3,4}
    # Line 2: y=x -> (s,s) for s in {1,2,3,4}  
    # Line 3: y=2x -> (s,2s) for s in {1,2,3,4}
    # This gives 12 points with 3 concurrent lines through origin
    
    pts = []
    for s in range(1, 5):
        pts.append([s, 0])
        pts.append([s, s])
        pts.append([s, 2*s])
    
    # Add more structure: also add points that create additional 3-point lines
    # The above has lines: y=0 (4 pts), y=x (4 pts), y=2x (4 pts)
    # Plus lines like through (1,0),(2,2),(3,4) -> slope 2, already counted
    # Through (1,0),(3,3) -> slope 3/2
    # Let's check if we can do better
    
    return pts

def _grid_with_diagonals():
    """4x3 grid which has many 3-point lines (rows, columns, diagonals)."""
    pts = []
    for x in range(4):
        for y in range(3):
            pts.append([x, y])
    return pts

def _two_parallel_lines():
    """6 points on each of 2 parallel lines, shifted to create diagonals."""
    pts = []
    for i in range(6):
        pts.append([i, 0])
        pts.append([i+1, 1])
    return pts

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    Uses hexagonal lattice structures and aggressive local search.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    # Candidate configurations - focus on structures with many 3+ point lines
    candidates = []
    
    # Config 1: 4x3 grid (has rows, cols, and diagonals with 3+ points)
    pts1 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(pts1)
    
    # Config 2: 3x4 grid
    pts2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(pts2)
    
    # Config 3: 3 concurrent lines with 4 points each (through origin)
    pts3 = []
    for s in range(1, 5):
        pts3.append([s, 0])
        pts3.append([s, s])
        pts3.append([s, 2*s])
    candidates.append(pts3)
    
    # Config 4: 2 parallel lines with 6 each, shifted for diagonals
    pts4 = [[i, 0] for i in range(6)] + [[i+1, 1] for i in range(6)]
    candidates.append(pts4)
    
    # Config 5: Hexagonal pattern - use lattice (2i+j, j) for small i,j
    # This creates a hexagonal-like structure
    pts5 = []
    for j in range(-2, 3):
        for i in range(-2, 3):
            x = 2*i + j
            y = j
            if -10 <= x <= 10 and -10 <= y <= 10:
                pts5.append([x, y])
    # Take first 12 distinct points
    seen = set()
    unique_pts5 = []
    for p in pts5:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_pts5.append(p)
    candidates.append(unique_pts5[:12])
    
    # Config 6: Points on 2 lines with 5 each + 2 more creating extra collinearities
    # Line 1: y=0, x=0..4
    # Line 2: y=2, x=0..4
    # Extra: (0,1) and (4,1) -> creates vertical 3-point lines at x=0 and x=4
    pts6 = [[i, 0] for i in range(5)] + [[i, 2] for i in range(5)] + [[0, 1], [4, 1]]
    candidates.append(pts6)
    
    # Config 7: 3 lines with 4 points each, not all concurrent
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=1,2,3,4
    # Line 3: y=2, x=2,3,4,5
    # This creates diagonal 3-point lines
    pts7 = [[i, 0] for i in range(4)] + [[i, 1] for i in range(1, 5)] + [[i, 2] for i in range(2, 6)]
    candidates.append(pts7)
    
    # Config 8: 6 on x-axis, 3 on y=1, 3 on y=2
    # Arrange to maximize 3-point diagonals
    pts8 = [[i, 0] for i in range(6)] + [[i, 1] for i in range(3)] + [[i+3, 1] for i in range(3)]
    # Fix: make y=1 and y=2 lines
    pts8 = [[i, 0] for i in range(6)] + [[i, 1] for i in range(3)] + [[i, 2] for i in range(3)]
    candidates.append(pts8)
    
    # Config 9: Pappus-like configuration
    # Two lines with 4 points each, plus 4 more creating cross-collinearities
    pts9 = [[i, 0] for i in range(4)] + [[i, 1] for i in range(4)]
    # Add points that create diagonal 3-point lines
    # (0,0), (1,1), (2,2) -> but (2,2) not in set
    # Let's try: (0,0), (2,1), (4,2) -> slope 1/2
    # Adjust: use points (0,0), (1,1), (2,2) by adding (2,2)
    # But we need exactly 12. Let's use a different approach.
    # Use: 4 on y=0, 4 on y=1, 2 on y=2, 2 on y=3
    pts9 = [[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[1, 2], [3, 2]] + [[0, 3], [4, 3]]
    candidates.append(pts9)
    
    # Config 10: Regular hexagon vertices + center + 5 more
    # Hexagon: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) scaled
    # Actually use: (2,0), (1,1), (-1,1), (-2,0), (-1,-1), (1,-1)
    # Plus (0,0) center
    # Plus 5 more to create additional collinearities
    pts10 = [[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1], [0, 0]]
    # Add 5 more: extend the lines
    pts10 += [[4, 0], [3, 2], [-3, 2], [-4, 0], [0, 2]]
    candidates.append(pts10)
    
    # Config 11: 5 on one line, 5 on another, 2 creating cross-lines
    pts11 = [[i, 0] for i in range(5)] + [[i, 1] for i in range(5)] + [[0, 2], [4, 2]]
    candidates.append(pts11)
    
    # Config 12: 4x3 grid but with one point moved to create more 3-point lines
    pts12 = [[x, y] for x in range(4) for y in range(3)]
    # Move (3,2) to (2,2) -> but that's already there
    # Move (3,2) to (1,3)
    pts12[11] = [1, 3]
    candidates.append(pts12)
    
    # Find the best configuration from candidates
    best_points = None
    best_count = float('inf')
    
    for idx, pts in enumerate(candidates):
        if time.time() > deadline - 30:  # Leave time for local search
            break
        if len(pts) != 12:
            continue
        count = _try_config(pts)
        if count is not None and count < best_count:
            best_count = count
            best_points = [list(p) for p in pts]
    
    # If no valid config found, fallback
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_count = _try_config(best_points)
        if best_count is None:
            best_points = [[i, 0] for i in range(11)] + [[0, 1]]
            best_count = 11
    
    # Aggressive local search: try to improve by moving one point at a time
    # with larger moves and multiple passes
    if best_points is not None and time.time() < deadline - 10:
        improved = True
        max_rounds = 50
        round_num = 0
        while improved and time.time() < deadline - 10 and round_num < max_rounds:
            improved = False
            round_num += 1
            # Increase move range as we iterate
            move_range = min(3 + round_num // 5, 10)
            for i in range(12):
                if time.time() > deadline - 10:
                    break
                orig_x, orig_y = best_points[i]
                # Try perturbations
                for dx in range(-move_range, move_range + 1):
                    for dy in range(-move_range, move_range + 1):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = [orig_x + dx, orig_y + dy]
                        if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                            continue
                        # Check distinctness
                        if any(tuple(new_pt) == tuple(best_points[j]) for j in range(12) if j != i):
                            continue
                        trial = [list(p) for p in best_points]
                        trial[i] = new_pt
                        count = _try_config(trial)
                        if count is not None and count < best_count:
                            best_points = trial
                            best_count = count
                            improved = True
                            # Don't break - keep searching for more improvements
    
    # Final polish: try swapping pairs of points
    if best_points is not None and time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, 12):
                    if time.time() > deadline - 5:
                        break
                    # Try swapping point i with a nearby point
                    orig_i = best_points[i][:]
                    orig_j = best_points[j][:]
                    # Move point i to where point j is, and point j to a nearby position
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            new_i = [orig_j[0] + dx, orig_j[1] + dy]
                            new_j = [orig_i[0], orig_i[1]]
                            if abs(new_i[0]) > 10**9 or abs(new_i[1]) > 10**9:
                                continue
                            # Check distinctness
                            trial = [list(p) for p in best_points]
                            trial[i] = new_i
                            trial[j] = new_j
                            if len(set(tuple(p) for p in trial)) != 12:
                                continue
                            count = _try_config(trial)
                            if count is not None and count < best_count:
                                best_points = trial
                                best_count = count
                                improved = True
    
    return best_points
