"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            # Canonical form: first nonzero of (a,b) positive
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    # Check not all collinear
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with 3x4 grid
    best_pts = [[x, y] for x in range(4) for y in range(3)]
    best_score = _count_ordinary(best_pts)
    
    # Try some known good starting configurations
    candidates = []
    # 3x4 grid (already best_pts)
    # 4x3 grid (same thing, just transposed)
    # Try a few structured configs
    # Config: 4 points on each of 3 lines forming a triangle
    # Line 1: y=0, x=0,1,2,3
    # Line 2: x=0, y=0,1,2,3 -> but (0,0) shared
    # Let's do: 3 lines, 4 points each, sharing vertices
    # Triangle with 4 points on each side
    # Vertices: (0,0), (10,0), (0,10)
    # Side 1: (0,0),(2.5,0),(5,0),(7.5,0) -> use integers: (0,0),(3,0),(6,0),(9,0)
    # Side 2: (0,0),(0,3),(0,6),(0,9)
    # Side 3: (9,0),(6,3),(3,6),(0,9)
    tri_pts = [(0,0),(3,0),(6,0),(9,0), (0,3),(0,6),(0,9), (6,3),(3,6)]
    # That's 9 points, need 3 more. Add (9,9)? No, let me think differently.
    
    # Alternative: use a 4x3 grid but try local search
    # Local search: try moving each point to nearby integer positions
    import random
    random.seed(42)
    
    # Reduced iterations to ensure finish within time limit
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 11)
        orig_x, orig_y = best_pts[idx]
        
        # Try a random nearby move
        for _ in range(5):
            new_x = orig_x + random.randint(-5, 5)
            new_y = orig_y + random.randint(-5, 5)
            
            # Keep original to avoid duplicates
            new_pts = [list(p) for p in best_pts]
            new_pts[idx] = [new_x, new_y]
            
            if not _is_valid(new_pts):
                continue
            
            score = _count_ordinary(new_pts)
            if score < best_score:
                best_pts = new_pts
                best_score = score
                break
    
    # Final hill climbing: try all single-point moves to nearby positions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(12):
            orig_x, orig_y = best_pts[idx]
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = orig_x + dx, orig_y + dy
                    new_pts = [list(p) for p in best_pts]
                    new_pts[idx] = [new_x, new_y]
                    if not _is_valid(new_pts):
                        continue
                    score = _count_ordinary(new_pts)
                    if score < best_score:
                        best_pts = new_pts
                        best_score = score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
