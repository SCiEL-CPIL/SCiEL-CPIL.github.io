import time
import random
from itertools import combinations
from collections import Counter

def _score(points):
    """Compute the maximum multiplicity of a positive doubled area."""
    areas = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    c = Counter(areas)
    return c.most_common(1)[0][1]

def construct():
    start = time.time()
    deadline = start + 70  # Safety margin before 90s kill
    
    # Start with a grid-based configuration
    # Try different grid sizes and spacings
    best_points = None
    best_score = 0
    
    # Generate candidate configurations from parametric families
    candidates = []
    
    # Family 1: Regular grids
    for nx in range(3, 8):
        for ny in range(3, 8):
            if nx * ny >= 30:
                continue
            # Pad to 30 with extra points
            pts = [(i, j) for i in range(nx) for j in range(ny)]
            if len(pts) < 30:
                # Add points on edges
                extra = 30 - len(pts)
                for i in range(nx):
                    if extra <= 0:
                        break
                    for j in range(ny):
                        if extra <= 0:
                            break
                        p = (i, j + ny)
                        if p not in pts and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                            pts.append(p)
                            extra -= 1
            if len(pts) == 30 and len(set(pts)) == 30:
                candidates.append(pts[:])
    
    # Family 2: Two parallel lines with varying spacing
    for spacing in range(1, 6):
        for offset in range(0, 10):
            pts = []
            for i in range(15):
                pts.append((i * spacing, offset))
            for i in range(15):
                pts.append((i * spacing, offset + 1))
            if len(set(pts)) == 30 and all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
                candidates.append(pts[:])
    
    # Family 3: Three parallel lines
    for h in range(1, 5):
        pts = []
        for i in range(10):
            pts.append((i, 0))
        for i in range(10):
            pts.append((i, h))
        for i in range(10):
            pts.append((i, 2*h))
        if len(set(pts)) == 30 and all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            candidates.append(pts[:])
    
    # Family 4: Arithmetic progression in x, constant y (single line won't work - all areas 0)
    # Instead: points on a circle-like pattern or other structured patterns
    
    # Family 5: Small square grid 5x6
    pts = [(i, j) for i in range(5) for j in range(6)]
    candidates.append(pts[:])
    
    # Family 6: 6x5 grid
    pts = [(i, j) for i in range(6) for j in range(5)]
    candidates.append(pts[:])
    
    # Evaluate candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]
    
    # Local search: swap points to improve score
    if best_points is None:
        best_points = [(i, 0) for i in range(15)] + [(i, 1) for i in range(15)]
        best_score = _score(best_points)
    
    current_points = best_points[:]
    current_score = best_score
    
    # Local search with swaps
    random.seed(42)
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Try random swaps
        for _ in range(50):
            if time.time() > deadline:
                break
            i, j = random.sample(range(30), 2)
            new_points = current_points[:]
            # Try moving point i to a random nearby location
            x, y = new_points[i]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        new_points[i] = (nx, ny)
                        if len(set(new_points)) == 30:
                            s = _score(new_points)
                            if s > current_score:
                                current_points = new_points[:]
                                current_score = s
                                break
                else:
                    continue
                break
    
    return current_points
