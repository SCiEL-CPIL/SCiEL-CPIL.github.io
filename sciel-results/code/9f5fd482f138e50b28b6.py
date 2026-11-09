import itertools
import numpy as np

def construct():
    # Try multiple candidate configurations and pick the best
    def count_max_multiplicity(points):
        """Count the maximum multiplicity of any positive doubled area."""
        from collections import Counter
        areas = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if det > 0:
                        areas.append(det)
        if not areas:
            return 0
        return max(Counter(areas).values())
    
    def generate_grid(m, n, sx=1, sy=1, ox=0, oy=0):
        """Generate m x n grid with spacing sx, sy and offset ox, oy."""
        pts = []
        for i in range(m):
            for j in range(n):
                x = ox + i * sx
                y = oy + j * sy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        return pts
    
    # Candidate configurations
    candidates = []
    
    # Standard 6x5 grid
    candidates.append(generate_grid(6, 5))
    # 5x6 grid
    candidates.append(generate_grid(5, 6))
    # 10x3 grid
    candidates.append(generate_grid(10, 3))
    # 3x10 grid
    candidates.append(generate_grid(3, 10))
    # 15x2 grid
    candidates.append(generate_grid(15, 2))
    # 2x15 grid
    candidates.append(generate_grid(2, 15))
    # 20x2 (only 40 points, take 30)
    pts = generate_grid(20, 2)
    if len(pts) >= 30:
        candidates.append(pts[:30])
    # 12x3 grid
    candidates.append(generate_grid(12, 3))
    # 3x12 grid
    candidates.append(generate_grid(3, 12))
    # 8x4 grid (32 points, take 30)
    pts = generate_grid(8, 4)
    if len(pts) >= 30:
        candidates.append(pts[:30])
    # 4x8 grid
    pts = generate_grid(4, 8)
    if len(pts) >= 30:
        candidates.append(pts[:30])
    # 7x5 grid (35 points, take 30)
    pts = generate_grid(7, 5)
    if len(pts) >= 30:
        candidates.append(pts[:30])
    # 5x7 grid
    pts = generate_grid(5, 7)
    if len(pts) >= 30:
        candidates.append(pts[:30])
    # 9x4 grid (36 points, take 30)
    pts = generate_grid(9, 4)
    if len(pts) >= 30:
        candidates.append(pts[:30])
    # 4x9 grid
    pts = generate_grid(4, 9)
    if len(pts) >= 30:
        candidates.append(pts[:30])
    
    # Also try with different spacing
    # Grid with spacing 2
    candidates.append(generate_grid(6, 5, sx=2, sy=2))
    candidates.append(generate_grid(5, 6, sx=2, sy=2))
    
    # Hexagonal-like: offset every other row
    def generate_hex(m, n, ox=0, oy=0):
        pts = []
        for i in range(m):
            offset = (i % 2) * 1  # offset by 1 for odd rows
            for j in range(n):
                x = ox + j * 2 + offset
                y = oy + i * 2
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        return pts
    
    pts = generate_hex(6, 5)
    if len(pts) >= 30:
        candidates.append(pts[:30])
    pts = generate_hex(5, 6)
    if len(pts) >= 30:
        candidates.append(pts[:30])
    
    # Find the best candidate
    best_points = None
    best_score = -1
    
    for pts in candidates:
        if len(pts) != 30:
            continue
        # Check all points are distinct and in range
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= x <= 30 and -30 <= y <= 30) for x, y in pts):
            continue
        
        score = count_max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts
    
    if best_points is None:
        # Fallback to simple grid
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
