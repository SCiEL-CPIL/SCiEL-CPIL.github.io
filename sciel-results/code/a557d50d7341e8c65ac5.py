import time
from fractions import Fraction

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    # For each pair, determine the line, then count how many points are on it
    # Use a set of normalized line representations
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2)
            # Represent as (A, B, C) where Ax + By + C = 0, normalized
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            # Normalize: make gcd(A,B,C) = 1, and ensure A>0 or (A==0 and B>0)
            from math import gcd
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # A line with k points appears C(k,2) times in line_counts
    # Ordinary lines are those with count == 1 (i.e., exactly 2 points)
    ordinary = 0
    for key, cnt in line_counts.items():
        if cnt == 1:
            ordinary += 1
        # cnt == 3 means 3 points (C(3,2)=3), cnt == 6 means 4 points, etc.
    return ordinary

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try multiple configurations and pick the best
    configs = []
    
    # Config 1: 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Config 2: 3x4 grid (transposed)
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Config 3: Two rows of 6 with shifted alignment
    # Points: (0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(1,1),(2,1),(3,1),(4,1),(5,1),(6,1)
    configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(1, 7)])
    
    # Config 4: Hexagonal-ish arrangement
    # 3 rows of 4, but with some diagonal structure
    configs.append([[0,0],[1,0],[2,0],[3,0],
                    [0,1],[1,1],[2,1],[3,1],
                    [0,2],[1,2],[2,2],[3,2]])
    
    # Config 5: 4 points on each of 3 parallel lines, evenly spaced
    # Try to maximize diagonal collinearities
    # Line y=0: x=0,1,2,3
    # Line y=3: x=0,1,2,3  
    # Line y=6: x=0,1,2,3
    # This gives diagonals of slope 1, 2, etc.
    configs.append([[x, 0] for x in range(4)] + [[x, 3] for x in range(4)] + [[x, 6] for x in range(4)])
    
    # Config 6: Try a configuration based on known orchard-planting solutions
    # 12 points: use a 4x3 grid but check if it's optimal
    # Let's also try: points on a parabola y = x^2 mod something
    # Or: (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,3),(1,3),(2,3),(3,3)
    configs.append([[0,0],[1,0],[2,0],[3,0],
                    [0,1],[1,1],[2,1],[3,1],
                    [0,3],[1,3],[2,3],[3,3]])
    
    # Config 7: 6 points forming a hexagon + 6 midpoints
    # Regular hexagon vertices (scaled): (1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1) - not regular
    # Let's use: (0,0),(2,0),(3,1),(2,2),(0,2),(-1,1) and midpoints
    # This is getting complex, let's just evaluate what we have
    
    best_points = None
    best_count = float('inf')
    
    for pts in configs:
        # Verify distinctness
        if len(set(tuple(p) for p in pts)) != 12:
            continue
        # Verify bounds
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in pts):
            continue
        # Verify not all collinear
        if all(pts[i][1] - pts[0][1] == (pts[1][1] - pts[0][1]) * (pts[i][0] - pts[0][0]) / (pts[1][0] - pts[0][0]) if pts[1][0] != pts[0][0] else True for i in range(2, 12)):
            continue
        cnt = count_ordinary_lines(pts)
        if cnt < best_count:
            best_count = cnt
            best_points = pts
    
    if best_points is None:
        # Fallback to 4x3 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
