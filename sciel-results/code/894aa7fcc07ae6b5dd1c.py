import itertools
import time

def score_points(points):
    n = len(points)
    if n < 3:
        return 0
    counts = {}
    # Optimized triple loop
    # 30 points -> 4060 triples. This is very fast.
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                # Doubled area is |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                area2 = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if area2 > 0:
                    counts[area2] = counts.get(area2, 0) + 1
    if not counts:
        return 0
    return max(counts.values())

def construct():
    start_time = time.time()
    candidates = []
    
    # Candidate 1: 6x5 grid (Seed)
    g1 = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(g1)
    
    # Candidate 2: 5x6 grid
    g2 = [(x, y) for x in range(5) for y in range(6)]
    candidates.append(g2)
    
    # Candidate 3: 3x10 grid
    g3 = [(x, y) for x in range(3) for y in range(10)]
    candidates.append(g3)
    
    # Candidate 4: 10x3 grid
    g4 = [(x, y) for x in range(10) for y in range(3)]
    candidates.append(g4)
    
    # Candidate 5: 2x15 grid
    g5 = [(x, y) for x in range(2) for y in range(15)]
    candidates.append(g5)
    
    # Candidate 6: 15x2 grid
    g6 = [(x, y) for x in range(15) for y in range(2)]
    candidates.append(g6)
    
    # Candidate 7: A compact 7x4 grid (28 pts) + 2 pts? 
    # Let's try a 7x4 grid plus 2 points that fit well.
    # Or just a 4x7 grid.
    g7 = [(x, y) for x in range(4) for y in range(7)]
    candidates.append(g7)
    
    # Candidate 8: 7x4 grid
    g8 = [(x, y) for x in range(7) for y in range(4)]
    candidates.append(g8)
    
    # Candidate 9: A 5x5 grid (25) + 5 points from next row?
    # 5x5 is 25. Add (5,0)..(5,4) -> 6x5. Already have that.
    # Add (5,0)..(5,2) and (6,0)..(6,1)? 
    # Let's try a "staircase" or irregular shape? 
    # Usually regular grids are best.
    
    # Let's also try a centered grid to ensure symmetry? 
    # The score is invariant to translation, so 0..W is fine.
    
    best_score = 0
    best_points = g1
    
    for pts in candidates:
        # Check bounds
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts) and len(set(pts)) == 30:
            s = score_points(pts)
            if s > best_score:
                best_score = s
                best_points = pts
                
    # If we have time, we could try a local search, but static evaluation is safe.
    # The 5x6 or 6x5 grid is the most likely winner.
    
    return best_points
