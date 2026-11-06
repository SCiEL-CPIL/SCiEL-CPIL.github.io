import random
import itertools

def count_collinear(points, k):
    """Count number of subsets of size k that are collinear."""
    count = 0
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = points[i], points[j]
            # Check other points
            # This is O(n^3) which is 20^3 = 8000. Fast.
            # But we need to count distinct k-tuples.
            # Better: for each pair, count how many other points are on the line.
            pass
    # Simpler: iterate all combinations of k points?
    # C(20, 3) = 1140. C(20, 4) = 4845.
    # Checking collinearity for each is fast.
    for comb in itertools.combinations(points, k):
        if is_collinear(comb):
            count += 1
    return count

def is_collinear(pts):
    # pts is a list of tuples (x,y)
    if len(pts) < 3:
        return False
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x3, y3 = pts[2]
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def construct():
    random.seed(42)
    best_points = None
    best_score = -1 # We want to maximize 3-collinearities? Or minimize 4?
    # Priority 1: No 4 collinear.
    # Priority 2: Maximize 3 collinearities.
    
    # Initial random set
    points = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(20)]
    # Ensure distinct
    while len(set(points)) < 20:
        points[random.randrange(20)] = (random.randint(-100, 100), random.randint(-100, 100))
        
    best_key = (count_collinear(points, 4), -count_collinear(points, 3))
    best_points = points
    
    for _ in range(10000):
        # Pick two indices
        i, j = random.sample(range(20), 2)
        # Generate new point for j
        new_p = (random.randint(-100, 100), random.randint(-100, 100))
        if new_p in points:
            continue
            
        old_p = points[j]
        points[j] = new_p
        
        c4 = count_collinear(points, 4)
        c3 = count_collinear(points, 3)
        
        # We want c4 = 0. If c4 > 0, we penalize heavily.
        # We want c3 to be high.
        current_key = (c4, -c3)
        
        if current_key < best_key:
            best_key = current_key
            best_points = points[:]
        else:
            # Revert if not better? Or accept with probability?
            # Simple hill climbing: only accept if better.
            # But we might get stuck.
            # Let's just keep the best found so far.
            points[j] = old_p
        # If we found a valid set (c4=0), we can stop?
        # No, we want to maximize c3 among valid sets.
        
    return [[p[0], p[1]] for p in best_points]
