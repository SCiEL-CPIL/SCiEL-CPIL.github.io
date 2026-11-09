import random
import time
import math

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    # Initial 4x5 grid
    # Coordinates scaled to fit within [-100, 100]
    # Grid points:
    # x in {-60, -20, 20, 60}
    # y in {-80, -40, 0, 40, 80}
    # This grid has many 4-collinear points (rows of 4, cols of 5, diagonals).
    # We must break 4-collinearities.
    
    xs = [-60, -20, 20, 60]
    ys = [-80, -40, 0, 40, 80]
    
    points = []
    for x in xs:
        for y in ys:
            points.append([x, y])
            
    # Shuffle to break symmetry
    random.shuffle(points)
    
    # Helper: check if 3 points are collinear
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
        
    # Helper: check if any 4 points are collinear
    # Returns True if valid (no 4 collinear), False otherwise
    def has_four_collinear(pts):
        n = len(pts)
        # Count collinear points on lines defined by pairs
        # To be efficient, we can just check all subsets of 4? No, that's 4845.
        # Better: For each pair, count how many points lie on the line.
        # If any count >= 4, return False.
        # But doing this from scratch is O(N^3).
        # For 20 points, 20^3 = 8000 ops. Very fast.
        
        # Optimization: We only need to check if adding a new point creates a 4-collinear.
        # But for initial setup and random restarts, we check all.
        
        # Line representation: (dx/g, dy/g, c/g) where c = x1*y2 - y1*x2
        # Or simpler: use a set of lines.
        
        # Since N=20 is small, O(N^3) is fine.
        for i in range(n):
            for j in range(i+1, n):
                count = 2
                for k in range(j+1, n):
                    if is_collinear(pts[i], pts[j], pts[k]):
                        count += 1
                        if count >= 4:
                            return True
        return False
        
    # Helper: count triples
    def count_triples(pts):
        n = len(pts)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(pts[i], pts[j], pts[k]):
                        count += 1
        return count
        
    # Initial check
    if has_four_collinear(points):
        # The grid has many 4-collinear points. We need to perturb them immediately.
        # We will do a greedy repair or just start local search which will reject invalid moves.
        # However, if the start is invalid, we can't compute a valid triple count for comparison?
        # Actually, the constraint is "no four collinear". The search must maintain this invariant.
        # So we need a valid start.
        
        # Let's generate a random valid start or perturb the grid until valid.
        # A better start: random points in the box? No, low triple density.
        # Let's perturb the grid points randomly by small amounts and check validity.
        
        for _ in range(1000):
            p = points[:]
            for i in range(20):
                # Random small shift
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                p[i] = [p[i][0] + dx, p[i][1] + dy]
                # Keep in bounds
                p[i][0] = max(-100, min(100, p[i][0]))
                p[i][1] = max(-100, min(100, p[i][1]))
            
            if not has_four_collinear(p):
                # Check distinctness
                if len(set(tuple(pt) for pt in p)) == 20:
                    points = p
                    break
        else:
            # Fallback: random points
            while True:
                p = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
                if len(set(tuple(pt) for pt in p)) == 20 and not has_four_collinear(p):
                    points = p
                    break

    current_triples = count_triples(points)
    best_points = points[:]
    best_triples = current_triples
    
    # Local Search
    # Move: pick random index, random small delta
    # Accept if:
    #   1. Valid (no 4 collinear, distinct)
    #   2. Triples increased (or equal with some prob)
    
    T = 5.0  # Temperature for simulated annealing
    T_min = 0.1
    
    # We want to maximize triples.
    
    while time.time() - start_time < deadline:
        i = random.randint(0, 19)
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        
        if dx == 0 and dy == 0:
            continue
            
        new_x = points[i][0] + dx
        new_y = points[i][1] + dy
        
        if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
            continue
            
        new_points = points[:]
        new_points[i] = [new_x, new_y]
        
        # Check distinctness
        # Since we only moved one point, we just need to check if new_points[i] is in the rest
        is_distinct = True
        for j in range(20):
            if j != i and new_points[j] == new_points[i]:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        # Check 4-collinear
        # We can do a fast check:
        # Any new 4-collinear must involve the moved point i.
        # So check all triples (i, j, k) and see if there is a 4th point l on the same line?
        # Or simpler: just run has_four_collinear. It's fast enough.
        if has_four_collinear(new_points):
            continue
            
        new_triples = count_triples(new_points)
        
        delta = new_triples - current_triples
        
        if delta > 0:
            # Accept
            points = new_points
            current_triples = new_triples
            if current_triples > best_triples:
                best_points = points[:]
                best_triples = current_triples
        else:
            # Accept with probability exp(delta / T)
            if random.random() < math.exp(delta / T):
                points = new_points
                current_triples = new_triples
                
        # Cool down
        T = max(T_min, T * 0.999)
        
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
