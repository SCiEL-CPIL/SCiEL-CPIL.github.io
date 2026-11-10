"""
Strategy: Line-Based Combinatorial Construction
Instead of moving individual points, we construct the configuration by selecting
sets of 4 collinear points (lines) from a precomputed catalog. We use a greedy
algorithm with backtracking to pack as many such lines as possible into a set
of <= 16 distinct points, ensuring no 5 points are collinear.
"""
import time
import random
from itertools import combinations

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Leave 10s margin for 150s limit

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def normalize_line(points):
    """
    Normalize a set of points to a canonical representation.
    Returns a tuple (dx, dy, c) such that y = (dy/dx)x + c, or similar.
    For vertical lines, handle separately.
    """
    if len(points) < 2:
        return None
    
    # Check if vertical
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    if len(set(xs)) == 1:
        # Vertical line x = c
        return (0, 1, xs[0])
        
    # Find slope
    x1, y1 = points[0]
    x2, y2 = points[1]
    dx = x2 - x1
    dy = y2 - y1
    
    # Reduce fraction
    if dx == 0:
        return (0, 1, x1)
        
    g = abs(__import__('math').gcd(dx, dy))
    dx //= g
    dy //= g
    
    if dx < 0:
        dx = -dx
        dy = -dy
        
    # c = y - (dy/dx)x => dx*c = dx*y - dy*x
    # To avoid floats, use integer form: dy*x - dx*y = c_int
    c_int = dy * x1 - dx * y1
    
    return (dx, dy, c_int)

def get_line_points(line_key, grid_bounds):
    """
    Given a normalized line key, generate all integer points on this line
    within the grid bounds.
    """
    dx, dy, c = line_key
    points = []
    
    if dx == 0:
        # Vertical line x = c
        x = c
        for y in range(grid_bounds[0], grid_bounds[1]+1):
            points.append((x, y))
    else:
        # dy*x - dx*y = c
        # We need to find integer solutions.
        # y = (dy*x - c) / dx
        # We iterate x and check if y is integer and in bounds.
        # To be efficient, we can find the step.
        # dy*x - c must be divisible by dx.
        # Let g = gcd(dy, dx). Since we reduced dx,dy, gcd(dy,dx)=1.
        # So dy*x - c must be 0 mod dx.
        # dy*x = c mod dx.
        # x = (c * inv(dy, dx)) mod dx.
        
        # However, iterating x over the full range is fine for small grids.
        # Grid is roughly -1000 to 1000? No, we can restrict to a smaller box 
        # where 16 points might fit densely. 
        # Actually, the problem says |x|,|y| <= 1000. 
        # But dense configurations are likely in a smaller box.
        # We will precompute lines in a box of [-50, 50] first, 
        # as 16 points with many 4-lines likely fit there.
        
        pass # This function is used in precomputation, see below

def precompute_lines(box_size=50):
    """
    Precompute all lines that contain at least 4 integer points
    within the box [-box_size, box_size].
    """
    lines = []
    coords = range(-box_size, box_size + 1)
    
    # We iterate over all pairs of points to define a line, then extend it.
    # This is O(N^4) if naive, but we can optimize.
    # Better: Iterate over all possible slopes (dx, dy) with small values.
    
    # Slopes with small dx, dy
    # Max slope magnitude? If dx, dy are large, the line is sparse.
    # For 4 points to fit in a box of size 2*B, the slope must be small.
    # If dx=1, dy=0 (horizontal), 4 points fit easily.
    # If dx=1, dy=1 (diagonal), 4 points fit easily.
    # If dx=1, dy=10, 4 points span 30 units, fits in B=50.
    
    max_slope = 10
    
    for dx in range(0, max_slope + 1):
        for dy in range(-max_slope, max_slope + 1):
            if dx == 0 and dy == 0:
                continue
            
            # Normalize
            if dx == 0:
                # Vertical
                pass
            else:
                # Ensure dx > 0 for unique representation
                if dx < 0:
                    dx, dy = -dx, -dy
                # Reduce
                g = abs(__import__('math').gcd(dx, dy))
                if g > 1:
                    dx //= g
                    dy //= g
                    if dx < 0:
                        dx, dy = -dx, -dy
            
            # Iterate over intercepts
            # Line equation: dy*x - dx*y = c
            # We need to find c such that there are at least 4 integer points in the box.
            
            if dx == 0:
                # Vertical x = c
                for c in coords:
                    # Points are (c, y) for y in coords
                    # All points in the column are collinear.
                    # The line is defined by x=c.
                    # Key: (0, 1, c)
                    key = (0, 1, c)
                    # We store the key. We don't store all points yet to save memory.
                    lines.append(key)
            else:
                # dy*x - dx*y = c
                # y = (dy*x - c) / dx
                # For a given c, the points are (x, (dy*x - c)//dx) where (dy*x - c) % dx == 0.
                # The x-coordinates of integer points on the line are spaced by dx.
                # We need to find c values that yield >= 4 points in the box.
                
                # c = dy*x - dx*y
                # x in [-B, B], y in [-B, B]
                # c ranges from dy*(-B) - dx*B to dy*B - dx*(-B)
                # c_min = -dy*B - dx*B (if dy>0, dx>0) -> -B(dy+dx)
                # c_max = dy*B + dx*B -> B(dy+dx)
                
                c_min = -box_size * (abs(dy) + dx)
                c_max = box_size * (abs(dy) + dx)
                
                # We don't need to check every c.
                # The set of c values that have integer solutions is all integers?
                # No, for a fixed (dx, dy), c can be any integer, but the points
                # shift.
                # Actually, for any integer c, there are integer solutions if gcd(dy, dx)=1.
                # We reduced them, so gcd is 1.
                # So for every c, there is a line.
                # We just need to check if that line has >= 4 points in the box.
                
                # Optimization: Instead of iterating all c, we can pick a reference point (x0, y0)
                # in the box and calculate c = dy*x0 - dx*y0.
                # Then the line passes through (x0, y0).
                # We can just iterate over all points (x0, y0) and generate the line key.
                # This duplicates keys, but we can use a set.
                
                pass # We will use the point-iteration method below for simplicity

    # Simpler Precomputation: Iterate all pairs of points in a smaller box
    # Box size 20 is likely sufficient for dense 16-point configs.
    box = 20
    points_in_box = [(x, y) for x in range(-box, box+1) for y in range(-box, box+1)]
    
    line_set = set()
    # To ensure we get lines with >= 4 points, we iterate pairs (p1, p2)
    # and check if the line defined by them contains other points in the box.
    # This is O(N^4) which is 41^4 ~ 2.8M, acceptable.
    
    for i in range(len(points_in_box)):
        p1 = points_in_box[i]
        for j in range(i+1, len(points_in_box)):
            p2 = points_in_box[j]
            
            key = normalize_line([p1, p2])
            if key is None:
                continue
            
            # Check if this line has at least 4 points in the box
            # We can do this by counting points in points_in_box that satisfy the line eq.
            # But doing this for every pair is slow.
            # Alternative: Only add to set, then filter later.
            line_set.add(key)
            
    # Filter: Keep only lines with >= 4 points in the box
    valid_lines = []
    for key in line_set:
        dx, dy, c = key
        count = 0
        if dx == 0:
            # x = c
            if -box <= c <= box:
                count = 2 * box + 1
        else:
            # dy*x - dx*y = c
            # y = (dy*x - c) / dx
            for x in range(-box, box+1):
                num = dy * x - c
                if num % dx == 0:
                    y = num // dx
                    if -box <= y <= box:
                        count += 1
                        if count >= 4:
                            break
        
        if count >= 4:
            # Store the line and its points
            pts = []
            if dx == 0:
                pts = [(c, y) for y in range(-box, box+1)]
            else:
                for x in range(-box, box+1):
                    num = dy * x - c
                    if num % dx == 0:
                        y = num // dx
                        if -box <= y <= box:
                            pts.append((x, y))
            valid_lines.append((key, pts))
            
    return valid_lines

def count_4_lines(points):
    """
    Count the number of lines containing exactly 4 of the given points.
    """
    if len(points) < 4:
        return 0
    
    pts_set = set(points)
    count = 0
    
    # Iterate over all pairs to define lines
    # This is O(N^4) for N=16, which is 16^4/2 ~ 16k, very fast.
    for i in range(len(points)):
        p1 = points[i]
        for j in range(i+1, len(points)):
            p2 = points[j]
            
            key = normalize_line([p1, p2])
            if key is None:
                continue
            
            # Count points on this line
            # We can just check all other points
            line_pts = [p1, p2]
            for k in range(len(points)):
                if k == i or k == j:
                    continue
                p3 = points[k]
                # Check if p3 is on the line
                dx, dy, c = key
                if dx == 0:
                    if p3[0] == c:
                        line_pts.append(p3)
                else:
                    if dy * p3[0] - dx * p3[1] == c:
                        line_pts.append(p3)
            
            if len(line_pts) == 4:
                # Ensure we count each line only once
                # The key is unique for the line.
                # But we are iterating pairs. We will count the same line multiple times.
                # We need to count distinct lines.
                # We can store seen keys.
                pass
                
    # Better: Use a set of line keys
    seen_keys = set()
    for i in range(len(points)):
        p1 = points[i]
        for j in range(i+1, len(points)):
            p2 = points[j]
            key = normalize_line([p1, p2])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            dx, dy, c = key
            line_pts = []
            for p in points:
                if dx == 0:
                    if p[0] == c:
                        line_pts.append(p)
                else:
                    if dy * p[0] - dx * p[1] == c:
                        line_pts.append(p)
                        
            if len(line_pts) == 4:
                count += 1
                
    return count

def has_5_collinear(points):
    """
    Check if any 5 points are collinear.
    """
    pts_set = set(points)
    n = len(points)
    for i in range(n):
        p1 = points[i]
        for j in range(i+1, n):
            p2 = points[j]
            key = normalize_line([p1, p2])
            if key is None:
                continue
            dx, dy, c = key
            count = 0
            for p in points:
                if dx == 0:
                    if p[0] == c:
                        count += 1
                else:
                    if dy * p[0] - dx * p[1] == c:
                        count += 1
            if count >= 5:
                return True
    return False

def construct():
    # 1. Precompute lines
    # Use a box of size 30 to be safe
    lines = precompute_lines(box_size=30)
    
    # Sort lines by "promisingness"
    # A line is promising if it has many points (more points = more potential to form 4-subsets? 
    # No, we want lines that intersect with other lines to form a dense cluster).
    # Let's just sort by the number of points in the line (descending)
    # Or better, randomize to avoid bias, but prefer lines with more points.
    
    # We will use a greedy set packing approach.
    # State: current set of points (list), current set of lines (list of keys)
    
    best_score = 0
    best_points = None
    
    # Initial seed: 4x4 grid
    seed_points = [(x, y) for x in range(4) for y in range(4)]
    # Shift to center?
    seed_points = [(x-1.5, y-1.5) for x in range(4) for y in range(4)]
    # Make integer:
    seed_points = [(x*2, y*2) for x in range(4) for y in range(4)]
    # This is 8x8 grid scaled by 2. Points: 0,2,4,6.
    
    # Let's start with the standard 4x4 grid (0-3)
    seed_points = [(x, y) for x in range(4) for y in range(4)]
    best_score = count_4_lines(seed_points)
    best_points = seed_points
    
    # Greedy construction
    # We try to add lines one by one.
    # If adding a line requires adding points, we check if total points <= 16.
    
    current_points = set(seed_points)
    current_lines = []
    
    # We will try to improve by swapping or adding.
    # Given the complexity, let's do a simple local search on the point set.
    # But the directive asks for a structural change.
    
    # Let's try a "Line Packing" search:
    # 1. Sort lines by length (desc)
    # 2. Try to pick lines such that their union has <= 16 points.
    
    lines_sorted = sorted(lines, key=lambda x: len(x[1]), reverse=True)
    
    # Randomize order slightly to explore different packs
    random.shuffle(lines_sorted)
    
    # Backtracking search for a set of lines
    def backtrack(line_index, current_pts_set, current_line_keys, depth):
        nonlocal best_score, best_points
        
        if _time_left() < 0.5:
            return
            
        if depth > 20: # Limit depth
            return
            
        # Evaluate current state
        if len(current_pts_set) == 16:
            pts = list(current_pts_set)
            if not has_5_collinear(pts):
                score = count_4_lines(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
                    print(f"New best: {score} at depth {depth}")
            return
            
        # Try to add a line
        for i in range(line_index, len(lines_sorted)):
            key, pts = lines_sorted[i]
            
            # Check if this line is compatible
            new_pts_set = current_pts_set.union(pts)
            if len(new_pts_set) > 16:
                continue
                
            # Check 5-collinear constraint on the new set
            # Optimization: Only check lines formed by the new points?
            # For now, check full set.
            if has_5_collinear(list(new_pts_set)):
                continue
                
            # If compatible, recurse
            backtrack(i + 1, new_pts_set, current_line_keys + [key], depth + 1)
            
            # Pruning: If we've tried enough lines at this level, stop
            if _time_left() < 1.0:
                return

    # Start backtracking from empty set? Or from seed?
    # Starting from empty might miss good clusters.
    # Let's start from the seed points' lines.
    
    seed_line_keys = set()
    for i in range(len(seed_points)):
        for j in range(i+1, len(seed_points)):
            key = normalize_line([seed_points[i], seed_points[j]])
            if key:
                seed_line_keys.add(key)
                
    # We don't have the pts for these keys in lines_sorted easily.
    # Let's just start from empty and let the search find a good pack.
    # To speed up, we can bias the search towards the center.
    
    # Run a few restarts with different randomizations
    for _ in range(5):
        random.shuffle(lines_sorted)
        backtrack(0, set(), [], 0)
        
        if _time_left() < 10.0:
            break
            
    # If best_points is still the seed, return it.
    if best_points is None:
        return [[x, y] for x, y in seed_points]
        
    return [[x, y] for x, y in best_points]

# Keep the original interface
def construct_original():
    return [[x, y] for x in range(4) for y in range(4)]

# The main entry point must be construct
# We override construct with our new logic
# But we must ensure it returns the right format.

# One issue: precompute_lines is slow if called every time.
# We should cache it.
_LINES_CACHE = None

def construct():
    global _LINES_CACHE
    if _LINES_CACHE is None:
        _LINES_CACHE = precompute_lines(box_size=30)
        
    lines = _LINES_CACHE
    
    best_score = 0
    best_points = None
    
    seed_points = [(x, y) for x in range(4) for y in range(4)]
    best_score = count_4_lines(seed_points)
    best_points = seed_points
    
    lines_sorted = sorted(lines, key=lambda x: len(x[1]), reverse=True)
    
    # We will use a simple greedy with random restarts
    # Instead of full backtracking which might be too slow, 
    # let's do a greedy pick with local search.
    
    for _ in range(100): # 100 restarts
        if _time_left() < 1.0:
            break
            
        random.shuffle(lines_sorted)
        
        current_pts_set = set()
        current_line_keys = set()
        
        for key, pts in lines_sorted:
            if _time_left() < 0.5:
                break
                
            new_pts_set = current_pts_set.union(pts)
            if len(new_pts_set) > 16:
                continue
                
            # Check 5-collinear
            # Optimization: Only check if the new points create a 5-collinear set
            # For simplicity, check full set
            if has_5_collinear(list(new_pts_set)):
                continue
                
            current_pts_set = new_pts_set
            current_line_keys.add(key)
            
            if len(current_pts_set) == 16:
                break
                
        if len(current_pts_set) == 16:
            pts = list(current_pts_set)
            score = count_4_lines(pts)
            if score > best_score:
                best_score = score
                best_points = pts
                print(f"Greedy found: {score}")
                
    # Local search on the best found
    if best_points:
        # Try to improve by swapping a point
        for _ in range(1000):
            if _time_left() < 1.0:
                break
            p = random.choice(best_points)
            # Try to move p to a new location
            # New location must be integer, |x|,|y|<=1000
            # And must not create 5-collinear
            # And must increase or maintain score
            
            # This is a bit weak. Let's stick to the greedy result.
            pass

    if best_points is None:
        return [[x, y] for x, y in seed_points]
        
    return [[x, y] for x, y in best_points]
