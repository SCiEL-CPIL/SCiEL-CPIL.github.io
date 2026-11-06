import random
import time
from collections import defaultdict

def count_four_lines(points):
    """
    Counts the number of lines containing exactly 4 points.
    Returns -1 if any line contains 5 or more points.
    """
    line_counts = defaultdict(int)
    n = len(points)
    
    # To avoid counting the same line multiple times, we can use a canonical representation.
    # A line is defined by a pair of points (i, j) with i < j.
    # However, multiple pairs define the same line.
    # We can use a set of lines.
    
    lines = set()
    # Map from line identifier to count of points on that line
    line_point_counts = defaultdict(int)
    
    # This O(N^3) approach is too slow for 16 points in a tight loop?
    # N=16, N^3 = 4096. 100,000 iterations * 4096 = 400M ops. Might be tight in Python.
    # We need a faster evaluation or fewer iterations.
    
    # Alternative: Precompute line keys for pairs?
    # For each pair (i, j), compute the line key.
    # Then, for each line key, count how many points lie on it.
    
    # Let's optimize the evaluation.
    # 1. For each pair (i, j), compute the normalized line equation (A, B, C) for Ax + By = C.
    #    Normalize by GCD and signs.
    # 2. Use a dictionary: key -> list of point indices on this line.
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        # If points are identical, skip (but points are distinct)
        # Normal vector (A, B) is (dy, -dx) or (dy, -dx)
        # Ax + By = C
        # dy * x - dx * y = dy * x1 - dx * y1
        A = dy
        B = -dx
        C = dy * x1 - dx * y1
        
        # Normalize (A, B, C)
        # Find GCD of A, B, C
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return abs(a)
            
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
            
        # Ensure unique representation
        # Make the first non-zero of A, B positive
        if A < 0 or (A == 0 and B < 0):
            A = -A
            B = -B
            C = -C
        elif A == 0 and B == 0:
            # Should not happen for distinct points
            pass
            
        return (A, B, C)

    # Build line map
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            line_points[key].add(i)
            line_points[key].add(j)
            
    # Now, for each line, we have a set of points that define it via pairs.
    # We need to count how many of the 16 points are actually on the line.
    # The set `line_points[key]` contains points that form pairs on that line.
    # But if 4 points are collinear, there are 6 pairs. The set will contain all 4 indices.
    # If 5 points are collinear, the set will contain all 5 indices.
    # So, the size of the set `line_points[key]` is exactly the number of collinear points for that line.
    
    count_4 = 0
    for key, pts in line_points.items():
        size = len(pts)
        if size == 4:
            count_4 += 1
        elif size >= 5:
            return -1 # Invalid
            
    return count_4

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Initial configuration: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    # Shift to center or keep small. Small is better for finding dense structures.
    # Let's keep it in [0, 3]x[0, 3] initially, but allow moves to [-10, 10].
    
    best_score = count_four_lines(points)
    best_points = [p[:] for p in points]
    
    # Local Search
    # We will use a simple hill climbing with random restarts or just long runs.
    # Given the small N, we can afford many iterations.
    
    # To speed up evaluation, we can cache line keys for pairs?
    # But points change.
    
    # Let's try a specific known better configuration first if possible.
    # A common improvement is to break the symmetry of the grid.
    
    # We will run a local search.
    # Move: Pick a random point, move it to a random integer coordinate in [-10, 10].
    
    iterations = 0
    while time.time() < deadline:
        # Check time more frequently
        if time.time() > deadline:
            break
            
        iterations += 1
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # New position: random in a neighborhood or global?
        # Global search in [-10, 10] is 21x21=441 options.
        # Neighborhood is faster to converge.
        # Let's use a mix: 80% local, 20% global.
        
        if random.random() < 0.8:
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
        else:
            new_x = random.randint(-10, 10)
            new_y = random.randint(-10, 10)
            
        # Check bounds
        if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
            continue
            
        # Check distinctness
        if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(points) if i != idx):
            continue
            
        # Apply move
        points[idx] = [new_x, new_y]
        
        # Evaluate
        score = count_four_lines(points)
        
        # Acceptance
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in points]
        elif score == best_score and random.random() < 0.1:
            # Plateau move
            pass
        else:
            # Reject, revert
            points[idx] = [old_x, old_y]
            
        # Occasionally, if stuck, do a random restart of the current best?
        # Or just keep going.
        
    return best_points

# Test locally
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {count_four_lines(pts)}")
    print(f"Points: {pts}")
