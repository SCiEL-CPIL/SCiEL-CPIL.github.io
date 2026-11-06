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
    
    # AG(2,4) embedding: 16 points in the affine plane over GF(4)
    # GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
    # Points are (x, y) where x, y in GF(4)
    # Lines in AG(2,4): for each direction (a,b) not both 0, and each intercept c,
    # the line ax + by = c has exactly 4 points
    # Total lines: 5 directions * 4 intercepts = 20 lines, each with 4 points
    
    # Map GF(4) elements to integers:
    # 0 -> 0
    # 1 -> 1  
    # alpha -> 2
    # alpha+1 -> 3
    # This gives a 4x4 grid, but we need to verify no 5 collinear
    
    # Actually, the simple 4x4 grid has only 10 four-point lines (4 rows + 4 cols + 2 diagonals)
    # We need a different embedding that preserves the AG(2,4) line structure
    
    # Key insight: AG(2,4) lines have a specific structure. We need to find integer coordinates
    # where the 20 lines of AG(2,4) map to 20 distinct integer lines, each containing exactly 4 points.
    
    # One approach: use the fact that AG(2,4) can be embedded in the plane using a field representation.
    # GF(4) can be represented as Z_2[i] where i^2 = i + 1
    # Elements: 0, 1, i, i+1
    # Map to integers: 0->0, 1->1, i->2, i+1->3
    
    # But this just gives the 4x4 grid. The issue is that the 4x4 grid's collinearities
    # don't match AG(2,4)'s line structure.
    
    # Alternative: use a more sophisticated mapping. The AG(2,4) configuration has 16 points
    # and 20 lines. We need to find 16 integer points such that exactly 20 lines contain exactly 4 points.
    
    # Let me try a known construction or use local search starting from a good initial config.
    
    # Start with the 4x4 grid and improve
    points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = count_four_lines(points)
    best_points = [p[:] for p in points]
    
    # Local Search with better move generation
    iterations = 0
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        iterations += 1
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Generate candidate moves: try to create new four-point lines
        # Strategy: pick 3 other points, find the 4th point that would make a line
        # through them, and move idx there
        
        candidates = set()
        
        # For each pair of other points, compute the line and find where the 3rd point
        # from another pair would put the 4th point
        for i in range(16):
            if i == idx:
                continue
            for j in range(i + 1, 16):
                if j == idx:
                    continue
                # Line through points[i] and points[j]
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                
                # For each other point k, check if it's on this line, and if so,
                # find a 4th point that would make a 4-point line
                for k in range(j + 1, 16):
                    if k == idx:
                        continue
                    x3, y3 = points[k]
                    # Check if (x3, y3) is on line through (x1,y1) and (x2,y2)
                    if dx != 0:
                        # y - y1 = (dy/dx)(x - x1)
                        # dy * (x - x1) = dx * (y - y1)
                        if dy * (x3 - x1) != dx * (y3 - y1):
                            continue
                    else:
                        if x3 != x1:
                            continue
                    
                    # (x1,y1), (x2,y2), (x3,y3) are collinear
                    # Find a 4th point that would be on this line and not already used
                    # The line is parametric: (x1 + t*dx, y1 + t*dy) for t in some set
                    # We need to find an integer point on this line not in our set
                    # and not equal to our 3 points
                    
                    # Try nearby points on the line
                    for t in range(-10, 11):
                        if t == 0 or t == 1:  # Skip existing points (assuming they're at t=0,1)
                            # Actually, we don't know the parameterization
                            pass
                        
                    # Simpler: just try random points near the line
                    # Or: compute the line equation and find integer solutions
                    if dx != 0:
                        slope_num = dy
                        slope_den = dx
                        # Line: y = (slope_num/slope_den)(x - x1) + y1
                        # y * slope_den = slope_num * (x - x1) + y1 * slope_den
                        # slope_den * y - slope_num * x = -slope_num * x1 + y1 * slope_den
                        A = -slope_num
                        B = slope_den
                        C = -slope_num * x1 + y1 * slope_den
                        
                        # Find integer solutions to Ax + By = C near the current point
                        # Try x values near old_x
                        for nx in range(old_x - 5, old_x + 6):
                            # By = C - Ax
                            rhs = C - A * nx
                            if B != 0 and rhs % B == 0:
                                ny = rhs // B
                                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                                    # Check distinctness
                                    if not any(p[0] == nx and p[1] == ny for ii, p in enumerate(points) if ii != idx):
                                        candidates.add((nx, ny))
                    else:
                        # Vertical line: x = x1
                        for ny in range(old_y - 5, old_y + 6):
                            if -1000 <= x1 <= 1000 and -1000 <= ny <= 1000:
                                if not any(p[0] == x1 and p[1] == ny for ii, p in enumerate(points) if ii != idx):
                                    candidates.add((x1, ny))
        
        # Also add some random candidates
        if random.random() < 0.3:
            for _ in range(5):
                nx = random.randint(-10, 10)
                ny = random.randint(-10, 10)
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    if not any(p[0] == nx and p[1] == ny for ii, p in enumerate(points) if ii != idx):
                        candidates.add((nx, ny))
        
        if not candidates:
            # Fallback: small random move
            nx = old_x + random.randint(-2, 2)
            ny = old_y + random.randint(-2, 2)
            if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                if not any(p[0] == nx and p[1] == ny for ii, p in enumerate(points) if ii != idx):
                    candidates.add((nx, ny))
        
        if not candidates:
            continue
        
        # Pick a random candidate
        nx, ny = random.choice(list(candidates))
        
        # Apply move
        points[idx] = [nx, ny]
        
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
        
        # Periodically try to reset to best if stuck
        if iterations % 1000 == 0:
            # Check if we've been stuck
            pass
    
    return best_points

# Test locally
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {count_four_lines(pts)}")
    print(f"Points: {pts}")
