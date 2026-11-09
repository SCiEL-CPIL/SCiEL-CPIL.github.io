"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    # Known configuration: 3x4 grid has 6 ordinary lines.
    # The theoretical minimum for 12 points is 6.
    # Let's verify if a 3x4 grid indeed has 6 ordinary lines.
    # Points: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2)
    
    # Let's count ordinary lines for the 3x4 grid.
    # Total pairs: C(12, 2) = 66.
    # Lines with >= 3 points:
    # - 4 horizontal lines (y=0,1,2,3? No, y goes 0..2, x goes 0..3)
    #   Wait, the code says: for x in range(4) for y in range(3)
    #   So x in {0,1,2,3}, y in {0,1,2}.
    #   Horizontal lines: y=0 (4 pts), y=1 (4 pts), y=2 (4 pts). -> 3 lines with 4 pts.
    #   Vertical lines: x=0 (3 pts), x=1 (3 pts), x=2 (3 pts), x=3 (3 pts). -> 4 lines with 3 pts.
    #   Diagonals:
    #     Slope 1: (0,0)-(1,1)-(2,2) [3 pts], (1,0)-(2,1)-(3,2) [3 pts].
    #     Slope -1: (0,2)-(1,1)-(2,0) [3 pts], (1,2)-(2,1)-(3,0) [3 pts].
    #     Slope 1/2: (0,0)-(2,1)? No, (0,0), (2,1) is slope 0.5. (0,0), (2,1), (4,2) no.
    #     Let's list all lines with >= 3 points.
    #     1. y=0: (0,0), (1,0), (2,0), (3,0)
    #     2. y=1: (0,1), (1,1), (2,1), (3,1)
    #     3. y=2: (0,2), (1,2), (2,2), (3,2)
    #     4. x=0: (0,0), (0,1), (0,2)
    #     5. x=1: (1,0), (1,1), (1,2)
    #     6. x=2: (2,0), (2,1), (2,2)
    #     7. x=3: (3,0), (3,1), (3,2)
    #     8. (0,0)-(1,1)-(2,2)
    #     9. (1,0)-(2,1)-(3,2)
    #     10. (0,2)-(1,1)-(2,0)
    #     11. (1,2)-(2,1)-(3,0)
    #     12. (0,0)-(2,1)? No. (0,1)-(2,2)? No.
    #     Are there any other lines with 3 points?
    #     Slope 1/2: (0,0), (2,1), (4,2) - out of range.
    #     Slope 2: (0,0), (1,2) - only 2 points? (0,0), (1,2). Next would be (2,4) out.
    #     So only 11 lines with >= 3 points.
    #     Let's count pairs covered by these lines.
    #     Lines with 4 pts: 3 lines * C(4,2) = 3 * 6 = 18 pairs.
    #     Lines with 3 pts: 8 lines * C(3,2) = 8 * 3 = 24 pairs.
    #     Total pairs on lines with >= 3 pts = 18 + 24 = 42.
    #     Total pairs = 66.
    #     Ordinary lines = Total pairs - Pairs on non-ordinary lines?
    #     No, an ordinary line contains exactly 2 points.
    #     Number of ordinary lines = Number of pairs that do NOT lie on a line with >= 3 points.
    #     Wait, if a pair lies on a line with 3 points, it's not an ordinary line.
    #     If a pair lies on a line with 4 points, it's not an ordinary line.
    #     Are there any pairs that lie on multiple lines with >= 3 points?
    #     No, two distinct points determine a unique line.
    #     So, ordinary lines = 66 - 42 = 24.
    #     
    #     Wait, the task says "Over the reals the minimum for 12 points is 6".
    #     The 3x4 grid has 24 ordinary lines. That's far from 6.
    #     I need a configuration with fewer ordinary lines.
    #     
    #     Known configurations with few ordinary lines:
    #     - 9 points in 3x3 grid: 3 lines of 4? No.
    #     - 12 points: Maybe a subset of a larger grid or a specific structure.
    #     - The Sylvester-Gallai theorem says there is always at least one ordinary line.
    #     - Minimum number of ordinary lines for n points:
    #       n=3: 3 (triangle)
    #       n=4: 3 (triangle + center? No. 3x3 grid minus 5? No.)
    #       Actually, for n=12, the minimum is known to be 6.
    #       One such configuration is the "3x4 grid" but wait, I calculated 24.
    #       Let me re-read. "Minimize the number of lines containing exactly two of the points".
    #       
    #       Let's check a different configuration.
    #       What about points on a conic? No, that gives many ordinary lines.
    #       What about a grid with some points removed?
    #       
    #       Let's try to find a configuration with 6 ordinary lines.
    #       If we have 6 ordinary lines, they account for 6 pairs.
    #       The remaining 60 pairs must be on lines with >= 3 points.
    #       This requires a very dense structure.
    #       
    #       Consider the 3x3 grid (9 points). It has 8 lines with 3 points (3 horiz, 3 vert, 2 diag).
    #       Pairs on these lines: 8 * 3 = 24.
    #       Total pairs: C(9,2) = 36.
    #       Ordinary lines: 36 - 24 = 12.
    #       
    #       If we add 3 more points to the 3x3 grid to make 12 points, can we reduce the ordinary lines?
    #       
    #       Let's try a 4x3 grid again. I calculated 24.
    #       Maybe a 4x4 grid minus 4 points?
    #       
    #       Let's look for known minimal configurations.
    #       For n=12, min ordinary lines = 6.
    #       One example is the "Böröczky" configuration or similar.
    #       
    #       Let's try a simple search for small integer coordinates.
    #       Since we have 120s, we can do a local search.
    #       Start with a 3x4 grid (24 ordinary lines) and try to swap points to reduce the count.
    #       
    #       However, the prompt says "Over the reals the minimum for 12 points is 6; whether integer (rational) coordinates reach 6 is not documented".
    #       This suggests 6 might not be achievable with integers, or it is.
    #       If 6 is not achievable, what is the best integer configuration?
    #       
    #       Let's try to improve the 3x4 grid.
    #       Or maybe a 4x4 grid with 4 points removed carefully.
    #       
    #       Let's write a helper to count ordinary lines.
    
    def count_ordinary(points):
        n = len(points)
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Normalize line equation
                # ax + by + c = 0
                # a = dy, b = -dx
                # c = -a*x1 - b*y1
                a = dy
                b = -dx
                c = -a*x1 - b*y1
                # Normalize (a, b, c)
                # Find gcd
                import math
                g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
                if g > 0:
                    a //= g
                    b //= g
                    c //= g
                # Ensure unique representation
                # a > 0 or (a == 0 and b > 0) or (a == 0 and b == 0 and c > 0)
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a = -a
                    b = -b
                    c = -c
                key = (a, b, c)
                if key not in lines:
                    lines[key] = []
                lines[key].append(i)
        
        ordinary = 0
        for key, pts in lines.items():
            if len(pts) == 2:
                ordinary += 1
        return ordinary

    # Let's test the 3x4 grid
    pts_34 = [[x, y] for x in range(4) for y in range(3)]
    print("3x4 grid ordinary lines:", count_ordinary(pts_34)) # Should be 24
    
    # Let's try a 4x4 grid minus 4 corners?
    # (0,0), (0,1), (0,2), (0,3)
    # (1,0), (1,1), (1,2), (1,3)
    # (2,0), (2,1), (2,2), (2,3)
    # (3,0), (3,1), (3,2), (3,3)
    # Remove (0,0), (0,3), (3,0), (3,3)
    pts_44 = []
    for x in range(4):
        for y in range(4):
            if not (x in [0,3] and y in [0,3]):
                pts_44.append([x, y])
    print("4x4 minus corners ordinary lines:", count_ordinary(pts_44))
    
    # Let's try a 3x3 grid + 3 points on the diagonals?
    # Or maybe a different structure.
    
    # Since the search space is large, let's do a random local search.
    import random
    import copy
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    best_pts = pts_34
    best_score = count_ordinary(best_pts)
    
    current_pts = [p[:] for p in best_pts]
    current_score = best_score
    
    # Random restarts
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Start from a random configuration or best found
        if restart == 0:
            curr = [p[:] for p in best_pts]
        else:
            # Random points in a box
            curr = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(12)]
            # Ensure distinct
            while len(set(map(tuple, curr))) < 12:
                curr[random.randint(0, 11)] = [random.randint(-5, 5), random.randint(-5, 5)]
        
        curr_score = count_ordinary(curr)
        
        # Local search
        for step in range(500):
            if time.time() > deadline:
                break
            
            i = random.randint(0, 11)
            old_x, old_y = curr[i]
            # Try moving point i
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 100 or abs(new_y) > 100:
                        continue
                    
                    # Check if new point collides with existing
                    new_point = [new_x, new_y]
                    collides = False
                    for j in range(12):
                        if j != i and curr[j] == new_point:
                            collides = True
                            break
                    if collides:
                        continue
                    
                    curr[i] = new_point
                    new_score = count_ordinary(curr)
                    
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_pts = [p[:] for p in curr]
                    else:
                        curr[i] = [old_x, old_y]
        
        if time.time() > deadline:
            break

    return best_pts
# EVOLVE-BLOCK-END
