"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points. Fast O(n^3) using sets."""
    n = len(points)
    if n < 3:
        return 0
    
    # Map line key to count of points on it
    # Key: (normalized dx, normalized dy, c)
    # We only need to count lines with exactly 2 points.
    # A line with k points contributes to C(k,2) pairs.
    # If we just count distinct lines with exactly 2 points, we can do:
    # 1. Iterate pairs (i, j), compute line key.
    # 2. Use a dict to store the set of points on that line.
    # 3. At the end, count keys where len(set) == 2.
    
    # This is O(n^3) worst case but for n=12 it's 12*11/2 = 66 pairs.
    # Checking if a point is on a line is O(1).
    # So we can just build the sets.
    
    lines = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            # Line equation: dy*x - dx*y = c
            # c = dy*x1 - dx*y1
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary += 1
            
    return ordinary

def _pappus_configuration():
    """
    Construct 12 points based on the Pappus Hexagon Theorem configuration.
    This configuration has 16 lines:
    - 3 lines with 4 points each (the main axes)
    - 12 lines with 3 points each (the sides of the hexagon and diagonals)
    - Wait, the Pappus configuration (3_4 4_3) has:
      12 points, 16 lines.
      Each point is on 4 lines.
      Each line has 3 points.
      Total incidences: 12*4 = 48, 16*3 = 48.
      If every line has exactly 3 points, then there are 0 ordinary lines?
      No, the minimum is 6. Let's check the structure.
      
      Actually, the Pappus configuration is a (3_12, 3_16) design?
      No, it's a (4_3, 3_16) design?
      Let's look up "Pappus configuration 12 points".
      It consists of 12 points and 16 lines.
      3 lines contain 4 points.
      12 lines contain 3 points.
      Incidences: 3*4 + 12*3 = 12 + 36 = 48.
      Points: 12. Average lines per point = 48/12 = 4.
      So each point is on 4 lines.
      
      If 3 lines have 4 points, and 12 lines have 3 points:
      Are there any lines with 2 points?
      Total pairs of points = C(12, 2) = 66.
      Pairs covered by lines:
      3 lines * C(4,2) + 12 lines * C(3,2) = 3*6 + 12*3 = 18 + 36 = 54.
      Remaining pairs = 66 - 54 = 12.
      These 12 pairs must form lines with exactly 2 points (ordinary lines).
      So there are exactly 12/2 = 6 ordinary lines?
      No, each ordinary line contains exactly 1 pair.
      So there are 12 ordinary lines?
      Wait. If a pair is not on a line with 3+ points, it defines an ordinary line.
      The number of ordinary lines is the number of such pairs.
      So 12 ordinary lines?
      
      Let's re-evaluate the lower bound.
      Sylvester-Gallai says at least 1 ordinary line.
      For 12 points, the minimum is known to be 6?
      Let's check the "orchard problem" or "ordinary lines problem".
      For n=12, the minimum number of ordinary lines is 6.
      This is achieved by the Pappus configuration?
      
      Let's check the math again.
      If min is 6, then 6 lines * 1 pair/line = 6 pairs are "ordinary".
      Total pairs = 66.
      Pairs in non-ordinary lines = 66 - 6 = 60.
      If we have lines of size 3 and 4:
      Let x be number of lines with 4 points.
      Let y be number of lines with 3 points.
      Let z be number of lines with 2 points (ordinary).
      
      Sum of C(k, 2) for all lines = 66.
      Sum of k for all lines = Sum of (lines per point) = 12 * L, where L is avg lines per point.
      
      In Pappus:
      3 lines of size 4: 3 * 6 = 18 pairs.
      12 lines of size 3: 12 * 3 = 36 pairs.
      Total pairs covered = 54.
      Remaining pairs = 66 - 54 = 12.
      So there are 12 ordinary lines?
      
      Wait, the lower bound for 12 points is 6.
      Which configuration achieves 6?
      
      Maybe the Pappus configuration has more lines?
      Or maybe my count of "remaining pairs" assumes that every remaining pair is on a distinct ordinary line.
      Yes, if two ordinary lines shared a pair, they would be the same line.
      So 12 pairs = 12 ordinary lines.
      
      So Pappus gives 12 ordinary lines?
      If the lower bound is 6, Pappus is not optimal for minimizing ordinary lines?
      
      Let's check the "minimum number of ordinary lines" table.
      n=3: 3
      n=4: 4 (or 3 if 3 collinear? No, if 3 collinear, the 4th point forms 3 ordinary lines with the 3. Total 3. Wait.
      Points A, B, C collinear. D.
      Lines: ABC (3 pts). AD, BD, CD (2 pts each).
      Ordinary lines: AD, BD, CD. Count = 3.
      So n=4 min is 3?
      
      n=5: 
      If 4 collinear + 1:
      Line of 4. 4 ordinary lines from the 5th point.
      Total 4.
      If 3 collinear + 2:
      Line of 3.
      If the 2 are on a line with 1 of the 3?
      Let's try a grid 2x2 + 1 center.
      (0,0), (1,0), (0,1), (1,1), (0.5, 0.5).
      Lines:
      y=0: 2 pts
      y=1: 2 pts
      x=0: 2 pts
      x=1: 2 pts
      y=x: 3 pts (0,0), (0.5,0.5), (1,1)
      y=1-x: 3 pts (1,0), (0.5,0.5), (0,1)
      Ordinary lines:
      (0,0)-(1,1) is part of y=x.
      (1,0)-(0,1) is part of y=1-x.
      (0,0)-(1,0) ordinary? Yes, y=0 has 2 pts.
      (0,1)-(1,1) ordinary? Yes, y=1 has 2 pts.
      (0,0)-(0,1) ordinary? Yes, x=0 has 2 pts.
      (1,0)-(1,1) ordinary? Yes, x=1 has 2 pts.
      (0,0)-(0.5,0.5) part of y=x.
      (1,1)-(0.5,0.5) part of y=x.
      (1,0)-(0.5,0.5) part of y=1-x.
      (0,1)-(0.5,0.5) part of y=1-x.
      (0,0)-(1,0) ...
      Total ordinary lines: 4 (sides of square).
      So n=5 min is 4?
      
      The task says: "Over the reals the minimum for 12 points is 6".
      So I need to find a configuration with 6 ordinary lines.
      
      If Pappus gives 12, it's not the one.
      
      What configuration gives 6?
      6 ordinary lines means 6 pairs are "isolated".
      60 pairs are in lines of size >= 3.
      
      Let's try to construct one.
      Maybe 3 lines of 4 points intersecting?
      If we have 3 lines of 4 points, and they are in "general position" (no two parallel, no three concurrent at a point that is not an intersection of the lines), they form a triangle.
      Vertices: 3 intersection points.
      Other points on lines: 3 lines * (4-2) = 6 points.
      Total points = 3 + 6 = 9.
      We need 12.
      
      Let's look at the "Pappus" inspiration again.
      Maybe I should just use a strong local search.
      The current program does local search but starts from bad seeds.
      I will start from a "Grid" and use a very aggressive local search with a fast counter.
      
      Strategy:
      1. Start with a 3x4 grid.
      2. Use a fast O(n^3) counter.
      3. Perform simulated annealing or random restarts with local search.
      4. Since n=12 is small, we can afford to be thorough.
      
      Let's implement a robust local search.
    """
    pass

def _fast_count(points):
    """
    Optimized count for n=12.
    Precompute all pair lines? No, just use the dict method which is very fast for n=12.
    """
    n = len(points)
    lines = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            if key not in lines:
                lines[key] = 0
            # We just need to know if it's 2 or >2.
            # Actually, we need to count lines with EXACTLY 2 points.
            # So we need to know the set size.
            # Using a set is safer.
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    count = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            count += 1
    return count

def _local_search(points, time_limit=5.0):
    """
    Local search to minimize ordinary lines.
    Moves a point to a random nearby location or a location on a line through two other points.
    """
    start_time = time.time()
    current = [p[:] for p in points]
    current_score = _fast_count(current)
    best = current[:]
    best_score = current_score
    
    n = 12
    
    while time.time() - start_time < time_limit:
        # Try moving each point
        for i in range(n):
            # Generate candidates
            candidates = set()
            
            # 1. Random small perturbations
            for _ in range(5):
                nx = current[i][0] + random.randint(-5, 5)
                ny = current[i][1] + random.randint(-5, 5)
                candidates.add((nx, ny))
                
            # 2. Points on lines defined by other pairs
            # For each pair (j, k) with j, k != i, extend the line
            for j in range(n):
                if j == i: continue
                for k in range(j + 1, n):
                    if k == i: continue
                    x1, y1 = current[j]
                    x2, y2 = current[k]
                    dx = x2 - x1
                    dy = y2 - y1
                    # Add points at t=1, -1, 2, -2 relative to j
                    for t in [-2, -1, 1, 2]:
                        nx = x1 + t * dx
                        ny = y1 + t * dy
                        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                            candidates.add((nx, ny))
                            
            # Test candidates
            for cand in candidates:
                if cand[0] == current[i][0] and cand[1] == current[i][1]:
                    continue
                # Check distinctness
                valid = True
                for j in range(n):
                    if j != i and current[j][0] == cand[0] and current[j][1] == cand[1]:
                        valid = False
                        break
                if not valid:
                    continue
                
                old = current[i][:]
                current[i] = list(cand)
                score = _fast_count(current)
                
                if score < current_score:
                    current_score = score
                    if score < best_score:
                        best_score = score
                        best = [p[:] for p in current]
                else:
                    current[i] = old
                    
    return best, best_score

def construct():
    """
    Construct 12 integer points with minimized ordinary lines.
    Uses multiple random restarts with local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = float('inf')
    
    # Seed 1: 3x4 Grid
    # This is a strong seed.
    grid_points = []
    for i in range(3):
        for j in range(4):
            grid_points.append([i * 10, j * 10])
            
    # Seed 2: Pappus-like (3 lines of 4)
    # Line 1: y=0, x=0,1,2,3
    # Line 2: x=0, y=0,1,2,3
    # Line 3: y=x, x=0,1,2,3 -> (0,0),(1,1),(2,2),(3,3)
    # Intersection (0,0) is shared.
    # Points:
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (0,2), (0,3)
    # (1,1), (2,2), (3,3)
    # Total 11 points. Need 1 more.
    # Add (3,1)?
    pappus_points = [
        [0,0], [1,0], [2,0], [3,0],
        [0,1], [0,2], [0,3],
        [1,1], [2,2], [3,3],
        [3,1]
    ]
    
    # Seed 3: Random
    def gen_random():
        pts = []
        seen = set()
        while len(pts) < 12:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x,y) not in seen:
                seen.add((x,y))
                pts.append([x,y])
        return pts

    seeds = [grid_points, pappus_points]
    
    # Add random seeds
    for _ in range(5):
        seeds.append(gen_random())
        
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        # Scale up to avoid small number issues? No, integers are fine.
        # But local search works better with some spread.
        # Let's just run local search on the seed.
        
        # To diversify, we can slightly perturb the seed before local search
        # But local search itself does perturbations.
        
        # Run local search
        # Allocate time based on remaining time
        remaining = deadline - time.time()
        t_limit = min(20.0, remaining * 0.8)
        
        pts, score = _local_search(seed, time_limit=t_limit)
        
        if score < best_score:
            best_score = score
            best_points = pts
            
        # If we found 6, we can stop early? 
        # The minimum is 6. If we find 6, it's optimal (assuming integer coords can reach it).
        if best_score == 6:
            break
            
    # If we didn't find anything good, fallback to grid
    if best_points is None:
        best_points = grid_points
        best_score = _fast_count(best_points)
        
    # Final validation and cleanup
    # Ensure 12 distinct points
    seen = set()
    unique_points = []
    for p in best_points:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
            
    while len(unique_points) < 12:
        for i in range(1, 1000):
            for j in range(1, 1000):
                p = [i, j]
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    unique_points.append(p)
                    break
            if len(unique_points) >= 12:
                break
                
    return unique_points[:12]
# EVOLVE-BLOCK-END
