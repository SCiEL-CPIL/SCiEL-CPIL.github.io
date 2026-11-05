import time
import random
from collections import defaultdict
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for 120s limit

    def get_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
        # Sign convention: first non-zero positive
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def compute_line_counts(points):
        """Compute line counts for all pairs of points."""
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        return line_counts

    def is_valid(points):
        """Check distinctness, bounds, and no 4 collinear."""
        if len(points) != 20:
            return False
        if len(set(tuple(p) for p in points)) != 20:
            return False
        for p in points:
            if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                return False
        
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        
        for line, count in line_counts.items():
            if count >= 4:
                return False
        return True

    def calc_score(points):
        """
        Compute the largest subset with no three collinear using branch and bound.
        """
        n = len(points)
        
        # Build for each point i, the set of points j that share a line with i and some other point
        # i.e., for each i, find all j such that there exists k != i,j with i,j,k collinear
        # This gives us the "conflict" graph: edge (i,j) if i and j cannot both be in a no-3-collinear set
        # Wait, that's not quite right. Two points i,j conflict if adding both would create a triple.
        # Actually, i and j conflict if there exists k such that {i,j,k} is collinear.
        # But that's not sufficient either - we need the full hypergraph.
        
        # Better approach: For each pair (i,j), check if they are on a line with a third point.
        # If yes, then we cannot include both i and j in our subset UNLESS we exclude the third point.
        # This is a 3-uniform hypergraph independence problem.
        
        # Let's use branch and bound on the hypergraph.
        # For each point i, precompute which other points are "conflicting" with i
        # Two points i,j conflict if there's a third point k on line(i,j)
        
        # Actually, let me think differently. 
        # A set S is valid (no 3 collinear) iff for every line, |S ∩ line| <= 2.
        # So we want max |S| such that for every line L, |S ∩ L| <= 2.
        
        # Precompute all lines and which points are on them
        line_to_points = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                if line not in line_to_points:
                    line_to_points[line] = set()
                line_to_points[line].add(i)
                line_to_points[line].add(j)
        
        # For each line, get the list of points
        lines = []
        for line, pts in line_to_points.items():
            if len(pts) >= 3:
                lines.append((line, sorted(pts)))
        
        # For each point, which lines does it appear in (with >= 3 points total)?
        point_lines = defaultdict(set)
        for line, pts in lines:
            for p in pts:
                point_lines[p].add(line)
        
        # Branch and bound
        # State: current set of included points, current set of excluded points
        # We process points one by one (in some order)
        
        # Order points by "conflict degree" - points that are on more lines should be decided first
        conflict_degree = [len(point_lines[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: -conflict_degree[i])
        
        best = 0
        
        def bb(idx, included, excluded):
            nonlocal best
            
            # Pruning: even if we include all remaining points, can we beat best?
            remaining_count = n - idx
            if len(included) + remaining_count <= best:
                return
            
            if idx == n:
                if len(included) > best:
                    best = len(included)
                return
            
            i = order[idx]
            
            # Option 1: Exclude i
            bb(idx + 1, included, excluded | {i})
            
            # Option 2: Include i (if it doesn't violate constraints)
            # Check: for each line containing i, count how many points from included are on that line
            can_include = True
            for line in point_lines[i]:
                # Count included points on this line
                count = 0
                for p in included:
                    if p in line_to_points[line]:
                        count += 1
                        if count >= 2:
                            can_include = False
                            break
                if not can_include:
                    break
            
            if can_include:
                bb(idx + 1, included | {i}, excluded)
        
        bb(0, set(), set())
        return best

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
    
    # Initial configuration: Random
    current_points = None
    for _ in range(100):
        pts = []
        while len(pts) < 20:
            p = random_point()
            if p not in pts:
                pts.append(p)
        if is_valid(pts):
            current_points = pts
            break
    
    if current_points is None:
        # Fallback to parabola if random fails (unlikely)
        current_points = [[x, x * x] for x in range(-10, 10)]
        
    current_score = calc_score(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Local Search
    # Use a faster validity check and score computation
    def fast_is_valid(points):
        """Fast validity check assuming distinctness and bounds already verified."""
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        for count in line_counts.values():
            if count >= 4:
                return False
        return True
    
    def fast_score(points):
        """Faster score computation using the same branch and bound but with early termination."""
        return calc_score(points)
    
    # Precompute lines for current points
    def get_line_counts(points):
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        return line_counts
    
    # Local search with time management
    search_start = time.time()
    
    while time.time() < deadline - 5.0:  # Leave 5s buffer
        # Pick a random point to move
        idx = random.randint(0, 19)
        
        improved = False
        
        # Try to place the point on a line with exactly 2 other points (to create a triple)
        # This helps minimize the no-3-collinear subset size
        tried_moves = 0
        max_moves = 50
        
        for i in range(20):
            if i == idx or tried_moves >= max_moves:
                continue
            for j in range(i + 1, 20):
                if j == idx or tried_moves >= max_moves:
                    continue
                
                p1 = current_points[i]
                p2 = current_points[j]
                line = get_line(p1, p2)
                
                # Check how many other points are on this line
                cnt = 0
                for k in range(20):
                    if k != i and k != j and k != idx:
                        if get_line(p1, current_points[k]) == line:
                            cnt += 1
                
                if cnt == 0:
                    # Place a new point on this line to form a triple
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    
                    # Try various positions on the line
                    for t in range(-20, 21):
                        if t == 0 or t == 1:
                            continue
                        nx = p1[0] + t * dx
                        ny = p1[1] + t * dy
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            new_candidate = [nx, ny]
                            if new_candidate not in current_points:
                                test_points = current_points[:]
                                test_points[idx] = new_candidate
                                if fast_is_valid(test_points):
                                    test_score = fast_score(test_points)
                                    if test_score < current_score:
                                        current_points = test_points
                                        current_score = test_score
                                        if current_score < best_score:
                                            best_score = current_score
                                            best_points = [p[:] for p in current_points]
                                        improved = True
                                        tried_moves += 1
                                        break
                                tried_moves += 1
                        if tried_moves >= max_moves:
                            break
                elif cnt == 1:
                    # Line has 3 points already (including p1, p2, and one other)
                    # Moving to this line would make 4 collinear - invalid
                    pass
            if improved and tried_moves >= max_moves:
                break
        
        if not improved:
            # Try random moves
            for _ in range(5):
                new_p = random_point()
                if new_p not in current_points:
                    test_points = current_points[:]
                    test_points[idx] = new_p
                    if fast_is_valid(test_points):
                        test_score = fast_score(test_points)
                        if test_score < current_score:
                            current_points = test_points
                            current_score = test_score
                            if current_score < best_score:
                                best_score = current_score
                                best_points = [p[:] for p in current_points]
                            improved = True
                            break
    
    # Final check
    if is_valid(best_points):
        return best_points
    else:
        # Fallback: use a known valid configuration
        # Points on a parabola y = x^2 for x in -10..9 have no 4 collinear
        return [[x, x * x] for x in range(-10, 10)]
