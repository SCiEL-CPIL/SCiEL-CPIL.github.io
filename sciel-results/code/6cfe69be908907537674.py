import random
import time
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def gcd3(a, b, c):
        return gcd(gcd(abs(a), abs(b)), abs(c))
    
    def check_no_four_collinear(points):
        n = len(points)
        # Map line to count of points
        # Line represented by (A, B, C) where Ax + By + C = 0, normalized
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                # A = y1 - y2, B = x2 - x1, C = x1*y2 - x2*y1
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                # Normalize
                if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
                    A, B, C = -A, -B, -C
                g = gcd3(A, B, C)
                if g > 0:
                    A //= g
                    B //= g
                    C //= g
                key = (A, B, C)
                if key in lines:
                    lines[key] += 1
                else:
                    lines[key] = 1
        
        for count in lines.values():
            if count >= 4:
                return False
        return True

    def count_triples(points):
        n = len(points)
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
                    A, B, C = -A, -B, -C
                g = gcd3(A, B, C)
                if g > 0:
                    A //= g
                    B //= g
                    C //= g
                key = (A, B, C)
                if key in lines:
                    lines[key].add(i)
                    lines[key].add(j)
                else:
                    lines[key] = {i, j}
        
        triple_count = 0
        for pts in lines.values():
            k = len(pts)
            if k >= 3:
                # C(k,3)
                triple_count += k * (k - 1) * (k - 2) // 6
        return triple_count

    def score(points):
        if not check_no_four_collinear(points):
            return 1000
        # Heuristic: more triples generally means a larger hitting set is needed?
        # Actually, we want to MINIMIZE the hitting set.
        # A lower bound for hitting set is max(ceil(k/3)) for any line with k points? No.
        # If we have many triples, it's harder to hit them all with few points.
        # So minimizing triple count is a good proxy for minimizing hitting set size?
        # Actually, if we can make all triples disjoint, hitting set = num_triples.
        # If we can group them, hitting set < num_triples.
        # Let's just use triple count as a proxy for "complexity" which correlates with score.
        return count_triples(points)

    # Initial Configuration: Jittered Grid
    # Base grid 5x4 = 20 points.
    # To avoid 4 collinear, we must break any line that has 4 points.
    # In a 5x4 grid, rows have 5 points (bad), cols have 4 points (bad).
    # So we must distort the grid significantly.
    
    def generate_initial():
        # Use a known sparse structure or random jitter
        # Let's try a "quasi-random" distribution
        pts = []
        used = set()
        # Try to place points such that no 4 are collinear
        # A simple heuristic: place points on a parabola y = x^2 mod P?
        # Or just random search for a valid start.
        
        # Better: Start with a valid small set and expand?
        # Let's use a deterministic jittered grid but with large jitter to break lines.
        # Base x: 0, 10, 20, 30, 40
        # Base y: 0, 10, 20, 30
        # Jitter each point by random small amount?
        # If jitter is too small, 4 might still be collinear if original was close?
        # Original grid points are integers. If we add integer jitter, we might create new collinearities.
        
        # Strategy: Random restarts until we find a valid one with low triple count.
        # Since 20 points in 201x201 grid, probability of 4 collinear is low for random points.
        
        for _ in range(100):
            pts = []
            used = set()
            valid = True
            for _ in range(20):
                for _ in range(50):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if (x, y) not in used:
                        # Check if adding this point creates 4 collinear with any 3 existing
                        # This is expensive O(N^3) per point. N is small here (0-20).
                        # Max 20 points. Check all triples of existing + new point.
                        is_bad = False
                        n_curr = len(pts)
                        if n_curr >= 3:
                            # Check all triples from pts
                            for i in range(n_curr):
                                for j in range(i+1, n_curr):
                                    for k in range(j+1, n_curr):
                                        p1 = pts[i]
                                        p2 = pts[j]
                                        p3 = pts[k]
                                        # Check if p1, p2, p3, new are collinear
                                        # (y2-y1)(x_new-x1) == (y3-y1)(x_new-x1) ? No.
                                        # Slope p1-p2 == Slope p1-new AND p1-p2 == Slope p3-p2?
                                        # Easier: Area of triangle (p1, p2, new) == 0 AND (p1, p3, new) == 0?
                                        # Actually, if p1, p2, p3 are not collinear, they can't be on a line with new.
                                        # So we only need to check if p1, p2, p3 are collinear? No, that's existing.
                                        # We need to check if there exist ANY 3 points in pts such that they are collinear with new.
                                        # So: For each pair (a,b) in pts, check if (a,b,new) are collinear.
                                        # If so, we need to check if there is a 4th point?
                                        # No, if (a,b,new) are collinear, we just need to ensure that line doesn't already have 3 points in pts.
                                        # So: Count points on line (a,b). If count >= 3, adding new makes 4.
                                        
                                        # Optimization: Pre-compute lines for current pts?
                                        # N is small. Just check pairs.
                                        x1, y1 = p1
                                        x2, y2 = p2
                                        x3, y3 = p3
                                        # Check collinearity of p1, p2, new
                                        # (y2-y1)*(x_new-x1) == (y_new-y1)*(x2-x1)
                                        # (y3-y1)*(x_new-x1) == (y_new-y1)*(x3-x1)
                                        # This is getting complicated.
                                        # Simpler: Just add the point, and check the whole set for 4-collinear?
                                        # That's O(N^3) for N=20 -> 20*19*18/6 = 1140 triples.
                                        # 20 points * 1140 checks * 100 restarts = 2.2M checks. Feasible.
                                        
                                        pass # Skip this inner loop logic, do full check after generation
                        used.add((x, y))
                        pts.append([x, y])
                        break
                else:
                    valid = False
                    break
            
            if valid:
                if check_no_four_collinear(pts):
                    return pts
        # Fallback: Random points
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        return pts

    best_points = None
    best_score_val = float('inf')
    
    # Initial generation
    p = generate_initial()
    if p:
        s = score(p)
        if s < best_score_val:
            best_points = p
            best_score_val = s
    else:
        # Emergency fallback
        best_points = [[i, 0] for i in range(20)] # Invalid, but prevents crash

    # Local Search
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = best_points[idx]
        
        # Try a few random moves
        found_improvement = False
        for _ in range(20):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            if (new_x, new_y) == (old_x, old_y):
                continue
                
            candidate = [pt[:] for pt in best_points]
            candidate[idx] = [new_x, new_y]
            
            # Check distinctness
            if len(set(map(tuple, candidate))) != 20:
                continue
            
            s = score(candidate)
            if s < best_score_val:
                best_points = candidate
                best_score_val = s
                found_improvement = True
                break
        
        if not found_improvement and random.random() < 0.1:
            # Restart with new random config occasionally
            new_pts = generate_initial()
            if new_pts:
                s = score(new_pts)
                if s < best_score_val:
                    best_points = new_pts
                    best_score_val = s

    return best_points
