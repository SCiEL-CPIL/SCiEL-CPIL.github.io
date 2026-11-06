import math
import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds, leaving 10s margin for safety

    # Helper to create a normalized line key
    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        # Normalize direction vector
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        # Ensure consistent sign (e.g., dx > 0 or dx == 0 and dy > 0)
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        
        # Line equation: A*x + B*y + C = 0
        # Normal vector (A, B) is perpendicular to (dx, dy) -> (dy, -dx)
        A = dy
        B = -dx
        # C = -(A*x1 + B*y1)
        C = -(A * x1 + B * y1)
        
        # Normalize (A, B, C) by their gcd
        g_abc = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
        if g_abc > 0:
            A //= g_abc
            B //= g_abc
            C //= g_abc
            
        # Ensure consistent sign for the line equation
        # We want the first non-zero of A, B, C to be positive
        if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    def count_on_line(points, key):
        if key is None:
            return 0
        A, B, C = key
        cnt = 0
        for p in points:
            x, y = p
            if A * x + B * y + C == 0:
                cnt += 1
        return cnt

    def evaluate(points):
        """
        Returns (score, line_counts) where line_counts is a dict: key -> count
        Score is the number of lines with exactly 4 points.
        """
        n = len(points)
        line_counts = {}
        invalid = False
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_counts:
                    # Count points on this line
                    c = count_on_line(points, key)
                    line_counts[key] = c
                    if c >= 5:
                        invalid = True
                        # We can break early if we just need validity, but we need counts for optimization
                        # However, if invalid, score is effectively -inf or we just track it.
                        # The problem says "no five collinear".
        
        if invalid:
            return -1000, line_counts
        
        score = 0
        for c in line_counts.values():
            if c == 4:
                score += 1
        
        return score, line_counts

    def local_search(points, max_iters, T_start, T_end):
        curr_points = [p[:] for p in points]
        curr_score, curr_line_counts = evaluate(curr_points)
        
        best_local = [p[:] for p in curr_points]
        best_local_score = curr_score
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            T = T_start * (T_end / T_start) ** (i / max_iters)
            
            # Move Generation Strategy:
            # 1. Try to move a point to lie on an existing 2- or 3-point line to make it 3 or 4.
            # 2. Try to move a point to break a 5-point line (if any) or reduce 4-point lines if we want to explore, 
            #    but primarily we want to create 4-point lines.
            # 3. Random small perturbations.
            
            move_type = random.random()
            
            new_points = [p[:] for p in curr_points]
            
            if move_type < 0.5:
                # Structured move: Pick a random point, and try to align it with two others
                idx = random.randint(0, 15)
                # Pick two other points
                others = [k for k in range(16) if k != idx]
                p1_idx, p2_idx = random.sample(others, 2)
                p1 = curr_points[p1_idx]
                p2 = curr_points[p2_idx]
                key = line_key(p1, p2)
                
                if key is not None:
                    # Find a point on this line that is close to curr_points[idx] but distinct
                    # Or just project curr_points[idx] onto the line?
                    # Simpler: Pick a random integer point on the line defined by p1, p2 within bounds.
                    # Line: A*x + B*y + C = 0
                    A, B, C = key
                    
                    # Parametric form: x = x1 + t*dx, y = y1 + t*dy
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    
                    # Find t such that point is near curr_points[idx]
                    # t = dot(P - P1, D) / dot(D, D)
                    # We want integer t.
                    # Let's just pick a random t in a reasonable range around the projection.
                    dot_dd = dx*dx + dy*dy
                    if dot_dd != 0:
                        px = curr_points[idx][0] - p1[0]
                        py = curr_points[idx][1] - p1[1]
                        t_float = (px*dx + py*dy) / dot_dd
                        t = int(round(t_float))
                        
                        # Try t, t-1, t+1
                        found = False
                        for dt in [0, -1, 1, -2, 2]:
                            tt = t + dt
                            nx = p1[0] + tt * dx
                            ny = p1[1] + tt * dy
                            if abs(nx) <= 1000 and abs(ny) <= 1000:
                                # Check if distinct from all other points
                                new_pt = [nx, ny]
                                is_distinct = True
                                for k in range(16):
                                    if k != idx and curr_points[k] == new_pt:
                                        is_distinct = False
                                        break
                                if is_distinct:
                                    new_points[idx] = new_pt
                                    found = True
                                    break
                        
                        if not found:
                            continue
            else:
                # Random small perturbation
                idx = random.randint(0, 15)
                coord = random.randint(0, 1)
                delta = random.choice([-2, -1, 1, 2])
                new_points[idx][coord] += delta
                if abs(new_points[idx][0]) > 1000 or abs(new_points[idx][1]) > 1000:
                    continue
                # Check distinctness
                is_distinct = True
                for k in range(16):
                    if k != idx and curr_points[k] == new_points[idx]:
                        is_distinct = False
                        break
                if not is_distinct:
                    continue

            # Evaluate new points
            new_score, new_line_counts = evaluate(new_points)
            
            if new_score >= curr_score:
                curr_points = new_points
                curr_score = new_score
                curr_line_counts = new_line_counts
                if curr_score > best_local_score:
                    best_local = [p[:] for p in curr_points]
                    best_local_score = curr_score
            else:
                delta_e = curr_score - new_score
                if delta_e == 0:
                    accept = True
                else:
                    accept_prob = math.exp(-delta_e / T) if T > 0 else 0
                    accept = random.random() < accept_prob
                if accept:
                    curr_points = new_points
                    curr_score = new_score
                    curr_line_counts = new_line_counts

        return best_local, best_local_score

    # Initial solution: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = evaluate(best_points)
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        # Try grid based perturbation
        if restarts % 2 == 0:
            start_pts = [[x, y] for x in range(4) for y in range(4)]
            # Perturb slightly
            for _ in range(5):
                idx = random.randint(0, 15)
                coord = random.randint(0, 1)
                delta = random.choice([-2, -1, 1, 2])
                start_pts[idx][coord] += delta
                # Ensure distinct and in bounds for start
                # Simple check
                if abs(start_pts[idx][0]) > 1000 or abs(start_pts[idx][1]) > 1000:
                    start_pts[idx][coord] -= delta
        else:
            # Random start in small box
            start_pts = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(16)]
            # Ensure distinct
            attempts = 0
            while len(set(tuple(p) for p in start_pts)) < 16 and attempts < 100:
                idx = random.randint(0, 15)
                start_pts[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
                attempts += 1
            if len(set(tuple(p) for p in start_pts)) < 16:
                # Fallback to grid if random generation fails
                start_pts = [[x, y] for x in range(4) for y in range(4)]
        
        res_pts, res_score = local_search(start_pts, 3000, 1.0, 0.01)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_pts
            
        restarts += 1
        
    return best_points
