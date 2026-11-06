"""Rewritten program with a stronger line-completion focused search."""
import time
import random
import math
import sys

def _score_points(points):
    """
    Returns (num_exact_4_lines, has_5_collinear) for 16 points.
    points: list of 16 [x, y]
    """
    n = 16
    line_pts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n, dy_n = dx // g, dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n, dy_n = -dx_n, -dy_n
            
            # Line equation: a*x + b*y = c
            # Normal vector (a, b) is perpendicular to direction (dx_n, dy_n)
            a, b = dy_n, -dx_n
            c = a * x1 + b * y1
            
            # Normalize line coefficients
            g2 = math.gcd(abs(a), abs(b))
            if g2 > 1:
                a, b, c = a // g2, b // g2, c // g2
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
                
            key = (a, b, c)
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
            
    exact_4 = 0
    has_5 = False
    for key, pts in line_pts.items():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
            # Early exit if we just need validity, but we need the count too.
            # We can break if we only cared about validity, but we return both.
            
    return exact_4, has_5

def _get_lines(points):
    """
    Returns a dict mapping line_key -> set of point_indices.
    """
    n = 16
    line_pts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n, dy_n = dx // g, dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n, dy_n = -dx_n, -dy_n
            a, b = dy_n, -dx_n
            c = a * x1 + b * y1
            g2 = math.gcd(abs(a), abs(b))
            if g2 > 1:
                a, b, c = a // g2, b // g2, c // g2
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    return line_pts

def _find_4th_point_candidates(line_key, existing_pts_on_line, all_points):
    """
    Given a line key (a, b, c) and the set of indices on it, find integer points
    on the line that are not in all_points.
    """
    a, b, c = line_key
    candidates = []
    occupied = set()
    for idx in all_points:
        occupied.add(tuple(all_points[idx]))
        
    # We need to find (x, y) integers such that a*x + b*y = c and |x|,|y| <= 1000
    # Also (x, y) not in occupied.
    # We can iterate x from -1000 to 1000? That's 2001 checks.
    # Or iterate over a range that makes sense.
    # Since we only care about points that might be "close" to existing ones to avoid huge jumps?
    # No, we should check all valid integer points on the line within bounds.
    
    # If b != 0:
    #   b*y = c - a*x
    #   y = (c - a*x) / b
    #   We need (c - a*x) % b == 0
    #   Let g = gcd(a, b). Then c must be divisible by g for solutions to exist.
    #   Let a' = a/g, b' = b/g, c' = c/g.
    #   b'*y = c' - a'*x
    #   Since gcd(a', b') = 1, we can find a particular solution.
    #   However, simple iteration over x is fast enough for 2000 values.
    
    if b != 0:
        # Iterate x
        # Optimization: We don't need to check ALL x. We can check x values that are "plausible".
        # But 2000 is small.
        for x in range(-1000, 1001):
            num = c - a * x
            if num % b == 0:
                y = num // b
                if -1000 <= y <= 1000:
                    pt = (x, y)
                    if pt not in occupied:
                        candidates.append(pt)
    else:
        # a*x = c
        if a != 0 and c % a == 0:
            x = c // a
            if -1000 <= x <= 1000:
                for y in range(-1000, 1001):
                    pt = (x, y)
                    if pt not in occupied:
                        candidates.append(pt)
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Initial configuration: Random points or a grid?
    # A 4x4 grid has many collinearities.
    # Let's start with a random configuration to avoid grid bias, or a perturbed grid.
    rng = random.Random(42)
    
    def generate_random_points():
        pts = []
        seen = set()
        while len(pts) < 16:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        return pts
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Run multiple restarts
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        current_points = generate_random_points()
        # Or start with a 4x4 grid and perturb?
        # Grid:
        # grid = [[x, y] for x in range(4) for y in range(4)]
        # current_points = [p[:] for p in grid]
        # for p in current_points:
        #    p[0] += rng.randint(-2, 2)
        #    p[1] += rng.randint(-2, 2)
        
        # Ensure distinct
        seen = set()
        valid_pts = []
        for p in current_points:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                valid_pts.append(p)
        while len(valid_pts) < 16:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                valid_pts.append([x, y])
        current_points = valid_pts[:16]
        
        current_score, has_5 = _score_points(current_points)
        if has_5:
            # If we start with 5 collinear, we need to fix it first
            # But random points rarely have 5 collinear.
            pass
        
        current_lines = _get_lines(current_points)
        
        T = 1.0
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            # Strategy:
            # 1. 50%: Line Completion Move
            # 2. 30%: Random Move
            # 3. 20%: Swap Move
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # Line Completion
                # Find lines with exactly 3 points
                lines_3 = [k for k, v in current_lines.items() if len(v) == 3]
                if lines_3:
                    target_line_key = rng.choice(lines_3)
                    pts_on_line = current_lines[target_line_key]
                    
                    # Get candidates for the 4th point
                    candidates = _find_4th_point_candidates(target_line_key, pts_on_line, current_points)
                    
                    if candidates:
                        # Pick a random candidate
                        new_pt = list(rng.choice(candidates))
                        
                        # We need to move one of the existing points to this new_pt.
                        # Which point? 
                        # Ideally, a point that is NOT on this line (so we don't break the 3-point set).
                        # And ideally, a point that is not part of many other 4-point lines.
                        
                        # Let's pick a point NOT on the target line.
                        candidates_to_move = [i for i in range(16) if i not in pts_on_line]
                        if candidates_to_move:
                            idx_to_move = rng.choice(candidates_to_move)
                            old_pt = current_points[idx_to_move][:]
                            current_points[idx_to_move] = new_pt
                            
                            new_score, new_has_5 = _score_points(current_points)
                            
                            if new_has_5:
                                # Revert
                                current_points[idx_to_move] = old_pt
                                no_improve += 1
                            else:
                                if new_score > current_score or (T > 0.05 and rng.random() < math.exp((new_score - current_score) / T)):
                                    if new_score > current_score:
                                        no_improve = 0
                                        if new_score > best_score:
                                            best_score = new_score
                                            best_points = [p[:] for p in current_points]
                                            best_lines = _get_lines(current_points)
                                    else:
                                        no_improve += 1
                                    current_score = new_score
                                    current_lines = _get_lines(current_points)
                                else:
                                    current_points[idx_to_move] = old_pt
                                    no_improve += 1
            elif move_type < 0.8:
                # Random Move
                idx = rng.randint(0, 15)
                old_pt = current_points[idx][:]
                # Move to a random nearby point
                dx = rng.randint(-5, 5)
                dy = rng.randint(-5, 5)
                new_x = old_pt[0] + dx
                new_y = old_pt[1] + dy
                
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                new_pt = [new_x, new_y]
                
                # Check if new_pt is already occupied
                if any(p == new_pt for p in current_points if p is not current_points[idx]):
                    continue
                
                current_points[idx] = new_pt
                new_score, new_has_5 = _score_points(current_points)
                
                if new_has_5:
                    current_points[idx] = old_pt
                    no_improve += 1
                else:
                    if new_score > current_score or (T > 0.05 and rng.random() < math.exp((new_score - current_score) / T)):
                        if new_score > current_score:
                            no_improve = 0
                            if new_score > best_score:
                                best_score = new_score
                                best_points = [p[:] for p in current_points]
                                best_lines = _get_lines(current_points)
                        else:
                            no_improve += 1
                        current_score = new_score
                        current_lines = _get_lines(current_points)
                    else:
                        current_points[idx] = old_pt
                        no_improve += 1
            else:
                # Swap Move
                i, j = rng.sample(range(16), 2)
                # Swap coordinates? Or swap points?
                # Swapping points doesn't change the set, so it doesn't change the score.
                # We must change the positions.
                # Let's try a "local swap" of coordinates? No, that's not a valid move in the same way.
                # Let's just do a random move with larger step.
                idx = i
                old_pt = current_points[idx][:]
                dx = rng.randint(-10, 10)
                dy = rng.randint(-10, 10)
                new_x = old_pt[0] + dx
                new_y = old_pt[1] + dy
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                new_pt = [new_x, new_y]
                if any(p == new_pt for p in current_points if p is not current_points[idx]):
                    continue
                current_points[idx] = new_pt
                new_score, new_has_5 = _score_points(current_points)
                if new_has_5:
                    current_points[idx] = old_pt
                    no_improve += 1
                else:
                    if new_score > current_score or (T > 0.05 and rng.random() < math.exp((new_score - current_score) / T)):
                        if new_score > current_score:
                            no_improve = 0
                            if new_score > best_score:
                                best_score = new_score
                                best_points = [p[:] for p in current_points]
                                best_lines = _get_lines(current_points)
                        else:
                            no_improve += 1
                        current_score = new_score
                        current_lines = _get_lines(current_points)
                    else:
                        current_points[idx] = old_pt
                        no_improve += 1
                        
            T = max(0.05, T * 0.99)
            
    if best_points is None:
        # Fallback: return a valid configuration
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

# EVOLVE-BLOCK-END
