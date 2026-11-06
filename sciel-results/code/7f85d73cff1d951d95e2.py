import time
import random
import math
from itertools import combinations

def _count_lines_exactly_4(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns -1 if any line contains 5 or more points (invalid).
    """
    n = len(points)
    if len(set(tuple(p) for p in points)) != n:
        return -1
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
    
    # Efficient line counting using direction vectors
    # A line is defined by a pair of points. We normalize the direction (dx, dy) 
    # and the "offset" to identify unique lines.
    # However, for n=16, the O(n^3) check is fast enough, but we can optimize by 
    # grouping points by lines defined by pairs.
    
    lines = {}
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
            dx_n = dx // g
            dy_n = dy // g
            
            # Canonical direction: ensure first non-zero component is positive
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            # The line is uniquely identified by (dx_n, dy_n) and a specific point on it
            # To make it unique, we can use the "cross product" offset: A*x + B*y = C
            # where A = -dy_n, B = dx_n.
            # C = A*x1 + B*y1
            A = -dy_n
            B = dx_n
            C = A * x1 + B * y1
            
            # Normalize C? Not strictly necessary if we just check membership, 
            # but for hashing, we need a consistent key.
            # The key (A, B, C) uniquely identifies the line in integer space 
            # provided (A,B) is primitive.
            
            line_key = (A, B, C)
            
            if line_key not in lines:
                lines[line_key] = set()
            
            lines[line_key].add(i)
            lines[line_key].add(j)
    
    # Now count lines with exactly 4 points
    count_4 = 0
    for key, pts in lines.items():
        if len(pts) == 4:
            count_4 += 1
        elif len(pts) >= 5:
            return -1
            
    return count_4

def _score_with_penalty(points):
    """
    Score = (# of 4-point lines) - 100 * (# of 5+ collinear lines)
    """
    n = len(points)
    if len(set(tuple(p) for p in points)) != n:
        return -10000
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -10000
            
    lines = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            A = -dy_n
            B = dx_n
            C = A * x1 + B * y1
            line_key = (A, B, C)
            
            if line_key not in lines:
                lines[line_key] = set()
            lines[line_key].add(i)
            lines[line_key].add(j)
            
    lines_4 = 0
    lines_5_plus = 0
    for key, pts in lines.items():
        if len(pts) == 4:
            lines_4 += 1
        elif len(pts) >= 5:
            lines_5_plus += 1
            
    return lines_4 - 100 * lines_5_plus

def _generate_grid_start():
    """Generate a 4x4 grid, which is a known local optimum."""
    points = []
    for i in range(4):
        for j in range(4):
            points.append([i * 10 - 15, j * 10 - 15])
    random.shuffle(points)
    return points

def _generate_quad_based_start():
    """
    Start with a few random points, then add points that complete 4-collinear sets.
    """
    points = []
    # Start with 4 random points
    for _ in range(4):
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if [x, y] not in points:
            points.append([x, y])
    
    # Add 12 more points, trying to form quads
    for _ in range(12):
        if len(points) < 3:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            points.append([x, y])
            continue
            
        # Pick 3 existing points
        p1, p2, p3 = random.sample(points, 3)
        
        # Check if they are collinear
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
            # They are collinear. Try to find a 4th point on this line
            # Line equation: A*x + B*y = C
            A = y1 - y2
            B = x2 - x1
            C = A * x1 + B * y1
            
            # Find an integer point on this line that is not one of the existing points
            # and is within bounds.
            # We can pick a random t and solve, or just scan nearby.
            # Let's pick a random point far away on the line.
            # Direction vector (dx, dy)
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0: continue
            dx_n = dx // g
            dy_n = dy // g
            
            # Pick a random offset
            t = random.randint(-20, 20)
            new_x = x1 + t * dx_n
            new_y = y1 + t * dy_n
            
            if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                if [new_x, new_y] not in points:
                    points.append([new_x, new_y])
                    continue
        
        # If not collinear or failed to add, just add a random point
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if [x, y] not in points:
            points.append([x, y])
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_points = None
    best_score = -10000
    
    # Multi-start
    starts = [
        _generate_grid_start(),
        _generate_quad_based_start(),
        _generate_quad_based_start(),
        _generate_quad_based_start()
    ]
    
    for s in starts:
        score = _score_with_penalty(s)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in s]
            
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Move type: 
        # 1. Move a single point to a random location
        # 2. "Snap" a point to complete a quad with 3 others
        # 3. Swap two points
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Snap to complete a quad
            # Pick 3 points
            p1, p2, p3 = random.sample(range(16), 3)
            x1, y1 = current_points[p1]
            x2, y2 = current_points[p2]
            x3, y3 = current_points[p3]
            
            # Check collinearity
            if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                A = y1 - y2
                B = x2 - x1
                C = A * x1 + B * y1
                
                # Try to move one of the other points (not p1,p2,p3) to this line
                candidates = [i for i in range(16) if i not in [p1, p2, p3]]
                if candidates:
                    idx = random.choice(candidates)
                    old_x, old_y = current_points[idx]
                    
                    # Find a point on the line
                    dx = x2 - x1
                    dy = y2 - y1
                    g = math.gcd(abs(dx), abs(dy))
                    if g == 0: g = 1
                    dx_n = dx // g
                    dy_n = dy // g
                    
                    # Pick a random t such that the new point is distinct and in bounds
                    # Avoid the existing 3 points
                    used_points = set([tuple(current_points[i]) for i in [p1, p2, p3]])
                    
                    found = False
                    for _ in range(10):
                        t = random.randint(-50, 50)
                        new_x = x1 + t * dx_n
                        new_y = y1 + t * dy_n
                        if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                            if (new_x, new_y) not in used_points and (new_x, new_y) != (old_x, old_y):
                                # Check if it creates 5-collinear with this line
                                # The line already has 3 points. Adding a 4th makes it 4.
                                # But does it create 5 with another point?
                                # We'll let the scoring function handle the penalty.
                                current_points[idx] = [new_x, new_y]
                                new_score = _score_with_penalty(current_points)
                                
                                if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                                    current_score = new_score
                                    if new_score > best_score:
                                        best_score = new_score
                                        best_points = [p[:] for p in current_points]
                                else:
                                    current_points[idx] = [old_x, old_y]
                                found = True
                                break
                    if not found:
                         # Revert if we didn't make a move, though we didn't change it yet in the loop above effectively
                         pass
                         # Actually, if not found, we didn't modify current_points in the successful branch.
                         # But we did modify it inside the loop. If the loop finishes without 'found', 
                         # it means we either didn't find a valid point or the last attempt failed and we reverted?
                         # No, we only revert if the move is rejected. If we find a point but reject it, we revert.
                         # If we don't find ANY point, we haven't modified current_points.
                         # So this is safe.
            
        elif move_type < 0.8:
            # Single point random move
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Small perturbation
            step = random.randint(-5, 5)
            step2 = random.randint(-5, 5)
            new_x = max(-1000, min(1000, old_x + step))
            new_y = max(-1000, min(1000, old_y + step2))
            
            if new_x == old_x and new_y == old_y:
                continue
                
            current_points[idx] = [new_x, new_y]
            new_score = _score_with_penalty(current_points)
            
            if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[idx] = [old_x, old_y]
                
        else:
            # Swap
            i, j = random.sample(range(16), 2)
            old_i = current_points[i][:]
            old_j = current_points[j][:]
            
            current_points[i] = old_j[:]
            current_points[j] = old_i[:]
            
            new_score = _score_with_penalty(current_points)
            
            if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[i] = old_i[:]
                current_points[j] = old_j[:]
                
        T = max(T_min, T * cooling_rate)
        
        # Restart if stuck
        if current_score < best_score - 5 and random.random() < 0.01:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            T = 5.0

    # Final validation
    final_score = _count_lines_exactly_4(best_points)
    if final_score < 0:
        # If invalid, try to fix by moving points
        # This is a fallback. Ideally, the penalty keeps us valid.
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            trial = [p[:] for p in best_points]
            idx = random.randint(0, 15)
            trial[idx][0] += random.randint(-2, 2)
            trial[idx][1] += random.randint(-2, 2)
            trial[idx][0] = max(-1000, min(1000, trial[idx][0]))
            trial[idx][1] = max(-1000, min(1000, trial[idx][1]))
            if _count_lines_exactly_4(trial) >= 0:
                best_points = trial
                break
                
    return best_points
