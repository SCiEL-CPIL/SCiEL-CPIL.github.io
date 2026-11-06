import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Initial configuration: 4x4 grid
    # This has 4 horizontal lines of 4, 4 vertical lines of 4, and 2 diagonals of 4.
    # Total 10 lines of 4? 
    # Grid: (0,0)..(3,3)
    # Horiz: y=0,1,2,3 -> 4 lines
    # Vert: x=0,1,2,3 -> 4 lines
    # Diag: (0,0)-(3,3) and (0,3)-(3,0) -> 2 lines
    # Total 10.
    # Are there 5 collinear? No.
    # So initial score is 10.
    
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            return None
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def compute_lines_and_score(pts):
        """
        Returns (score, has_five, line_to_indices)
        score: number of lines with exactly 4 points
        has_five: True if any line has >= 5 points
        line_to_indices: dict mapping line_key to set of point indices
        """
        n = len(pts)
        line_to_indices = defaultdict(set)
        
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is not None:
                    line_to_indices[key].add(i)
                    line_to_indices[key].add(j)
        
        score = 0
        has_five = False
        for key, indices in line_to_indices.items():
            cnt = len(indices)
            if cnt == 4:
                score += 1
            elif cnt >= 5:
                has_five = True
        
        return score, has_five, line_to_indices

    def is_valid(pts):
        # Check distinctness and bounds
        seen = set()
        for x, y in pts:
            if abs(x) > 1000 or abs(y) > 1000:
                return False
            if (x, y) in seen:
                return False
            seen.add((x, y))
        return True

    # Initial evaluation
    current_score, current_has_five, current_lines = compute_lines_and_score(points)
    if current_has_five:
        # Should not happen for 4x4 grid, but just in case
        current_score = -10000
        
    best_points = [p[:] for p in points]
    best_score = current_score
    
    T = 1000.0
    T_min = 0.1
    cooling_rate = 0.99
    
    random.seed(42)
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Generate move
        idx = random.randint(0, 15)
        old_point = points[idx][:]
        
        new_x, new_y = old_point[0], old_point[1]
        
        if random.random() < 0.3:
            # Heuristic: Try to complete a 4-point line
            # Find lines with exactly 3 points
            three_point_lines = [key for key, inds in current_lines.items() if len(inds) == 3]
            
            if three_point_lines:
                target_key = random.choice(three_point_lines)
                target_inds = current_lines[target_key]
                
                # Check if idx is already on this line
                if idx not in target_inds:
                    # We want to move point idx onto this line.
                    # The line is defined by (dx, dy, c).
                    # Equation: dy*x - dx*y = c
                    # We need an integer point (x,y) on this line, distinct from others.
                    # We can try a few nearby points on the line.
                    
                    dx, dy, c = target_key
                    
                    # Find a point on the line near old_point
                    # Parametric: x = x0 + t*dx, y = y0 + t*dy
                    # We need x0, y0 on the line.
                    # Let's just scan a small window around old_point for points on the line.
                    found_move = False
                    for dx_scan in range(-10, 11):
                        if found_move: break
                        for dy_scan in range(-10, 11):
                            cand_x = old_point[0] + dx_scan
                            cand_y = old_point[1] + dy_scan
                            if abs(cand_x) > 1000 or abs(cand_y) > 1000:
                                continue
                            # Check if on line
                            if dy * cand_x - dx * cand_y == c:
                                # Check distinctness
                                is_dup = False
                                for k in range(16):
                                    if k != idx and points[k][0] == cand_x and points[k][1] == cand_y:
                                        is_dup = True
                                        break
                                if not is_dup:
                                    new_x, new_y = cand_x, cand_y
                                    found_move = True
                                    break
                    if found_move:
                        pass # Use this move
                    else:
                        # Fallback to random if heuristic fails
                        new_x = old_point[0] + random.randint(-5, 5)
                        new_y = old_point[1] + random.randint(-5, 5)
                else:
                    # Point is already on the 3-line. Moving it might break it.
                    # Just do a small random move
                    new_x = old_point[0] + random.randint(-2, 2)
                    new_y = old_point[1] + random.randint(-2, 2)
            else:
                # No 3-point lines, do random
                new_x = old_point[0] + random.randint(-5, 5)
                new_y = old_point[1] + random.randint(-5, 5)
        else:
            # Random move
            if random.random() < 0.1:
                new_x = old_point[0] + random.randint(-20, 20)
                new_y = old_point[1] + random.randint(-20, 20)
            else:
                new_x = old_point[0] + random.randint(-5, 5)
                new_y = old_point[1] + random.randint(-5, 5)

        if abs(new_x) > 1000 or abs(new_y) > 1000:
            T *= cooling_rate
            continue

        new_point = [new_x, new_y]
        
        # Check distinctness
        is_dup = False
        for i in range(16):
            if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                is_dup = True
                break
        if is_dup:
            T *= cooling_rate
            continue

        # Apply move
        points[idx] = new_point
        
        # Evaluate
        new_score, new_has_five, new_lines = compute_lines_and_score(points)
        
        if new_has_five:
            new_score = -10000 + new_score # Penalize heavily
            
        if new_score > current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
            current_score = new_score
            current_lines = new_lines
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            # Revert
            points[idx] = old_point
            # We don't need to revert current_lines explicitly if we recompute on accept, 
            # but for efficiency we could. However, compute_lines_and_score is O(N^2) which is 120 ops.
            # It's fast enough. We just keep current_lines from the last accepted state.
            # Wait, if we revert, current_lines is still the one from the *previous* accepted state.
            # So we don't need to update it on rejection.
            
        T *= cooling_rate

    # Final check
    final_score, final_has_five, _ = compute_lines_and_score(best_points)
    if final_has_five:
        # Fallback to initial grid
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
