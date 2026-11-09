import time
import random
from math import gcd
from itertools import combinations

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def fast_count(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1
    return four_count

def grid_configuration():
    points = [[x, y] for x in range(4) for y in range(4)]
    return points, fast_count(points)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Strategy: Line-Driven Construction + Local Search
    # 1. Start with a base grid or random config
    # 2. Use a "line-focused" move: pick a line with 3 points, add a 4th from a pool
    #    OR pick a line with 5+ points, move one point to break it
    #    OR pick a random point and move it to a position that creates a new 4-line
    # 3. Use simulated annealing but with smarter moves

    points = [[x, y] for x in range(4) for y in range(4)]
    current_score = fast_count(points)
    best_points = [p[:] for p in points]
    best_score = current_score

    if current_score < 0:
        return best_points

    # Precompute line structure for faster updates
    def build_line_structure(pts):
        lines = {}
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        return lines

    def get_score_from_lines(lines):
        score = 0
        for key, indices in lines.items():
            if len(indices) == 4:
                score += 1
            elif len(indices) >= 5:
                return -1
        return score

    def remove_point_from_lines(lines, idx, pts):
        to_delete = []
        for key, indices in lines.items():
            if idx in indices:
                indices.remove(idx)
                if not indices:
                    to_delete.append(key)
        for key in to_delete:
            del lines[key]

    def add_point_to_lines(lines, idx, pts):
        p = pts[idx]
        n = len(pts)
        touched_lines = set()
        for j in range(n):
            if j == idx:
                continue
            key = get_line_key(p, pts[j])
            if key is None:
                continue
            if key in touched_lines:
                continue
            touched_lines.add(key)
            if key not in lines:
                lines[key] = set()
            lines[key].add(idx)

    lines = build_line_structure(points)
    
    # Simulated Annealing with Line-Driven Moves
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.99995
    
    def get_effective_score(lines):
        score = 0
        invalid_count = 0
        for key, indices in lines.items():
            c = len(indices)
            if c == 4:
                score += 1
            elif c >= 5:
                invalid_count += 1
        if invalid_count > 0:
            return -1000 - invalid_count
        return score

    current_eff_score = get_effective_score(lines)

    # Precompute a pool of candidate positions near the grid
    candidate_positions = set()
    for x in range(-5, 10):
        for y in range(-5, 10):
            candidate_positions.add((x, y))
    candidate_positions = list(candidate_positions)

    while time.time() < deadline:
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.3:
            # Line-driven move: find a line with 3 points, try to add a 4th
            three_lines = [(k, v) for k, v in lines.items() if len(v) == 3]
            if three_lines:
                line_key, line_pts = random.choice(three_lines)
                # Find the 3 points on this line
                p1, p2, p3 = list(line_pts)
                # Generate candidate positions that lie on this line
                # Use parametric form: p = p1 + t*(p2-p1)
                dx = points[p2][0] - points[p1][0]
                dy = points[p2][1] - points[p1][1]
                # Try integer t values
                candidates = []
                for t in range(-3, 4):
                    if t == 0 or t == 1:
                        continue
                    nx = points[p1][0] + t * dx
                    ny = points[p1][1] + t * dy
                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                        candidates.append((nx, ny))
                
                if candidates:
                    nx, ny = random.choice(candidates)
                    # Find which point to move (one not on this line)
                    non_line_pts = [i for i in range(16) if i not in line_pts]
                    if non_line_pts:
                        idx = random.choice(non_line_pts)
                        old_x, old_y = points[idx]
                        
                        # Check collision
                        collision = False
                        for i in range(16):
                            if i != idx and points[i][0] == nx and points[i][1] == ny:
                                collision = True
                                break
                        if not collision:
                            points[idx] = [nx, ny]
                            remove_point_from_lines(lines, idx, points)
                            add_point_to_lines(lines, idx, points)
                            
                            new_eff_score = get_effective_score(lines)
                            delta = new_eff_score - current_eff_score
                            
                            if delta >= 0:
                                current_eff_score = new_eff_score
                                if get_score_from_lines(lines) > best_score:
                                    best_score = get_score_from_lines(lines)
                                    best_points = [p[:] for p in points]
                            else:
                                if random.random() < pow(2.71828, delta / T):
                                    current_eff_score = new_eff_score
                                else:
                                    points[idx] = [old_x, old_y]
                                    remove_point_from_lines(lines, idx, points)
                                    add_point_to_lines(lines, idx, points)
        
        elif move_type < 0.5:
            # Break a 5+ line: move one point off a bad line
            bad_lines = [(k, v) for k, v in lines.items() if len(v) >= 5]
            if bad_lines:
                line_key, line_pts = random.choice(bad_lines)
                idx = random.choice(list(line_pts))
                old_x, old_y = points[idx]
                
                # Move to a random nearby position
                for _ in range(10):
                    nx = old_x + random.randint(-5, 5)
                    ny = old_y + random.randint(-5, 5)
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    collision = False
                    for i in range(16):
                        if i != idx and points[i][0] == nx and points[i][1] == ny:
                            collision = True
                            break
                    if collision:
                        continue
                    
                    points[idx] = [nx, ny]
                    remove_point_from_lines(lines, idx, points)
                    add_point_to_lines(lines, idx, points)
                    
                    new_eff_score = get_effective_score(lines)
                    delta = new_eff_score - current_eff_score
                    
                    if delta >= 0:
                        current_eff_score = new_eff_score
                        if get_score_from_lines(lines) > best_score:
                            best_score = get_score_from_lines(lines)
                            best_points = [p[:] for p in points]
                        break
                    else:
                        if random.random() < pow(2.71828, delta / T):
                            current_eff_score = new_eff_score
                            break
                        else:
                            points[idx] = [old_x, old_y]
                            remove_point_from_lines(lines, idx, points)
                            add_point_to_lines(lines, idx, points)
        
        else:
            # Random move: pick a random point and move it
            idx = random.randint(0, 15)
            old_x, old_y = points[idx]
            
            if random.random() < 0.1:
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
            else:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
            
            new_x = old_x + dx
            new_y = old_y + dy
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            collision = False
            for i in range(16):
                if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                    collision = True
                    break
            if collision:
                continue
            
            points[idx] = [new_x, new_y]
            remove_point_from_lines(lines, idx, points)
            add_point_to_lines(lines, idx, points)
            
            new_eff_score = get_effective_score(lines)
            delta = new_eff_score - current_eff_score
            
            if delta >= 0:
                current_eff_score = new_eff_score
                if get_score_from_lines(lines) > best_score:
                    best_score = get_score_from_lines(lines)
                    best_points = [p[:] for p in points]
            else:
                if random.random() < pow(2.71828, delta / T):
                    current_eff_score = new_eff_score
                else:
                    points[idx] = [old_x, old_y]
                    remove_point_from_lines(lines, idx, points)
                    add_point_to_lines(lines, idx, points)
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart with higher T if stuck
            if random.random() < 0.01:
                T = 3.0

    # Final check
    final_score = fast_count(best_points)
    if final_score > best_score:
        best_score = final_score
    elif final_score < 0:
        grid_pts, grid_score = grid_configuration()
        return grid_pts

    return best_points
# EVOLVE-BLOCK-END
