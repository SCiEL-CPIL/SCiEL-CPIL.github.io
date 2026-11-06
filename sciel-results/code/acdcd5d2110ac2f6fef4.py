import time
import random
import math
from itertools import combinations

def _count_lines_exactly_4(points):
    n = len(points)
    if len(set(tuple(p) for p in points)) != n:
        return -1
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
    lines_found = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            indices_on_line = set()
            for k in range(n):
                pk = points[k]
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    indices_on_line.add(k)
            if len(indices_on_line) >= 5:
                return -1
            if len(indices_on_line) == 4:
                line_id = tuple(sorted(indices_on_line))
                lines_found.add(line_id)
    return len(lines_found)

def _score_with_penalty(points):
    n = len(points)
    if len(set(tuple(p) for p in points)) != n:
        return -10000
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -10000
    lines_4 = 0
    lines_5_plus = 0
    seen_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            indices_on_line = set()
            for k in range(n):
                pk = points[k]
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    indices_on_line.add(k)
            if len(indices_on_line) >= 2:
                line_id = tuple(sorted(indices_on_line))
                if line_id not in seen_lines:
                    seen_lines.add(line_id)
                    if len(indices_on_line) == 4:
                        lines_4 += 1
                    elif len(indices_on_line) >= 5:
                        lines_5_plus += 1
    return lines_4 - 100 * lines_5_plus

def _line_key(p1, p2):
    """Canonical key for the line through p1 and p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        g = 1
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: dy*x - dx*y = c
    c = dy * p1[0] - dx * p1[1]
    return (dx, dy, c)

def _count_lines_fast(points):
    """
    Fast counting of 4-point lines and 5+ violations.
    Returns (num_4_lines, num_5_plus_lines, list_of_5plus_lines)
    """
    n = len(points)
    line_to_indices = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_to_indices:
                line_to_indices[key] = set()
            line_to_indices[key].add(i)
            line_to_indices[key].add(j)
    
    num_4 = 0
    num_5_plus = 0
    bad_lines = []
    for key, idxs in line_to_indices.items():
        cnt = len(idxs)
        if cnt == 4:
            num_4 += 1
        elif cnt >= 5:
            num_5_plus += 1
            bad_lines.append(tuple(sorted(idxs)))
    return num_4, num_5_plus, bad_lines

def _build_grid_lines(grid_size, num_points=16):
    """
    Generate all 4-point collinear subsets from a grid_size x grid_size grid.
    Returns list of (line_key, frozenset_of_indices)
    """
    points = []
    for x in range(grid_size):
        for y in range(grid_size):
            points.append((x, y))
    n = len(points)
    
    # Find all lines with exactly 4 points
    line_to_indices = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_to_indices:
                line_to_indices[key] = set()
            line_to_indices[key].add(i)
            line_to_indices[key].add(j)
    
    lines_4 = []
    for key, idxs in line_to_indices.items():
        if len(idxs) == 4:
            lines_4.append((key, frozenset(idxs)))
    
    return lines_4, points

def _greedy_line_selection(lines_4, num_points=16, max_lines=None):
    """
    Greedily select 4-point lines that can be realized with <= num_points total.
    Uses a hypergraph approach: each line is a hyperedge of size 4.
    We want to maximize number of hyperedges while total vertices <= num_points.
    """
    if max_lines is None:
        max_lines = len(lines_4)
    
    selected = []
    used_points = set()
    
    # Sort lines by "efficiency" - prefer lines that share points with existing selection
    # Start with random ordering, then greedy
    random.shuffle(lines_4)
    
    for key, idxs in lines_4:
        new_points_needed = idxs - used_points
        if len(used_points | idxs) > num_points:
            continue
        # Accept this line
        selected.append((key, idxs))
        used_points |= idxs
        if len(selected) >= max_lines:
            break
    
    return selected, used_points

def _assign_coordinates(selected_lines, grid_points, num_points=16):
    """
    Assign actual coordinates to points based on selected lines.
    The selected_lines contain indices into grid_points.
    We need to map these to actual 16-point positions.
    """
    # Collect all used point indices
    all_used = set()
    for key, idxs in selected_lines:
        all_used |= idxs
    
    # We have at most 16 points. Assign them coordinates from grid_points.
    # But we need to ensure collinearity is preserved.
    # The grid_points already have collinearity, so we just pick the used ones.
    # If we have fewer than 16, add random extra points.
    
    used_list = sorted(all_used)
    if len(used_list) < num_points:
        # Add unused grid points
        all_grid_indices = set(range(len(grid_points)))
        remaining = sorted(all_grid_indices - all_used)
        for idx in remaining:
            if len(used_list) >= num_points:
                break
            used_list.append(idx)
    
    used_list = used_list[:num_points]
    return [list(grid_points[i]) for i in used_list]

def _repair_no_five_collinear(points, deadline):
    """
    Repair: if any 5+ collinear, move one of the collinear points to break the line.
    """
    points = [p[:] for p in points]
    for iteration in range(500):
        if time.time() > deadline:
            break
        num_4, num_5_plus, bad_lines = _count_lines_fast(points)
        if num_5_plus == 0:
            return points, num_4
        # Pick a bad line and move one of its points
        bad_line = random.choice(bad_lines)
        move_idx = random.choice(bad_line)
        old_x, old_y = points[move_idx]
        # Try small random moves
        for _ in range(50):
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            # Check distinctness
            points[move_idx] = [new_x, new_y]
            if len(set(tuple(p) for p in points)) != 16:
                continue
            n4, n5, _ = _count_lines_fast(points)
            if n5 < num_5_plus:
                break
            # Keep if no worse
            if n5 == num_5_plus and n4 >= 0:
                break
            points[move_idx] = [old_x, old_y]
    return points, _count_lines_fast(points)[0]

def _local_search(points, deadline, max_iters=20000):
    """
    Local search: try moving individual points to improve 4-line count while avoiding 5+.
    """
    points = [p[:] for p in points]
    best_points = [p[:] for p in points]
    best_score = _score_with_penalty(points)
    current_score = best_score
    
    T = 5.0
    T_min = 0.1
    cooling = 0.9998
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Propose a move: try to place on an existing 3-point line to make it 4
        # Or just random small move
        if random.random() < 0.5:
            # Try to find a 3-point line and place on it
            n = len(points)
            found_placement = False
            for i in range(n):
                for j in range(i + 1, n):
                    if i == idx or j == idx:
                        continue
                    p1 = points[i]
                    p2 = points[j]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    if dx == 0 and dy == 0:
                        continue
                    # Check if exactly one other point is on this line
                    count = 0
                    for k in range(n):
                        if k == idx:
                            continue
                        pk = points[k]
                        if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                            count += 1
                    if count == 2:  # Two other points, so placing idx makes 3... we want 3 others
                        # Actually we want lines with exactly 3 points currently, so placing makes 4
                        pass
                    if count == 3:  # Three other points on line, placing makes 4
                        # Find a point on this line that's not occupied
                        # Parametrize line: p1 + t*(dx,dy)
                        # Try integer t values
                        for t in range(-10, 11):
                            nx = p1[0] + t * dx
                            ny = p1[1] + t * dy
                            if abs(nx) > 1000 or abs(ny) > 1000:
                                continue
                            # Check not already occupied
                            occupied = False
                            for k in range(n):
                                if k != idx and points[k][0] == nx and points[k][1] == ny:
                                    occupied = True
                                    break
                            if not occupied:
                                points[idx] = [nx, ny]
                                new_score = _score_with_penalty(points)
                                if new_score > current_score:
                                    current_score = new_score
                                    found_placement = True
                                    break
                                else:
                                    delta = new_score - current_score
                                    if delta < 0 and random.random() < math.exp(delta / T):
                                        current_score = new_score
                                        found_placement = True
                                        break
                                    else:
                                        points[idx] = [old_x, old_y]
                        if found_placement:
                            break
                if found_placement:
                    break
            if found_placement:
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
                T = max(T_min, T * cooling)
                continue
        
        # Random small move
        step = random.randint(-4, 4)
        step2 = random.randint(-4, 4)
        new_x = max(-1000, min(1000, old_x + step))
        new_y = max(-1000, min(1000, old_y + step2))
        if new_x == old_x and new_y == old_y:
            continue
        
        points[idx] = [new_x, new_y]
        new_score = _score_with_penalty(points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            delta = new_score - current_score
            if delta < 0 and random.random() < math.exp(delta / T):
                current_score = new_score
            else:
                points[idx] = [old_x, old_y]
        
        T = max(T_min, T * cooling)
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_points = None
    best_score = -10000
    
    # Strategy: Build from grid line structures
    for grid_size in [4, 5, 6]:
        if time.time() > deadline:
            break
        
        lines_4, grid_points = _build_grid_lines(grid_size)
        
        # Try multiple greedy selections
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Shuffle and greedily select lines
            shuffled = lines_4[:]
            random.shuffle(shuffled)
            
            selected, used_points = _greedy_line_selection(shuffled, 16, max_lines=len(shuffled))
            
            if len(selected) < 2:
                continue
            
            # Assign coordinates
            points = _assign_coordinates(selected, grid_points, 16)
            
            # Center the points
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            points = [[p[0] - cx, p[1] - cy] for p in points]
            
            # Check bounds
            valid = True
            for x, y in points:
                if abs(x) > 1000 or abs(y) > 1000:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check distinctness
            if len(set(tuple(p) for p in points)) != 16:
                continue
            
            # Repair 5-collinear
            points, n4 = _repair_no_five_collinear(points, deadline)
            
            # Local search to improve
            points, score = _local_search(points, deadline, max_iters=5000)
            
            # Final validation
            final_score = _count_lines_exactly_4(points)
            if final_score >= 0:
                if final_score > best_score:
                    best_score = final_score
                    best_points = [p[:] for p in points]
    
    # Also try direct grid subsets as fallback
    if best_points is None:
        # 4x4 grid
        grid44 = [[x - 2, y - 2] for x in range(4) for y in range(4)]
        score = _count_lines_exactly_4(grid44)
        if score >= 0:
            best_points = grid44
            best_score = score
    
    # If we still have time, do more local search from best
    if best_points is not None and time.time() < deadline - 10:
        best_points, best_score = _local_search(best_points, deadline, max_iters=10000)
        # Re-validate
        final_score = _count_lines_exactly_4(best_points)
        if final_score >= 0:
            best_score = final_score
        else:
            # Fallback to a known valid config
            grid44 = [[x - 2, y - 2] for x in range(4) for y in range(4)]
            best_points = grid44
            best_score = _count_lines_exactly_4(grid44)
    
    if best_points is None:
        # Ultimate fallback
        grid44 = [[x - 2, y - 2] for x in range(4) for y in range(4)]
        best_points = grid44
    
    return best_points
