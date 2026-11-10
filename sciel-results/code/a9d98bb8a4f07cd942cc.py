"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict
from math import gcd

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 0, x1)
    if dy == 0:
        return (0, 1, y1)
    a = dy
    b = -dx
    c = a * x1 + b * y1
    g = gcd(abs(a), abs(b))
    if g > 1:
        a //= g
        b //= g
        c //= g
    if a < 0:
        a = -a
        b = -b
        c = -c
    elif a == 0 and b < 0:
        b = -b
        c = -c
    return (a, b, c)

def _build_line_map(points):
    """Build line map: line_key -> set of point indices."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    return line_map

def _score_from_line_map(line_map):
    """Count lines with exactly 4 points, and check no 5+ collinear."""
    count_4 = 0
    max_collinear = 0
    for key, idx_set in line_map.items():
        sz = len(idx_set)
        if sz >= 5:
            return -1, sz  # invalid
        if sz == 4:
            count_4 += 1
        if sz > max_collinear:
            max_collinear = sz
    return count_4, max_collinear

def _count_4_points(points):
    """Count lines containing exactly 4 of the given points. Returns -1 if invalid."""
    line_map = _build_line_map(points)
    count_4, _ = _score_from_line_map(line_map)
    return count_4

def _is_valid(points):
    """Check validity: 16 distinct, bounds, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    line_map = _build_line_map(points)
    for key, idx_set in line_map.items():
        if len(idx_set) >= 5:
            return False
    return True

def _score_move(points, idx, new_pt, line_map=None):
    """Fast score evaluation for moving point idx to new_pt.
    Only recompute lines involving point idx. Returns (new_score, new_line_map) or (None, None) if invalid."""
    old_pt = points[idx]
    if old_pt == new_pt:
        return None, None
    if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
        return None, None
    # Check distinctness
    for i in range(16):
        if i != idx and points[i] == new_pt:
            return None, None
    
    # Build line map for current state if not provided
    if line_map is None:
        line_map = _build_line_map(points)
    
    cur_score, _ = _score_from_line_map(line_map)
    if cur_score < 0:
        return None, None
    
    # Remove all lines involving old point idx
    lines_to_remove = []
    for key, idx_set in list(line_map.items()):
        if idx in idx_set:
            idx_set.discard(idx)
            if len(idx_set) == 0:
                lines_to_remove.append(key)
    for key in lines_to_remove:
        del line_map[key]
    
    # Add new point and recompute lines involving it
    new_points = [list(p) for p in points]
    new_points[idx] = list(new_pt)
    
    for j in range(16):
        if j == idx:
            continue
        key = _line_key(new_pt, new_points[j])
        if key is not None:
            line_map[key].add(idx)
            line_map[key].add(j)
    
    # Score the new state
    new_score, _ = _score_from_line_map(line_map)
    if new_score < 0:
        return None, None
    
    return new_score, line_map

def _simulated_annealing(points, start_temp=10.0, end_temp=0.01, max_time=50.0, seed=None):
    """Simulated annealing to maximize 4-point lines with proper exponential acceptance."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    current = [list(p) for p in points]
    line_map = _build_line_map(current)
    cur_score, _ = _score_from_line_map(line_map)
    if cur_score < 0:
        cur_score = -100
    
    best = [list(p) for p in current]
    best_score = max(0, cur_score)
    
    total_time = max_time
    
    while time.time() - start_time < max_time:
        elapsed = time.time() - start_time
        temp = max(end_temp, start_temp * (1 - elapsed / total_time))
        
        idx = random.randint(0, 15)
        # Mix of small, medium, and large moves
        r = random.random()
        if r < 0.4:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        elif r < 0.7:
            dx = random.randint(-8, 8)
            dy = random.randint(-8, 8)
        elif r < 0.9:
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
        else:
            dx = random.randint(-30, 30)
            dy = random.randint(-30, 30)
        
        if dx == 0 and dy == 0:
            continue
        
        new_x = current[idx][0] + dx
        new_y = current[idx][1] + dy
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        new_pt = (new_x, new_y)
        new_score, new_line_map = _score_move(current, idx, new_pt, line_map)
        if new_score is None:
            continue
        
        delta = new_score - cur_score
        if delta >= 0:
            current[idx] = [new_x, new_y]
            line_map = new_line_map
            cur_score = new_score
            if cur_score > best_score:
                best = [list(p) for p in current]
                best_score = cur_score
        else:
            # Proper exponential acceptance
            if temp > 0.001:
                # Use log to avoid underflow
                log_prob = delta / temp
                if log_prob > -50:  # Avoid underflow
                    if random.random() < math.exp(log_prob):
                        current[idx] = [new_x, new_y]
                        line_map = new_line_map
                        cur_score = new_score
    
    return best, best_score

def _generate_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]

def _generate_perturbed_grid(perturb_prob=0.3, max_perturb=3):
    """Generate a perturbed 4x4 grid."""
    pts = []
    for x in range(4):
        for y in range(4):
            if random.random() < perturb_prob:
                nx = x + random.randint(-max_perturb, max_perturb)
                ny = y + random.randint(-max_perturb, max_perturb)
            else:
                nx, ny = x, y
            pts.append([nx, ny])
    # Ensure distinctness
    seen = set()
    valid_pts = []
    for p in pts:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            valid_pts.append(p)
    # If we lost points, add random ones
    while len(valid_pts) < 16:
        nx = random.randint(-5, 8)
        ny = random.randint(-5, 8)
        if (nx, ny) not in seen:
            seen.add((nx, ny))
            valid_pts.append([nx, ny])
    return valid_pts[:16]

def _generate_lattice_structure():
    """Try a different structured starting point."""
    return [[x, y] for x in range(1, 5) for y in range(1, 5)]

def _generate_random_cluster():
    """Generate a random clustered configuration."""
    pts = []
    seen = set()
    # Create 4 clusters of 4 points
    for c in range(4):
        cx = random.randint(-2, 2)
        cy = random.randint(-2, 2)
        for _ in range(4):
            nx = cx + random.randint(-2, 2)
            ny = cy + random.randint(-2, 2)
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                pts.append([nx, ny])
    # Fill remaining
    while len(pts) < 16:
        nx = random.randint(-5, 5)
        ny = random.randint(-5, 5)
        if (nx, ny) not in seen:
            seen.add((nx, ny))
            pts.append([nx, ny])
    return pts[:16]

def _generate_affine_fixed():
    """Generate configuration with 3 fixed points at (0,0), (1,0), (0,1) and 13 random points."""
    pts = [[0, 0], [1, 0], [0, 1]]
    seen = {(0, 0), (1, 0), (0, 1)}
    while len(pts) < 16:
        x = random.randint(-5, 10)
        y = random.randint(-5, 10)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts

def _generate_affine_structured():
    """Generate structured configuration with affine-fixed points."""
    # Start with the 3 fixed points
    pts = [[0, 0], [1, 0], [0, 1]]
    seen = {(0, 0), (1, 0), (0, 1)}
    
    # Add points that create 4-collinear lines
    # Line through (0,0) and (1,0): add (2,0) and (3,0)
    pts.append([2, 0])
    pts.append([3, 0])
    seen.add((2, 0))
    seen.add((3, 0))
    
    # Line through (0,0) and (0,1): add (0,2) and (0,3)
    pts.append([0, 2])
    pts.append([0, 3])
    seen.add((0, 2))
    seen.add((0, 3))
    
    # Add more points to create additional 4-collinear lines
    # Diagonal: (1,0), (1,1), (1,2), (1,3)
    pts.append([1, 1])
    pts.append([1, 2])
    pts.append([1, 3])
    seen.add((1, 1))
    seen.add((1, 2))
    seen.add((1, 3))
    
    # Another diagonal: (0,1), (1,1), (2,1), (3,1)
    pts.append([2, 1])
    pts.append([3, 1])
    seen.add((2, 1))
    seen.add((3, 1))
    
    # Fill remaining with random points
    while len(pts) < 16:
        x = random.randint(-3, 6)
        y = random.randint(-3, 6)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    
    return pts[:16]

def _local_polish(points, max_time=30.0, radius=8):
    """Intensive local search with multiple passes and wider neighborhood."""
    start_time = time.time()
    current = [list(p) for p in points]
    line_map = _build_line_map(current)
    cur_score, _ = _score_from_line_map(line_map)
    if cur_score < 0:
        cur_score = 0
        line_map = _build_line_map(current)
    
    best = [list(p) for p in current]
    best_score = cur_score
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for idx in range(16):
            if time.time() - start_time > max_time:
                break
            # Try all moves in radius
            for dx in range(-radius, radius + 1):
                if time.time() - start_time > max_time:
                    break
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = current[idx][0] + dx
                    ny = current[idx][1] + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_pt = (nx, ny)
                    # Quick distinctness check
                    if any(current[i] == new_pt for i in range(16) if i != idx):
                        continue
                    new_score, new_line_map = _score_move(current, idx, new_pt, line_map)
                    if new_score is None:
                        continue
                    if new_score > cur_score:
                        current[idx] = [nx, ny]
                        line_map = new_line_map
                        cur_score = new_score
                        if cur_score > best_score:
                            best = [list(p) for p in current]
                            best_score = cur_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best, best_score

def _targeted_line_search(points, max_time=20.0):
    """Targeted search that focuses on creating 4-point lines."""
    start_time = time.time()
    current = [list(p) for p in points]
    line_map = _build_line_map(current)
    cur_score, _ = _score_from_line_map(line_map)
    if cur_score < 0:
        cur_score = 0
        line_map = _build_line_map(current)
    
    best = [list(p) for p in current]
    best_score = cur_score
    
    # Find lines with 3 points and try to extend them to 4
    while time.time() - start_time < max_time:
        found_improvement = False
        # Get current 3-point lines
        three_point_lines = []
        for key, idx_set in line_map.items():
            if len(idx_set) == 3:
                three_point_lines.append((key, list(idx_set)))
        
        for key, idx_set in three_point_lines:
            if time.time() - start_time > max_time:
                break
            
            # For each 3-point line, try to find a 4th point
            # The line is defined by any two points in idx_set
            p1 = current[idx_set[0]]
            p2 = current[idx_set[1]]
            
            # Try candidate points near the line
            # Get the line equation
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Try points at various distances along and near the line
            for t in range(-2, 5):
                if t == 0 or t == 1:
                    continue
                cx = p1[0] + t * dx
                cy = p1[1] + t * dy
                if abs(cx) > 1000 or abs(cy) > 1000:
                    continue
                # Check if this point is already in the set
                if any(current[i] == (cx, cy) for i in range(16)):
                    continue
                # Check if adding this point creates 5 collinear
                # Temporarily add and check
                temp_points = [list(p) for p in current]
                # Find an index to replace or check if we can add
                # Actually, we need to move a point, not add
                pass
            
            # Alternative: try moving each point to a position that extends a 3-line
            for move_idx in range(16):
                if move_idx in idx_set:
                    continue
                # Try to move current[move_idx] to a position on this line
                # that would make it 4 collinear
                # Get the line through two of the 3 points
                for i in range(len(idx_set)):
                    for j in range(i + 1, len(idx_set)):
                        pa = current[idx_set[i]]
                        pb = current[idx_set[j]]
                        dx2 = pb[0] - pa[0]
                        dy2 = pb[1] - pa[1]
                        
                        # Try positions on this line
                        for t in range(-3, 6):
                            if t == 0 or t == 1:
                                continue
                            nx = pa[0] + t * dx2
                            ny = pa[1] + t * dy2
                            if abs(nx) > 1000 or abs(ny) > 1000:
                                continue
                            new_pt = (nx, ny)
                            if any(current[k] == new_pt for k in range(16) if k != move_idx):
                                continue
                            new_score, new_line_map = _score_move(current, move_idx, new_pt, line_map)
                            if new_score is not None and new_score > cur_score:
                                current[move_idx] = [nx, ny]
                                line_map = new_line_map
                                cur_score = new_score
                                if cur_score > best_score:
                                    best = [list(p) for p in current]
                                    best_score = cur_score
                                found_improvement = True
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            
            if found_improvement:
                break
        
        if not found_improvement:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 130  # Keep 20s safety margin before 150s limit
    
    random.seed(12345)
    
    # Multiple initial configurations including affine-fixed ones
    initial_configs = [
        _generate_grid_4x4(),
        _generate_lattice_structure(),
        _generate_perturbed_grid(perturb_prob=0.1, max_perturb=1),
        _generate_perturbed_grid(perturb_prob=0.3, max_perturb=2),
        _generate_random_cluster(),
        _generate_affine_fixed(),
        _generate_affine_structured(),
    ]
    
    best_points = None
    best_score = -1
    
    # Score all initial configs
    for config in initial_configs:
        if not _is_valid(config):
            continue
        s = _count_4_points(config)
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in config]
    
    if best_points is None:
        best_points = _generate_grid_4x4()
        best_score = 0
    
    # SA from best initial
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        pts, score = _simulated_annealing(best_points, start_temp=8.0, end_temp=0.005,
                                           max_time=min(30, remaining - 10), seed=42)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # Targeted line search from current best
    if time.time() - start_time < deadline - 10:
        remaining = deadline - (time.time() - start_time) - 10
        pts, score = _targeted_line_search(best_points, max_time=min(15, remaining))
        if score > best_score:
            best_points = pts
            best_score = score
    
    # SA from other good initial configs
    for config in initial_configs:
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 15:
            break
        if not _is_valid(config):
            continue
        s = _count_4_points(config)
        if s < best_score - 2:  # Only try if close to best
            continue
        time_per_trial = min(10, remaining / 4)
        pts, score = _simulated_annealing(config, start_temp=6.0, end_temp=0.01,
                                           max_time=time_per_trial, seed=200)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # Final intensive local polish
    if time.time() - start_time < deadline - 5:
        remaining = deadline - (time.time() - start_time) - 5
        pts, score = _local_polish(best_points, max_time=remaining, radius=6)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # Final validation
    if not _is_valid(best_points):
        return _generate_grid_4x4()
    return best_points
# EVOLVE-BLOCK-END
