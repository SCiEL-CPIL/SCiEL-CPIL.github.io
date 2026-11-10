import numpy as np
import time
import math
import random

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(points, lines):
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    A = lines[:, 0].astype(np.int64)
    B = lines[:, 1].astype(np.int64)
    C = lines[:, 2].astype(np.int64)
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _build_candidate_lines(points, max_coeff=8, max_C_range=80):
    """Build a rich set of candidate lines."""
    n = len(points)
    points_int = points.astype(np.int64)
    seen_lines = set()
    candidate_lines = []
    
    # Lines from point pairs
    for i in range(n):
        x1, y1 = int(points_int[i, 0]), int(points_int[i, 1])
        for j in range(i + 1, n):
            x2, y2 = int(points_int[j, 0]), int(points_int[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            if key not in seen_lines:
                seen_lines.add(key)
                candidate_lines.append(key)
    
    # Systematic small-coefficient lines
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            g = math.gcd(abs(A), abs(B))
            if g > 1:
                continue  # Only primitive (A,B) pairs
            # Determine C range based on point spread
            px = points_int[:, 0]
            py = points_int[:, 1]
            # C = -(A*x + B*y), so range depends on A, B and point range
            c_vals = A * px + B * py
            if len(c_vals) > 0:
                c_min = int(c_vals.min()) - 10
                c_max = int(c_vals.max()) + 10
            else:
                c_min = -max_C_range
                c_max = max_C_range
            # Cap the range to avoid explosion
            c_min = max(c_min, -max_C_range)
            c_max = min(c_max, max_C_range)
            for C in range(c_min, c_max + 1):
                key = _normalize_line(A, B, C)
                if key not in seen_lines:
                    seen_lines.add(key)
                    candidate_lines.append(key)
    
    return np.array(candidate_lines, dtype=np.int64), seen_lines

def _find_best_lines_for_points(points, k=27, candidate_arr=None):
    """Find the k lines with maximum total incidence for the given points."""
    n = len(points)
    points_int = points.astype(np.int64)
    
    if candidate_arr is None or len(candidate_arr) == 0:
        candidate_arr, _ = _build_candidate_lines(points)
    
    if len(candidate_arr) == 0:
        return np.array([[_normalize_line(0, 1, -c)] for c in range(1, 28)], dtype=np.int64)
    
    px = points_int[:, 0:1].T  # (1, N)
    py = points_int[:, 1:2].T  # (1, N)
    A = candidate_arr[:, 0:1].astype(np.int64)
    B = candidate_arr[:, 1:1+1].astype(np.int64)
    C = candidate_arr[:, 2:3].astype(np.int64)
    
    # Compute incidence counts
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    if len(counts) > k:
        idx = np.argsort(-counts)
        top_idx = idx[:k]
    else:
        top_idx = np.arange(len(counts))
    
    return candidate_arr[top_idx].astype(np.int64)

def _generate_point_sets():
    """Generate diverse initial point configurations."""
    sets = []
    
    # 6x9 grid
    pts = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
    sets.append(pts)
    
    # 9x6 grid
    pts = np.array([[i, j] for i in range(9) for j in range(6)], dtype=np.int64)
    sets.append(pts)
    
    # 3x18 grid
    pts = np.array([[i, j] for i in range(3) for j in range(18)], dtype=np.int64)
    sets.append(pts)
    
    # 18x3 grid
    pts = np.array([[i, j] for i in range(18) for j in range(3)], dtype=np.int64)
    sets.append(pts)
    
    # 2x27 grid
    pts = np.array([[i, j] for i in range(2) for j in range(27)], dtype=np.int64)
    sets.append(pts)
    
    # 27x2 grid
    pts = np.array([[i, j] for i in range(27) for j in range(2)], dtype=np.int64)
    sets.append(pts)
    
    # Scaled 6x9 (spacing 2)
    pts = np.array([[i*2, j*2] for i in range(6) for j in range(9)], dtype=np.int64)
    sets.append(pts)
    
    # Scaled 6x9 (spacing 3)
    pts = np.array([[i*3, j*3] for i in range(6) for j in range(9)], dtype=np.int64)
    sets.append(pts)
    
    # Sheared 6x9
    for shear in [1, 2, -1]:
        pts = np.array([[i + shear*j, j] for i in range(6) for j in range(9)], dtype=np.int64)
        if len(set(map(tuple, pts.tolist()))) == 54:
            sets.append(pts)
    
    # Staggered rows
    for stagger in [1, 2]:
        pts = []
        for j in range(9):
            offset = (j % 2) * stagger
            for i in range(6):
                pts.append([i + offset, j])
        pts = np.array(pts, dtype=np.int64)
        if len(set(map(tuple, pts.tolist()))) == 54:
            sets.append(pts)
    
    # Two interleaved grids (scaled to integers)
    pts = []
    for i in range(6):
        for j in range(9):
            if (i + j) % 2 == 0:
                pts.append([2*i, 2*j])
            else:
                pts.append([2*i+1, 2*j+1])
    pts = np.array(pts, dtype=np.int64)
    if len(pts) == 54 and len(set(map(tuple, pts.tolist()))) == 54:
        sets.append(pts)
    
    # L-shape
    pts = []
    for i in range(18):
        pts.append([i, 0])
    for j in range(1, 37):
        pts.append([0, j])
    pts = np.array(pts, dtype=np.int64)
    if len(pts) == 54 and len(set(map(tuple, pts.tolist()))) == 54:
        sets.append(pts)
    
    # Diamond shape
    cx, cy = 20, 20
    pts = []
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            if abs(dx) + abs(dy) <= 6 and len(pts) < 54:
                pts.append([cx + dx, cy + dy])
    if len(pts) == 54:
        pts = np.array(pts, dtype=np.int64)
        if len(set(map(tuple, pts.tolist()))) == 54:
            sets.append(pts)
    
    # Random perturbations of 6x9 grid
    for _ in range(10):
        base = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
        pts = base.copy()
        for _ in range(random.randint(2, 6)):
            idx = random.randint(0, 53)
            pts[idx, 0] += random.randint(-3, 3)
            pts[idx, 1] += random.randint(-3, 3)
        if len(set(map(tuple, pts.tolist()))) == 54:
            sets.append(pts)
    
    return sets

def _refine_set(points, lines, deadline, max_steps=5000):
    """Refine configuration by moving points to maximize incidence."""
    current_score = _count_incidence(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    no_improve_count = 0
    
    # Pre-compute point set for fast conflict checking
    point_set = set(map(tuple, best_points.tolist()))
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        i = random.randint(0, 53)
        ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
        
        moved = False
        # Try offsets in random order, biased toward small moves
        offsets = [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5) if (dx, dy) != (0, 0)]
        random.shuffle(offsets)
        
        for dx, dy in offsets:
            nx, ny = ox + dx, oy + dy
            
            # Quick conflict check
            if (nx, ny) in point_set:
                continue
            
            # Apply move
            point_set.discard((ox, oy))
            point_set.add((nx, ny))
            best_points[i, 0] = nx
            best_points[i, 1] = ny
            
            # Find best lines for new configuration
            new_lines = _find_best_lines_for_points(best_points, 27)
            new_score = _count_incidence(best_points, new_lines)
            
            if new_score > current_score:
                current_score = new_score
                best_lines = new_lines
                best_score = current_score
                no_improve_count = 0
                moved = True
                break
            else:
                # Revert
                point_set.discard((nx, ny))
                point_set.add((ox, oy))
                best_points[i, 0] = ox
                best_points[i, 1] = oy
        
        if not moved:
            no_improve_count += 1
            if no_improve_count > 100:
                # Big jump to escape local optimum
                i = random.randint(0, 53)
                ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
                nx = ox + random.randint(-8, 8)
                ny = oy + random.randint(-8, 8)
                
                if (nx, ny) not in point_set:
                    point_set.discard((ox, oy))
                    point_set.add((nx, ny))
                    best_points[i, 0] = nx
                    best_points[i, 1] = ny
                    
                    new_lines = _find_best_lines_for_points(best_points, 27)
                    new_score = _count_incidence(best_points, new_lines)
                    
                    if new_score >= current_score:
                        current_score = new_score
                        best_lines = new_lines
                        best_score = current_score
                        no_improve_count = 0
                    else:
                        point_set.discard((nx, ny))
                        point_set.add((ox, oy))
                        best_points[i, 0] = ox
                        best_points[i, 1] = oy
    
    return best_points, best_lines, best_score

def _polish_set(points, lines, deadline, max_steps=500):
    """Final polish with strict improvements only."""
    current_score = _count_incidence(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    point_set = set(map(tuple, best_points.tolist()))
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        
        i = random.randint(0, 53)
        ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
        
        moved = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                
                if (nx, ny) in point_set:
                    continue
                
                point_set.discard((ox, oy))
                point_set.add((nx, ny))
                best_points[i, 0] = nx
                best_points[i, 1] = ny
                
                new_lines = _find_best_lines_for_points(best_points, 27)
                new_score = _count_incidence(best_points, new_lines)
                
                if new_score > current_score:
                    current_score = new_score
                    best_lines = new_lines
                    best_score = current_score
                    moved = True
                    break
                else:
                    point_set.discard((nx, ny))
                    point_set.add((ox, oy))
                    best_points[i, 0] = ox
                    best_points[i, 1] = oy
            
            if moved:
                break
    
    return best_points, best_lines, best_score

def _multi_start_refine(points, lines, deadline, num_starts=4):
    """Run multiple refinement passes from perturbed starting points."""
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _count_incidence(points, lines)
    
    for s in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        perturbed = best_points.copy()
        for _ in range(random.randint(1, 4)):
            idx = random.randint(0, 53)
            perturbed[idx, 0] += random.randint(-3, 3)
            perturbed[idx, 1] += random.randint(-3, 3)
        
        if len(set(map(tuple, perturbed.tolist()))) != 54:
            continue
        
        lines_s = _find_best_lines_for_points(perturbed, 27)
        score_s = _count_incidence(perturbed, lines_s)
        
        if score_s >= best_score - 5:
            p, l, sc = _refine_set(perturbed, lines_s, deadline, max_steps=1000)
            if sc > best_score:
                best_points = p
                best_lines = l
                best_score = sc
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Phase 1: Evaluate initial candidate sets
    candidate_sets = _generate_point_sets()
    
    for pts in candidate_sets:
        if time.time() > deadline - 60:
            break
        lines = _find_best_lines_for_points(pts, 27)
        score = _count_incidence(pts, lines)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines.copy()
    
    if best_points is None:
        best_points = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
        best_lines = _find_best_lines_for_points(best_points, 27)
        best_score = _count_incidence(best_points, best_lines)
    
    # Phase 2: Main refinement
    if time.time() < deadline - 120:
        best_points, best_lines, best_score = _refine_set(
            best_points, best_lines, deadline, max_steps=3000
        )
    
    # Phase 3: Multi-start refinement
    if time.time() < deadline - 60:
        best_points, best_lines, best_score = _multi_start_refine(
            best_points, best_lines, deadline, num_starts=3
        )
    
    # Phase 4: Final polish
    if time.time() < deadline - 20:
        best_points, best_lines, best_score = _polish_set(
            best_points, best_lines, deadline, max_steps=300
        )
    
    # Final validation and cleanup
    points = np.array(best_points, dtype=np.int64)
    lines = np.array(best_lines, dtype=np.int64)
    
    # Ensure exactly 54 unique points
    unique_pts = []
    seen = set()
    for p in points:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            seen.add(key)
            unique_pts.append(key)
    
    while len(unique_pts) < 54:
        unique_pts.append((1000 + len(unique_pts), 2000 + len(unique_pts)))
    unique_pts = unique_pts[:54]
    points = np.array(unique_pts, dtype=np.int64)
    
    # Ensure exactly 27 unique valid lines
    valid_lines = []
    seen_lines = set()
    for l in lines:
        key = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if key not in seen_lines and (key[0] != 0 or key[1] != 0):
            seen_lines.add(key)
            valid_lines.append(key)
    
    c = 1000
    while len(valid_lines) < 27:
        key = _normalize_line(0, 1, -c)
        if key not in seen_lines:
            seen_lines.add(key)
            valid_lines.append(key)
        c += 1
    
    valid_lines = valid_lines[:27]
    lines = np.array(valid_lines, dtype=np.int64)
    
    return points, lines
