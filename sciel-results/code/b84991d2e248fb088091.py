import numpy as np
import time
import random
from math import gcd

def _normalize_line(a, b, c):
    """Return canonical (A, B, C) for a*x + b*y + c = 0. Returns None if zero line."""
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    # First nonzero positive
    for v in [a, b, c]:
        if v != 0:
            if v < 0:
                a, b, c = -a, -b, -c
            break
    return (a, b, c)

def _is_valid(points, lines):
    """Validate against contract."""
    try:
        p = np.asarray(points)
        l = np.asarray(lines)
    except Exception:
        return False
    if p.shape != (54, 2) or l.shape != (27, 3):
        return False
    for arr in [p, l]:
        if not np.all(np.isfinite(arr)):
            return False
        if not np.all(np.abs(arr - np.round(arr)) < 1e-6):
            return False
    if len(set(map(tuple, np.round(p, 6).astype(int).tolist()))) != 54:
        return False
    seen_lines = set()
    for row in np.round(l, 6).astype(int).tolist():
        canon = _normalize_line(row[0], row[1], row[2])
        if canon is None:
            return False
        if canon in seen_lines:
            return False
        seen_lines.add(canon)
    return True

def _incidence_count(points, lines):
    """Compute exact incidence count. Returns int."""
    pts = np.asarray(points)
    lns = np.asarray(lines)
    x = pts[:, 0, None]
    y = pts[:, 1, None]
    A = lns[:, 0, None, None]
    B = lns[:, 1, None, None]
    C = lns[:, 2, None, None]
    vals = A * x + B * y + C
    return int(np.sum(vals == 0))

def _grid_construction(k=3):
    """Baseline grid."""
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    return points, lines

def _build_incidence_structure(points, lines):
    """
    Build helper structures for fast delta updates.
    Returns:
      point_line_inc: (54, 27) boolean array, point_line_inc[i,j] = True if point i is on line j
      line_point_counts: (27,) int array, number of points on each line
      point_line_counts: (54,) int array, number of lines through each point
      line_canons: list of 27 canonical tuples
    """
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    n_pts = len(pts)
    n_lines = len(lns)
    
    point_line_inc = np.zeros((n_pts, n_lines), dtype=bool)
    for i in range(n_pts):
        for j in range(n_lines):
            A, B, C = lns[j]
            if A * pts[i, 0] + B * pts[i, 1] + C == 0:
                point_line_inc[i, j] = True
    
    line_point_counts = point_line_inc.sum(axis=0)
    point_line_counts = point_line_inc.sum(axis=1)
    line_canons = [_normalize_line(int(lns[j,0]), int(lns[j,1]), int(lns[j,2])) for j in range(n_lines)]
    
    return point_line_inc, line_point_counts, point_line_counts, line_canons

def _compute_delta_point_move(point_line_inc, line_point_counts, point_line_counts, 
                               old_point, new_point, lines, n_lines):
    """
    Compute change in total incidence count when moving a point from old_point to new_point.
    Returns delta_incidence.
    """
    # Find which index this point is (we'll pass index separately for efficiency)
    pass  # This function is replaced by index-based version below

def _simulated_annealing(points, lines, start_time, deadline, seed=42):
    """
    Simulated annealing over point and line moves.
    Uses fast delta updates.
    """
    rng = random.Random(seed)
    
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    n_pts = 54
    n_lines = 27
    
    # Build incidence structure
    point_line_inc, line_point_counts, point_line_counts, line_canons = _build_incidence_structure(pts, lns)
    total_inc = int(line_point_counts.sum())
    
    best_total = total_inc
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Candidate points: large lattice
    min_coord = -20
    max_coord = 20
    cand_points = []
    for x in range(min_coord, max_coord + 1):
        for y in range(min_coord, max_coord + 1):
            cand_points.append((x, y))
    cand_points = np.array(cand_points, dtype=np.int64)
    
    # Candidate lines: generate from small A, B, C
    cand_lines = []
    for A in range(-10, 11):
        for B in range(-10, 11):
            if A == 0 and B == 0:
                continue
            for C in range(-30, 31):
                canon = _normalize_line(A, B, C)
                if canon:
                    cand_lines.append(canon)
    # Deduplicate
    seen = set()
    unique_cands = []
    for cl in cand_lines:
        if cl not in seen:
            seen.add(cl)
            unique_cands.append(cl)
    cand_lines = np.array(unique_cands, dtype=np.int64)
    n_cand_lines = len(cand_lines)
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    n_iters = 50000  # Will be limited by time
    
    # Track current state
    current_pts = pts.copy()
    current_lns = lns.copy()
    current_canons = line_canons[:]
    
    # For fast point move delta:
    # When moving point i from old_pos to new_pos:
    #   For each line j: 
    #     was_incident = point_line_inc[i, j]
    #     now_incident = (A_j * new_x + B_j * new_y + C_j == 0)
    #     delta_j = (1 if now_incident else 0) - (1 if was_incident else 0)
    #   total_delta = sum(delta_j)
    
    # For fast line move delta:
    # When replacing line j with new_line:
    #   For each point i:
    #     was_incident = point_line_inc[i, j]
    #     now_incident = (A_new * x_i + B_new * y_i + C_new == 0)
    #     delta_i = (1 if now_incident else 0) - (1 if was_incident else 0)
    #   total_delta = sum(delta_i)
    
    iter_count = 0
    no_improve_count = 0
    
    while time.time() < deadline - 30 and iter_count < n_iters:
        iter_count += 1
        T = T_start * (T_end / T_start) ** (iter_count / n_iters)
        
        # Choose move type: 60% point move, 40% line move
        if rng.random() < 0.6:
            # Point move
            i = rng.randint(0, n_pts - 1)
            old_pos = current_pts[i]
            new_pos = cand_points[rng.randint(0, len(cand_points) - 1)]
            
            # Check if new_pos is already used by another point
            if any(np.array_equal(current_pts[k], new_pos) for k in range(n_pts) if k != i):
                continue
            
            # Compute delta
            delta = 0
            for j in range(n_lines):
                A, B, C = current_lns[j]
                was = point_line_inc[i, j]
                now = (A * new_pos[0] + B * new_pos[1] + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
            
            new_total = total_inc + delta
            
            if new_total >= total_inc or rng.random() < np.exp(delta / max(T, 1e-10)):
                # Accept move
                total_inc = new_total
                current_pts[i] = new_pos
                # Update point_line_inc for point i
                for j in range(n_lines):
                    A, B, C = current_lns[j]
                    point_line_inc[i, j] = (A * new_pos[0] + B * new_pos[1] + C == 0)
                # Update counts
                point_line_counts[i] = point_line_inc[i, :].sum()
                # line_point_counts doesn't change for this point move? 
                # Actually it does: line j's count changes if point i was or isn't on it
                # Recompute line_point_counts from scratch is expensive, let's track differently
                # For now, recompute line_point_counts
                line_point_counts = point_line_inc.sum(axis=0)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = current_pts.copy()
                    best_lns = current_lns.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Line move
            j = rng.randint(0, n_lines - 1)
            old_line = current_lns[j]
            old_canon = current_canons[j]
            new_line = cand_lines[rng.randint(0, n_cand_lines - 1)]
            new_canon = _normalize_line(int(new_line[0]), int(new_line[1]), int(new_line[2]))
            
            if new_canon is None:
                continue
            if new_canon == old_canon:
                continue
            # Check if new_canon is already used by another line
            if any(current_canons[k] == new_canon for k in range(n_lines) if k != j):
                continue
            
            # Compute delta
            delta = 0
            A_new, B_new, C_new = new_line
            for i in range(n_pts):
                was = point_line_inc[i, j]
                now = (A_new * current_pts[i, 0] + B_new * current_pts[i, 1] + C_new == 0)
                delta += (1 if now else 0) - (1 if was else 0)
            
            new_total = total_inc + delta
            
            if new_total >= total_inc or rng.random() < np.exp(delta / max(T, 1e-10)):
                # Accept move
                total_inc = new_total
                current_lns[j] = new_line
                current_canons[j] = new_canon
                # Update point_line_inc for line j
                for i in range(n_pts):
                    point_line_inc[i, j] = (A_new * current_pts[i, 0] + B_new * current_pts[i, 1] + C_new == 0)
                # Update counts
                line_point_counts[j] = point_line_inc[:, j].sum()
                point_line_counts = point_line_inc.sum(axis=1)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = current_pts.copy()
                    best_lns = current_lns.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
    
    return best_pts, best_lns, best_total

def _structured_config_1():
    """6x9 grid with best lines."""
    from collections import defaultdict
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    line_counts = defaultdict(int)
    n_pts = len(pts)
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            A = -dy
            B = dx
            C = dy * x1 - dx * y1
            canon = _normalize_line(A, B, C)
            if canon:
                line_counts[canon] += 1
    
    # Count actual incidences for all unique lines
    all_lines = list(line_counts.keys())
    line_inc_counts = []
    for canon in all_lines:
        A, B, C = canon
        cnt = 0
        for pt in pts:
            if A * pt[0] + B * pt[1] + C == 0:
                cnt += 1
        line_inc_counts.append((cnt, canon))
    
    line_inc_counts.sort(key=lambda x: -x[0])
    selected = []
    seen = set()
    for cnt, canon in line_inc_counts:
        if canon not in seen:
            selected.append(canon)
            seen.add(canon)
            if len(selected) == 27:
                break
    
    if len(selected) == 27:
        lines = np.array(selected, dtype=np.int64)
        return pts, lines
    return None, None

def _structured_config_2():
    """9x6 grid with best lines."""
    from collections import defaultdict
    xs = np.arange(9)
    ys = np.arange(6)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    line_counts = defaultdict(int)
    n_pts = len(pts)
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            A = -dy
            B = dx
            C = dy * x1 - dx * y1
            canon = _normalize_line(A, B, C)
            if canon:
                line_counts[canon] += 1
    
    all_lines = list(line_counts.keys())
    line_inc_counts = []
    for canon in all_lines:
        A, B, C = canon
        cnt = 0
        for pt in pts:
            if A * pt[0] + B * pt[1] + C == 0:
                cnt += 1
        line_inc_counts.append((cnt, canon))
    
    line_inc_counts.sort(key=lambda x: -x[0])
    selected = []
    seen = set()
    for cnt, canon in line_inc_counts:
        if canon not in seen:
            selected.append(canon)
            seen.add(canon)
            if len(selected) == 27:
                break
    
    if len(selected) == 27:
        lines = np.array(selected, dtype=np.int64)
        return pts, lines
    return None, None

def _structured_config_3():
    """Concentric circles / structured point sets with chord lines."""
    # Try points on a few concentric circles
    points = []
    # Circle radius 0: (0,0)
    points.append((0, 0))
    # Circle radius 1: 4 points
    for angle in range(4):
        import math
        x = round(math.cos(2 * math.pi * angle / 4))
        y = round(math.sin(2 * math.pi * angle / 4))
        points.append((x, y))
    # Circle radius 2: 8 points
    for angle in range(8):
        import math
        x = round(2 * math.cos(2 * math.pi * angle / 8))
        y = round(2 * math.sin(2 * math.pi * angle / 8))
        points.append((x, y))
    # Circle radius 3: 12 points
    for angle in range(12):
        import math
        x = round(3 * math.cos(2 * math.pi * angle / 12))
        y = round(3 * math.sin(2 * math.pi * angle / 12))
        points.append((x, y))
    # Circle radius 4: 16 points
    for angle in range(16):
        import math
        x = round(4 * math.cos(2 * math.pi * angle / 16))
        y = round(4 * math.sin(2 * math.pi * angle / 16))
        points.append((x, y))
    # Circle radius 5: 16 points (total 54)
    for angle in range(16):
        import math
        x = round(5 * math.cos(2 * math.pi * angle / 16))
        y = round(5 * math.sin(2 * math.pi * angle / 16))
        points.append((x, y))
    
    # Remove duplicates
    unique_points = list(set(points))
    if len(unique_points) < 54:
        # Pad with grid points
        for x in range(-5, 6):
            for y in range(-5, 6):
                if (x, y) not in unique_points and len(unique_points) < 54:
                    unique_points.append((x, y))
    
    pts = np.array(unique_points[:54], dtype=np.int64)
    
    from collections import defaultdict
    line_counts = defaultdict(int)
    n_pts = len(pts)
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            A = -dy
            B = dx
            C = dy * x1 - dx * y1
            canon = _normalize_line(A, B, C)
            if canon:
                line_counts[canon] += 1
    
    all_lines = list(line_counts.keys())
    line_inc_counts = []
    for canon in all_lines:
        A, B, C = canon
        cnt = 0
        for pt in pts:
            if A * pt[0] + B * pt[1] + C == 0:
                cnt += 1
        line_inc_counts.append((cnt, canon))
    
    line_inc_counts.sort(key=lambda x: -x[0])
    selected = []
    seen = set()
    for cnt, canon in line_inc_counts:
        if canon not in seen:
            selected.append(canon)
            seen.add(canon)
            if len(selected) == 27:
                break
    
    if len(selected) == 27:
        lines = np.array(selected, dtype=np.int64)
        return pts, lines
    return None, None

def _structured_config_4():
    """Parabolic arc points with tangent/secant lines."""
    # Points on y = x^2 for x in -5..5 (11 points) plus some others
    points = []
    for x in range(-5, 6):
        points.append((x, x*x))
    # Add points on y = 2x^2
    for x in range(-3, 4):
        points.append((x, 2*x*x))
    # Add points on y = -x^2 + 10
    for x in range(-3, 4):
        points.append((x, -x*x + 10))
    # Pad to 54
    unique_points = list(set(points))
    for x in range(-8, 9):
        for y in range(-10, 25):
            if (x, y) not in unique_points and len(unique_points) < 54:
                unique_points.append((x, y))
    
    pts = np.array(unique_points[:54], dtype=np.int64)
    
    from collections import defaultdict
    line_counts = defaultdict(int)
    n_pts = len(pts)
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            A = -dy
            B = dx
            C = dy * x1 - dx * y1
            canon = _normalize_line(A, B, C)
            if canon:
                line_counts[canon] += 1
    
    all_lines = list(line_counts.keys())
    line_inc_counts = []
    for canon in all_lines:
        A, B, C = canon
        cnt = 0
        for pt in pts:
            if A * pt[0] + B * pt[1] + C == 0:
                cnt += 1
        line_inc_counts.append((cnt, canon))
    
    line_inc_counts.sort(key=lambda x: -x[0])
    selected = []
    seen = set()
    for cnt, canon in line_inc_counts:
        if canon not in seen:
            selected.append(canon)
            seen.add(canon)
            if len(selected) == 27:
                break
    
    if len(selected) == 27:
        lines = np.array(selected, dtype=np.int64)
        return pts, lines
    return None, None

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences with 54 points and 27 lines on integer lattice.
    Uses simulated annealing with fast delta updates.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    # Try multiple structured configs and pick the best starting point
    configs = []
    
    p1, l1 = _grid_construction(3)
    configs.append((p1, l1, _incidence_count(p1, l1)))
    
    p2, l2 = _structured_config_1()
    if p2 is not None:
        configs.append((p2, l2, _incidence_count(p2, l2)))
    
    p3, l3 = _structured_config_2()
    if p3 is not None:
        configs.append((p3, l3, _incidence_count(p3, l3)))
    
    p4, l4 = _structured_config_3()
    if p4 is not None:
        configs.append((p4, l4, _incidence_count(p4, l4)))
    
    p5, l5 = _structured_config_4()
    if p5 is not None:
        configs.append((p5, l5, _incidence_count(p5, l5)))
    
    # Sort by incidence count descending
    configs.sort(key=lambda x: -x[2])
    
    best_points = configs[0][0].copy()
    best_lines = configs[0][1].copy()
    best_count = configs[0][2]
    
    # Run simulated annealing from the best config
    if time.time() < deadline - 60:
        sa_pts, sa_lns, sa_count = _simulated_annealing(
            best_points, best_lines, start_time, deadline, seed=42
        )
        if sa_count > best_count:
            best_points = sa_pts
            best_lines = sa_lns
            best_count = sa_count
    
    # If time remains, try SA from second-best config
    if len(configs) > 1 and time.time() < deadline - 90:
        sa_pts2, sa_lns2, sa_count2 = _simulated_annealing(
            configs[1][0], configs[1][1], start_time, deadline, seed=123
        )
        if sa_count2 > best_count:
            best_points = sa_pts2
            best_lines = sa_lns2
            best_count = sa_count2
    
    # If time remains, try SA from third-best config
    if len(configs) > 2 and time.time() < deadline - 120:
        sa_pts3, sa_lns3, sa_count3 = _simulated_annealing(
            configs[2][0], configs[2][1], start_time, deadline, seed=999
        )
        if sa_count3 > best_count:
            best_points = sa_pts3
            best_lines = sa_lns3
            best_count = sa_count3
    
    # Final validation
    best_points = np.array(best_points, dtype=np.int64)
    best_lines = np.array(best_lines, dtype=np.int64)
    
    if not _is_valid(best_points, best_lines):
        # Fallback to safe grid
        best_points, best_lines = _grid_construction(3)
    
    return best_points, best_lines
