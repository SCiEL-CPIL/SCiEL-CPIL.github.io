import numpy as np
import time
from math import gcd
from collections import defaultdict
import random

def _normalize_line(a, b, c):
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    for v in [a, b, c]:
        if v != 0:
            if v < 0:
                a, b, c = -a, -b, -c
            break
    return (a, b, c)

def _is_valid(points, lines):
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
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return int(np.sum(vals == 0))

def _build_incidence(pts, lns):
    """Build point->line incidence: for each point, which lines contain it."""
    n_pts, n_lines = len(pts), len(lns)
    # incidence[i] = set of line indices that contain point i
    incidence = [set() for _ in range(n_pts)]
    for j in range(n_lines):
        A, B, C = lns[j]
        for i in range(n_pts):
            if A * pts[i, 0] + B * pts[i, 1] + C == 0:
                incidence[i].add(j)
    return incidence

def _line_score(lns, pts, line_idx):
    """Number of points on line line_idx."""
    A, B, C = lns[line_idx]
    return int(np.sum(A * pts[:, 0] + B * pts[:, 1] + C == 0))

def _generate_grid(w, h):
    """Generate w x h grid of points."""
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x, y))
    return np.array(points, dtype=np.int64)

def _generate_lines_for_grid(w, h):
    """Generate candidate lines for a w x h grid, sorted by length."""
    lines = []
    # Verticals
    for x in range(w):
        lines.append((1, 0, -x))
    # Horizontals
    for y in range(h):
        lines.append((0, 1, -y))
    # Diagonals slope 1: x - y = c
    for c in range(-(h-1), w):
        lines.append((1, -1, -c))
    # Diagonals slope -1: x + y = c
    for c in range(0, w + h):
        lines.append((1, 1, -c))
    # Additional slopes
    for A in range(-3, 4):
        for B in range(-3, 4):
            if A == 0 and B == 0:
                continue
            if (A, B) in [(1,0),(0,1),(1,-1),(1,1),(-1,0),(0,-1),(-1,1),(-1,-1)]:
                continue
            for C in range(-15, 16):
                lines.append((A, B, C))
    
    # Normalize and deduplicate
    seen = set()
    unique_lines = []
    for l in lines:
        c = _normalize_line(l[0], l[1], l[2])
        if c and c not in seen:
            seen.add(c)
            unique_lines.append(c)
    return unique_lines

def _select_best_lines(pts, candidate_lines, n_lines=27):
    """Select n_lines best lines by point count, no duplicates."""
    scored = []
    for l in candidate_lines:
        A, B, C = l
        cnt = int(np.sum(A * pts[:, 0] + B * pts[:, 1] + C == 0))
        scored.append((cnt, l))
    scored.sort(key=lambda x: -x[0])
    
    selected = []
    seen = set()
    for cnt, l in scored:
        if l not in seen:
            seen.add(l)
            selected.append(l)
            if len(selected) == n_lines:
                break
    return selected

def _initialize(w, h):
    """Initialize with grid and best lines."""
    pts = _generate_grid(w, h)
    candidates = _generate_lines_for_grid(w, h)
    lns = _select_best_lines(pts, candidates, 27)
    return pts, np.array(lns, dtype=np.int64)

def _local_search(pts, lns, deadline, seed=42):
    """Fast local search using incidence tracking."""
    rng = random.Random(seed)
    pts = pts.copy()
    lns = lns.copy()
    
    n_pts, n_lines = 54, 27
    
    # Build incidence
    incidence = _build_incidence(pts, lns)
    current_score = sum(len(s) for s in incidence)
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Line scores
    line_scores = np.array([_line_score(lns, pts, j) for j in range(n_lines)])
    
    T_start = 8.0
    T_end = 0.1
    max_iters = 2000000
    iters = 0
    
    while time.time() < deadline - 10:
        iters += 1
        progress = min(iters / max_iters, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        if rng.random() < 0.65:
            # Move a point
            i = rng.randint(0, n_pts - 1)
            ox, oy = int(pts[i, 0]), int(pts[i, 1])
            
            # Old contribution: number of lines containing point i
            old_count = len(incidence[i])
            
            moved = False
            for _ in range(20):
                dx = rng.randint(-4, 4)
                dy = rng.randint(-4, 4)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                
                # Check duplicates
                is_dup = False
                for k in range(n_pts):
                    if k != i and pts[k, 0] == nx and pts[k, 1] == ny:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                # New contribution: count lines containing (nx, ny)
                new_count = 0
                for j in range(n_lines):
                    A, B, C = lns[j]
                    if A * nx + B * ny + C == 0:
                        new_count += 1
                
                delta = new_count - old_count
                
                if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                    # Update incidence
                    pts[i] = [nx, ny]
                    incidence[i] = set()
                    for j in range(n_lines):
                        A, B, C = lns[j]
                        if A * nx + B * ny + C == 0:
                            incidence[i].add(j)
                    current_score += delta
                    moved = True
                    break
            
            if not moved:
                continue
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                best_lns = lns.copy()
                
        else:
            # Replace a line
            j = rng.randint(0, n_lines - 1)
            old_line = tuple(lns[j])
            old_score = line_scores[j]
            
            # Generate new line candidate
            if rng.random() < 0.4:
                # Line through 2 random points
                i1 = rng.randint(0, n_pts - 1)
                i2 = rng.randint(0, n_pts - 1)
                if i1 == i2:
                    continue
                x1, y1 = int(pts[i1, 0]), int(pts[i1, 1])
                x2, y2 = int(pts[i2, 0]), int(pts[i2, 1])
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                new_line = _normalize_line(A, B, C)
            else:
                # Random line from pool
                A = rng.randint(-6, 6)
                B = rng.randint(-6, 6)
                if A == 0 and B == 0:
                    continue
                C = rng.randint(-35, 35)
                new_line = _normalize_line(A, B, C)
            
            if new_line is None or new_line == old_line:
                continue
            
            # Check duplicates
            is_dup = False
            for k in range(n_lines):
                if k != j and tuple(lns[k]) == new_line:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            # Compute new line score
            A, B, C = new_line
            new_score = int(np.sum(A * pts[:, 0] + B * pts[:, 1] + C == 0))
            delta = new_score - old_score
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                lns[j] = new_line
                line_scores[j] = new_score
                current_score += delta
                
                # Update incidence
                for i in range(n_pts):
                    if j in incidence[i]:
                        incidence[i].remove(j)
                    if A * pts[i, 0] + B * pts[i, 1] + C == 0:
                        incidence[i].add(j)
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                best_lns = lns.copy()
                
    return best_pts, best_lns, best_score

def _polish(pts, lns, deadline):
    """Deterministic polish: try best move for each point/line."""
    pts = pts.copy()
    lns = lns.copy()
    n_pts, n_lines = 54, 27
    
    incidence = _build_incidence(pts, lns)
    current_score = sum(len(s) for s in incidence)
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try moving each point
        for i in range(n_pts):
            ox, oy = int(pts[i, 0]), int(pts[i, 1])
            old_count = len(incidence[i])
            best_delta = 0
            best_pos = None
            
            # Try positions in range
            x_min = max(0, int(min(pts[:, 0])) - 3)
            x_max = int(max(pts[:, 0])) + 3
            y_min = max(0, int(min(pts[:, 1])) - 3)
            y_max = int(max(pts[:, 1])) + 3
            
            for nx in range(x_min, x_max + 1):
                for ny in range(y_min, y_max + 1):
                    if nx == ox and ny == oy:
                        continue
                    is_dup = False
                    for k in range(n_pts):
                        if k != i and pts[k, 0] == nx and pts[k, 1] == ny:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    new_count = 0
                    for j in range(n_lines):
                        A, B, C = lns[j]
                        if A * nx + B * ny + C == 0:
                            new_count += 1
                    
                    delta = new_count - old_count
                    if delta > best_delta:
                        best_delta = delta
                        best_pos = (nx, ny)
            
            if best_pos is not None and best_delta > 0:
                nx, ny = best_pos
                pts[i] = [nx, ny]
                incidence[i] = set()
                for j in range(n_lines):
                    A, B, C = lns[j]
                    if A * nx + B * ny + C == 0:
                        incidence[i].add(j)
                current_score += best_delta
                improved = True
        
        # Try replacing each line
        for j in range(n_lines):
            old_line = tuple(lns[j])
            old_score = int(np.sum(old_line[0] * pts[:, 0] + old_line[1] * pts[:, 1] + old_line[2] == 0))
            best_delta = 0
            best_line = None
            
            # Try lines through pairs of points
            seen_cands = set()
            for i1 in range(n_pts):
                for i2 in range(i1 + 1, n_pts):
                    x1, y1 = int(pts[i1, 0]), int(pts[i1, 1])
                    x2, y2 = int(pts[i2, 0]), int(pts[i2, 1])
                    A = y2 - y1
                    B = x1 - x2
                    C = x2 * y1 - x1 * y2
                    cand = _normalize_line(A, B, C)
                    if cand is None or cand == old_line or cand in seen_cands:
                        continue
                    seen_cands.add(cand)
                    
                    is_dup = False
                    for k in range(n_lines):
                        if k != j and tuple(lns[k]) == cand:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    new_score = int(np.sum(cand[0] * pts[:, 0] + cand[1] * pts[:, 1] + cand[2] == 0))
                    delta = new_score - old_score
                    if delta > best_delta:
                        best_delta = delta
                        best_line = cand
            
            if best_line is not None and best_delta > 0:
                lns[j] = best_line
                current_score += best_delta
                improved = True
                
                # Update incidence
                A, B, C = best_line
                for i in range(n_pts):
                    if j in incidence[i]:
                        incidence[i].remove(j)
                    if A * pts[i, 0] + B * pts[i, 1] + C == 0:
                        incidence[i].add(j)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            best_lns = lns.copy()
    
    return best_pts, best_lns, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin (630s limit)
    
    best_pts = None
    best_lns = None
    best_score = 0
    
    # Multi-start with different grid configurations
    configs = [(6, 9), (7, 8), (9, 6), (8, 7), (18, 3), (27, 2)]
    
    for w, h in configs:
        if time.time() > deadline - 60:
            break
        
        # Initialize
        pts, lns = _initialize(w, h)
        
        # Local search
        time_per_start = max(30, (deadline - 60 - (time.time() - start_time)) / max(1, len(configs)))
        sub_deadline = time.time() + time_per_start
        pts, lns, score = _local_search(pts, lns, sub_deadline, seed=w * 100 + h)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            best_lns = lns.copy()
    
    # Polish the best
    if best_pts is not None and time.time() < deadline - 20:
        best_pts, best_lns, best_score = _polish(best_pts, best_lns, deadline)
    
    # Final validation
    pts = np.array(best_pts, dtype=np.int64)
    lns = np.array(best_lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for x in range(6):
            lines.append((1, 0, -x))
        for y in range(9):
            lines.append((0, 1, -y))
        for k in range(12):
            lines.append((1, 1, -k))
        lns = np.array(lines, dtype=np.int64)
    
    return pts, lns
