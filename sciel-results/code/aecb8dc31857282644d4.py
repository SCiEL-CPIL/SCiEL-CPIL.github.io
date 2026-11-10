import numpy as np
import time
import math
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line (A,B,C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_two_points(p1, p2):
    """Return normalized line through two points, or None if same point."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)

def _count_incidences(points, lines):
    """Count point-line incidences exactly."""
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _validate(points, lines):
    """Check validity of the configuration."""
    if len(points) != 54 or len(lines) != 27:
        return False
    point_set = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not (isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer))):
            return False
        if x != int(x) or y != int(y):
            return False
        if (x, y) in point_set:
            return False
        point_set.add((x, y))
    line_set = set()
    for l in lines:
        if len(l) != 3:
            return False
        A, B, C = l
        if not (isinstance(A, (int, np.integer)) and isinstance(B, (int, np.integer)) and isinstance(C, (int, np.integer))):
            return False
        if A != int(A) or B != int(B) or C != int(C):
            return False
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(int(A), int(B), int(C))
        if norm in line_set:
            return False
        line_set.add(norm)
    return True

def _build_candidate_lines(points):
    """Build a pool of candidate lines from all pairs of points plus extras."""
    n = len(points)
    line_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            ln = _line_from_two_points(tuple(points[i]), tuple(points[j]))
            if ln is not None and ln != (0, 0, 0):
                line_set.add(ln)
    for i in range(n):
        x, y = points[i]
        line_set.add(_normalize_line(1, 0, -x))
        line_set.add(_normalize_line(0, 1, -y))
    for a in range(-5, 6):
        for b in range(-5, 6):
            if a == 0 and b == 0:
                continue
            for c in range(-30, 31):
                ln = _normalize_line(a, b, c)
                if ln != (0, 0, 0):
                    line_set.add(ln)
    return list(line_set)

def _greedy_select_lines(points, candidates, num_lines, deadline):
    """Greedy: pick lines with highest point count."""
    n = len(points)
    m = len(candidates)
    line_counts = np.zeros(m, dtype=np.int64)
    for j in range(m):
        A, B, C = candidates[j]
        cnt = 0
        for i in range(n):
            x, y = points[i]
            if A * x + B * y + C == 0:
                cnt += 1
        line_counts[j] = cnt
    order = np.argsort(-line_counts)
    selected = []
    for idx in order:
        if time.time() > deadline:
            break
        if len(selected) >= num_lines:
            break
        selected.append(idx)
    sel_lines = np.array([candidates[j] for j in selected], dtype=np.int64)
    total = sum(line_counts[j] for j in selected)
    return sel_lines, int(total)

def _swap_refine_lines(points, lines, candidates, deadline):
    """Try to improve by swapping individual lines."""
    n = len(points)
    m = len(lines)
    current_score = _count_incidences(points, lines)
    current_set = set(tuple(lines[i]) for i in range(m))
    cand_scores = {}
    for cand in candidates:
        A, B, C = cand
        cnt = 0
        for i in range(n):
            x, y = points[i]
            if A * x + B * y + C == 0:
                cnt += 1
        cand_scores[cand] = cnt
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for j in range(m):
            if time.time() > deadline:
                break
            old_line = tuple(lines[j])
            old_score = cand_scores.get(old_line, 0)
            best_cand = None
            best_gain = 0
            for cand in candidates:
                if cand in current_set or cand == old_line:
                    continue
                gain = cand_scores[cand] - old_score
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand
            if best_cand is not None and best_gain > 0:
                current_set.discard(old_line)
                current_set.add(best_cand)
                lines[j] = np.array(best_cand, dtype=np.int64)
                current_score += best_gain
                improved = True
    return lines, current_score

def _point_refine(points, lines, deadline, radius=6):
    """Move points to maximize incidences."""
    n = len(points)
    m = len(lines)
    current_score = _count_incidences(points, lines)
    point_set = set(tuple(points[i]) for i in range(n))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            old_x, old_y = points[i]
            old_pos = (int(old_x), int(old_y))
            old_inc = 0
            for j in range(m):
                A, B, C = lines[j]
                if A * old_x + B * old_y + C == 0:
                    old_inc += 1
            best_gain = 0
            best_pos = None
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = (old_x + dx, old_y + dy)
                    if new_pos in point_set:
                        continue
                    new_inc = 0
                    for j in range(m):
                        A, B, C = lines[j]
                        if A * new_pos[0] + B * new_pos[1] + C == 0:
                            new_inc += 1
                    gain = new_inc - old_inc
                    if gain > best_gain:
                        best_gain = gain
                        best_pos = new_pos
            if best_pos is not None and best_gain > 0:
                point_set.discard(old_pos)
                point_set.add(best_pos)
                points[i] = np.array(best_pos, dtype=np.int64)
                current_score += best_gain
                improved = True
    return points, current_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 620
    
    # Initial: 3x18 grid
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append([a, -1, b])
    lines = np.array(rows, dtype=np.int64)
    
    best_score = _count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Phase 1: Greedy line selection from candidate pool
    if time.time() < deadline - 120:
        candidates = _build_candidate_lines(points)
        sel_lines, greedy_score = _greedy_select_lines(points, candidates, 27, deadline - 120)
        if len(sel_lines) == 27 and greedy_score > best_score:
            best_score = greedy_score
            best_lines = sel_lines.copy()
    
    # Phase 2: Swap refine lines
    if time.time() < deadline - 60:
        curr_lines = best_lines.copy()
        candidates = _build_candidate_lines(best_points)
        curr_lines, swap_score = _swap_refine_lines(best_points, curr_lines, candidates, deadline - 60)
        if swap_score > best_score:
            best_score = swap_score
            best_lines = curr_lines.copy()
    
    # Phase 3: Point refinement
    if time.time() < deadline - 20:
        curr_points = best_points.copy()
        curr_points, pt_score = _point_refine(curr_points, best_lines, deadline - 20, radius=6)
        if pt_score > best_score:
            best_score = pt_score
            best_points = curr_points.copy()
    
    # Phase 4: Another round of line swap after point moves
    if time.time() < deadline - 5:
        curr_lines = best_lines.copy()
        candidates = _build_candidate_lines(best_points)
        curr_lines, swap2_score = _swap_refine_lines(best_points, curr_lines, candidates, deadline - 5)
        if swap2_score > best_score:
            best_score = swap2_score
            best_lines = curr_lines.copy()
    
    if not _validate(best_points, best_lines):
        return points, lines
    
    return best_points, best_lines
