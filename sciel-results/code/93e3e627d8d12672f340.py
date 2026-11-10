import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line (A, B, C) so gcd(|A|,|B|,|C|) = 1 and first nonzero is positive."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _incidence_count(points, lines):
    """Count (p,l) pairs where A*x + B*y + C == 0."""
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    val = A * x + B * y + C
    return int(np.count_nonzero(val == 0))

def _compute_incidence_matrix(points, lines):
    """Compute boolean incidence matrix (27, 54)."""
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    val = A * x + B * y + C
    return (val == 0)

def _is_valid(points, lines):
    """Quick validity check."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        A, B, C = int(A), int(B), int(C)
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in line_set:
            return False
        line_set.add(norm)
    return True

def _build_grid_points(cols, rows):
    """Build a grid of points."""
    xs = np.arange(cols)
    ys = np.arange(rows)
    PX, PY = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)

def _build_structural_config(cols, rows):
    """
    Build a structured configuration with horizontal, vertical, and diagonal lines.
    For a cols x rows grid, we have cols*rows points.
    Lines: cols horizontal (y=const), rows vertical (x=const), and diagonals.
    """
    points = _build_grid_points(cols, rows)
    n_pts = cols * rows
    
    # We need exactly 54 points and 27 lines
    if n_pts != 54:
        # Take first 54 points
        points = points[:54]
    
    lines_list = []
    line_set = set()
    
    # Horizontal lines: y = j for j in range(rows)
    for j in range(rows):
        norm = _normalize_line(0, 1, -j)
        if norm not in line_set:
            line_set.add(norm)
            lines_list.append(norm)
    
    # Vertical lines: x = i for i in range(cols)
    for i in range(cols):
        norm = _normalize_line(1, 0, -i)
        if norm not in line_set:
            line_set.add(norm)
            lines_list.append(norm)
    
    # Diagonal lines with slope 1: y - x = c
    # For a cols x rows grid, diagonals exist for c in range(-(cols-1), rows)
    for c in range(-(cols-1), rows):
        norm = _normalize_line(-1, 1, c)
        if norm not in line_set:
            line_set.add(norm)
            lines_list.append(norm)
    
    # Diagonal lines with slope -1: y + x = c
    for c in range(0, cols + rows - 1):
        norm = _normalize_line(1, 1, -c)
        if norm not in line_set:
            line_set.add(norm)
            lines_list.append(norm)
    
    # If we have more than 27 lines, select the best 27
    if len(lines_list) > 27:
        # Score each line by its incidence count
        scores = []
        for A, B, C in lines_list:
            val = A * points[:, 0] + B * points[:, 1] + C
            scores.append(int(np.count_nonzero(val == 0)))
        
        # Sort by score descending, take top 27
        order = np.argsort(-scores)
        selected = [lines_list[i] for i in order[:27]]
        lines_list = selected
    elif len(lines_list) < 27:
        # Add more diagonal lines or other lines to fill up to 27
        # Try more slopes
        for slope_num in range(-3, 4):
            for slope_den in range(1, 4):
                if slope_den == 0:
                    continue
                g = gcd(abs(slope_num), slope_den)
                sn = slope_num // g
                sd = slope_den // g
                if sn == 0 and sd == 1:
                    continue  # already have horizontal
                if sd == 1 and sn == 0:
                    continue
                # Line: sd*y - sn*x = c  =>  -sn*x + sd*y - c = 0
                for c in range(-10, 11):
                    norm = _normalize_line(-sn, sd, -c)
                    if norm not in line_set:
                        line_set.add(norm)
                        lines_list.append(norm)
                        if len(lines_list) >= 27:
                            break
                if len(lines_list) >= 27:
                    break
            if len(lines_list) >= 27:
                break
        
        if len(lines_list) > 27:
            scores = []
            for A, B, C in lines_list:
                val = A * points[:, 0] + B * points[:, 1] + C
                scores.append(int(np.count_nonzero(val == 0)))
            order = np.argsort(-scores)
            lines_list = [lines_list[i] for i in order[:27]]
        
        # If still not enough, pad
        while len(lines_list) < 27:
            lines_list.append((1, 0, 0))
    
    lines = np.array(lines_list[:27], dtype=np.int64)
    return points, lines

def _local_search(points, lines, deadline, seed=42):
    """
    Greedy local search with multiple move types.
    """
    points = points.copy()
    lines = lines.copy()
    
    M = _compute_incidence_matrix(points, lines)
    current_score = int(np.sum(M))
    best_score = current_score
    best_points = points.copy()
    best_lines = lines.copy()
    
    point_set = set(map(tuple, points.tolist()))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        norm = _normalize_line(int(A), int(B), int(C))
        line_set.add(norm)
    
    rng = np.random.default_rng(seed)
    
    iters = 0
    no_improve = 0
    while time.time() < deadline:
        iters += 1
        no_improve += 1
        
        if iters % 2000 == 0:
            current_score = _incidence_count(points, lines)
            M = _compute_incidence_matrix(points, lines)
            point_set = set(map(tuple, points.tolist()))
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = lines.copy()
                no_improve = 0
        
        if no_improve > 4000:
            # Perturb to escape local optimum
            for _ in range(5):
                p_idx = rng.integers(0, 54)
                dx = rng.integers(-3, 4)
                dy = rng.integers(-3, 4)
                nx, ny = points[p_idx][0] + dx, points[p_idx][1] + dy
                if (nx, ny) not in point_set or (nx, ny) == tuple(points[p_idx]):
                    point_set.discard(tuple(points[p_idx]))
                    points[p_idx] = [nx, ny]
                    point_set.add((nx, ny))
            M = _compute_incidence_matrix(points, lines)
            current_score = int(np.sum(M))
            no_improve = 0
            continue
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Move a single point to maximize incidences
            p_idx = rng.integers(0, 54)
            orig_x, orig_y = points[p_idx]
            old_inc = int(np.sum(M[:, p_idx]))
            
            best_delta = 0
            best_pos = None
            
            # Try candidates: nearby points and points on existing lines
            candidates = set()
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    candidates.add((orig_x + dx, orig_y + dy))
            
            # Also try points on each line
            for l_idx in range(27):
                A, B, C = lines[l_idx]
                if B != 0:
                    for dx in range(-5, 6):
                        nx = orig_x + dx
                        num = -(A * nx + C)
                        if num % B == 0:
                            ny = num // B
                            if -200 <= nx <= 200 and -200 <= ny <= 200:
                                candidates.add((nx, ny))
                elif A != 0:
                    for dy in range(-5, 6):
                        ny = orig_y + dy
                        num = -(B * ny + C)
                        if num % A == 0:
                            nx = num // A
                            if -200 <= nx <= 200 and -200 <= ny <= 200:
                                candidates.add((nx, ny))
            
            for new_x, new_y in candidates:
                if (new_x, new_y) in point_set and (new_x, new_y) != (orig_x, orig_y):
                    continue
                
                new_inc = 0
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    if A * new_x + B * new_y + C == 0:
                        new_inc += 1
                        
                delta = new_inc - old_inc
                if delta > best_delta:
                    best_delta = delta
                    best_pos = (new_x, new_y)
            
            if best_pos is not None and best_delta > 0:
                point_set.discard((orig_x, orig_y))
                points[p_idx] = best_pos
                point_set.add(best_pos)
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    M[l_idx, p_idx] = (A * best_pos[0] + B * best_pos[1] + C == 0)
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    no_improve = 0
                    
        elif move_type < 0.8:
            # Replace a line with the best available line
            l_idx = rng.integers(0, 27)
            old_inc = int(np.sum(M[l_idx, :]))
            
            best_new_inc = old_inc
            best_norm = None
            
            # Sample pairs
            for _ in range(200):
                p1_idx = rng.integers(0, 54)
                p2_idx = rng.integers(0, 54)
                if p1_idx == p2_idx:
                    continue
                    
                x1, y1 = points[p1_idx]
                x2, y2 = points[p2_idx]
                
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                if A == 0 and B == 0:
                    continue
                    
                norm = _normalize_line(A, B, C)
                if norm in line_set:
                    continue
                
                new_inc = 0
                for p_idx in range(54):
                    x, y = points[p_idx]
                    if norm[0] * x + norm[1] * y + norm[2] == 0:
                        new_inc += 1
                        
                if new_inc > best_new_inc:
                    best_new_inc = new_inc
                    best_norm = norm
            
            if best_norm is not None and best_new_inc > old_inc:
                line_set.discard(_normalize_line(int(lines[l_idx][0]), int(lines[l_idx][1]), int(lines[l_idx][2])))
                lines[l_idx] = best_norm
                line_set.add(best_norm)
                for p_idx in range(54):
                    x, y = points[p_idx]
                    M[l_idx, p_idx] = (best_norm[0] * x + best_norm[1] * y + best_norm[2] == 0)
                current_score += best_new_inc - old_inc
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    no_improve = 0
                    
        else:
            # Move two points
            p1_idx = rng.integers(0, 54)
            p2_idx = rng.integers(0, 54)
            if p1_idx == p2_idx:
                continue
            
            orig_x1, orig_y1 = points[p1_idx]
            orig_x2, orig_y2 = points[p2_idx]
            
            best_delta = 0
            best_move = None
            
            for _ in range(15):
                dx1 = rng.integers(-4, 5)
                dy1 = rng.integers(-4, 5)
                dx2 = rng.integers(-4, 5)
                dy2 = rng.integers(-4, 5)
                if dx1 == 0 and dy1 == 0 and dx2 == 0 and dy2 == 0:
                    continue
                
                nx1, ny1 = orig_x1 + dx1, orig_y1 + dy1
                nx2, ny2 = orig_x2 + dx2, orig_y2 + dy2
                
                if (nx1, ny1) in point_set and (nx1, ny1) != (orig_x1, orig_y1):
                    continue
                if (nx2, ny2) in point_set and (nx2, ny2) != (orig_x2, orig_y2):
                    continue
                if (nx1, ny1) == (nx2, ny2):
                    continue
                
                old_inc1 = int(np.sum(M[:, p1_idx]))
                old_inc2 = int(np.sum(M[:, p2_idx]))
                
                new_inc1 = 0
                new_inc2 = 0
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    if A * nx1 + B * ny1 + C == 0:
                        new_inc1 += 1
                    if A * nx2 + B * ny2 + C == 0:
                        new_inc2 += 1
                        
                delta = (new_inc1 + new_inc2) - (old_inc1 + old_inc2)
                if delta > best_delta:
                    best_delta = delta
                    best_move = (nx1, ny1, nx2, ny2)
            
            if best_move is not None and best_delta > 0:
                nx1, ny1, nx2, ny2 = best_move
                point_set.discard((orig_x1, orig_y1))
                point_set.discard((orig_x2, orig_y2))
                points[p1_idx] = [nx1, ny1]
                points[p2_idx] = [nx2, ny2]
                point_set.add((nx1, ny1))
                point_set.add((nx2, ny2))
                
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    M[l_idx, p1_idx] = (A * nx1 + B * ny1 + C == 0)
                    M[l_idx, p2_idx] = (A * nx2 + B * ny2 + C == 0)
                    
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    no_improve = 0

    return best_points, best_lines, best_score

def _simulated_annealing(points, lines, deadline, seed=42):
    """
    Simulated annealing to escape local optima.
    """
    points = points.copy()
    lines = lines.copy()
    
    M = _compute_incidence_matrix(points, lines)
    current_score = int(np.sum(M))
    best_score = current_score
    best_points = points.copy()
    best_lines = lines.copy()
    
    point_set = set(map(tuple, points.tolist()))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        norm = _normalize_line(int(A), int(B), int(C))
        line_set.add(norm)
    
    rng = np.random.default_rng(seed)
    
    T = 8.0
    T_min = 0.01
    cooling_rate = 0.9997
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        if iters % 5000 == 0:
            current_score = _incidence_count(points, lines)
            M = _compute_incidence_matrix(points, lines)
            point_set = set(map(tuple, points.tolist()))
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = lines.copy()
            
            if T < T_min:
                T = 1.0
                for _ in range(8):
                    p_idx = rng.integers(0, 54)
                    dx = rng.integers(-8, 9)
                    dy = rng.integers(-8, 9)
                    nx, ny = points[p_idx][0] + dx, points[p_idx][1] + dy
                    if (nx, ny) not in point_set or (nx, ny) == tuple(points[p_idx]):
                        point_set.discard(tuple(points[p_idx]))
                        points[p_idx] = [nx, ny]
                        point_set.add((nx, ny))
                M = _compute_incidence_matrix(points, lines)
                current_score = int(np.sum(M))
                continue
        
        T *= cooling_rate
        
        move_type = rng.random()
        
        if move_type < 0.6:
            # Move a single point
            p_idx = rng.integers(0, 54)
            orig_x, orig_y = points[p_idx]
            dx = rng.integers(-6, 7)
            dy = rng.integers(-6, 7)
            if dx == 0 and dy == 0:
                continue
            new_x, new_y = orig_x + dx, orig_y + dy
            
            if (new_x, new_y) in point_set and (new_x, new_y) != (orig_x, orig_y):
                continue
            
            old_inc = int(np.sum(M[:, p_idx]))
            new_inc = 0
            for l_idx in range(27):
                A, B, C = lines[l_idx]
                if A * new_x + B * new_y + C == 0:
                    new_inc += 1
            
            delta = new_inc - old_inc
            if delta >= 0 or rng.random() < np.exp(delta / T):
                point_set.discard((orig_x, orig_y))
                points[p_idx] = [new_x, new_y]
                point_set.add((new_x, new_y))
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    M[l_idx, p_idx] = (A * new_x + B * new_y + C == 0)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    
        elif move_type < 0.85:
            # Replace a line
            l_idx = rng.integers(0, 27)
            old_inc = int(np.sum(M[l_idx, :]))
            
            p1_idx = rng.integers(0, 54)
            p2_idx = rng.integers(0, 54)
            if p1_idx == p2_idx:
                continue
                
            x1, y1 = points[p1_idx]
            x2, y2 = points[p2_idx]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue
                
            norm = _normalize_line(A, B, C)
            if norm in line_set:
                continue
            
            new_inc = 0
            for p_idx in range(54):
                x, y = points[p_idx]
                if norm[0] * x + norm[1] * y + norm[2] == 0:
                    new_inc += 1
            
            delta = new_inc - old_inc
            if delta >= 0 or rng.random() < np.exp(delta / T):
                line_set.discard(_normalize_line(int(lines[l_idx][0]), int(lines[l_idx][1]), int(lines[l_idx][2])))
                lines[l_idx] = norm
                line_set.add(norm)
                for p_idx in range(54):
                    x, y = points[p_idx]
                    M[l_idx, p_idx] = (norm[0] * x + norm[1] * y + norm[2] == 0)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    
        else:
            # Move two points
            p1_idx = rng.integers(0, 54)
            p2_idx = rng.integers(0, 54)
            if p1_idx == p2_idx:
                continue
            
            orig_x1, orig_y1 = points[p1_idx]
            orig_x2, orig_y2 = points[p2_idx]
            
            dx1 = rng.integers(-4, 5)
            dy1 = rng.integers(-4, 5)
            dx2 = rng.integers(-4, 5)
            dy2 = rng.integers(-4, 5)
            if dx1 == 0 and dy1 == 0 and dx2 == 0 and dy2 == 0:
                continue
            
            nx1, ny1 = orig_x1 + dx1, orig_y1 + dy1
            nx2, ny2 = orig_x2 + dx2, orig_y2 + dy2
            
            if (nx1, ny1) in point_set and (nx1, ny1) != (orig_x1, orig_y1):
                continue
            if (nx2, ny2) in point_set and (nx2, ny2) != (orig_x2, orig_y2):
                continue
            if (nx1, ny1) == (nx2, ny2):
                continue
            
            old_inc1 = int(np.sum(M[:, p1_idx]))
            old_inc2 = int(np.sum(M[:, p2_idx]))
            
            new_inc1 = 0
            new_inc2 = 0
            for l_idx in range(27):
                A, B, C = lines[l_idx]
                if A * nx1 + B * ny1 + C == 0:
                    new_inc1 += 1
                if A * nx2 + B * ny2 + C == 0:
                    new_inc2 += 1
            
            delta = (new_inc1 + new_inc2) - (old_inc1 + old_inc2)
            if delta >= 0 or rng.random() < np.exp(delta / T):
                point_set.discard((orig_x1, orig_y1))
                point_set.discard((orig_x2, orig_y2))
                points[p1_idx] = [nx1, ny1]
                points[p2_idx] = [nx2, ny2]
                point_set.add((nx1, ny1))
                point_set.add((nx2, ny2))
                
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    M[l_idx, p1_idx] = (A * nx1 + B * ny1 + C == 0)
                    M[l_idx, p2_idx] = (A * nx2 + B * ny2 + C == 0)
                    
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()

    return best_points, best_lines, best_score

def _greedy_line_selection(points, k=27):
    """
    Greedy maximum coverage: select k lines to maximize total incidences.
    Each line's 'coverage' is the number of points it hits.
    Greedily pick the line that adds the most new coverage.
    """
    n_pts = len(points)
    # Generate candidate lines from all pairs
    candidate_lines = set()
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            candidate_lines.add(norm)
    
    if not candidate_lines:
        return np.zeros((k, 3), dtype=np.int64)
    
    # Precompute which points each candidate line hits
    line_points = {}
    for norm in candidate_lines:
        A, B, C = norm
        hits = []
        for i in range(n_pts):
            x, y = points[i]
            if A * x + B * y + C == 0:
                hits.append(i)
        line_points[norm] = hits
    
    # Greedy selection
    selected_lines = []
    covered = set()
    remaining = set(candidate_lines)
    
    for _ in range(k):
        if not remaining:
            break
        best_line = None
        best_gain = -1
        for norm in remaining:
            gain = len(set(line_points[norm]) - covered)
            if gain > best_gain:
                best_gain = gain
                best_line = norm
        if best_line is not None:
            selected_lines.append(best_line)
            covered.update(line_points[best_line])
            remaining.discard(best_line)
    
    while len(selected_lines) < k:
        selected_lines.append((1, 0, 0))
    
    return np.array(selected_lines[:k], dtype=np.int64)

def _joint_optimize(points, lines, deadline, seed=42):
    """
    Joint optimization: alternate between optimizing points given lines,
    and optimizing lines given points, using greedy coverage.
    """
    points = points.copy()
    lines = lines.copy()
    
    best_score = _incidence_count(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    rng = np.random.default_rng(seed)
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        # Step 1: Optimize lines given points (greedy coverage)
        new_lines = _greedy_line_selection(points, 27)
        
        # Check validity
        line_set = set()
        valid = True
        for i in range(27):
            A, B, C = new_lines[i]
            A, B, C = int(A), int(B), int(C)
            if A == 0 and B == 0:
                valid = False
                break
            norm = _normalize_line(A, B, C)
            if norm in line_set:
                valid = False
                break
            line_set.add(norm)
        
        if valid:
            score = _incidence_count(points, new_lines)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                best_lines = new_lines.copy()
                lines = new_lines.copy()
        
        # Step 2: Optimize points given lines (move each point to best position)
        M = _compute_incidence_matrix(points, lines)
        point_set = set(map(tuple, points.tolist()))
        
        for p_idx in range(54):
            orig_x, orig_y = points[p_idx]
            old_inc = int(np.sum(M[:, p_idx]))
            
            best_delta = 0
            best_pos = None
            
            # Try nearby positions and positions on lines
            candidates = set()
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    candidates.add((orig_x + dx, orig_y + dy))
            
            for l_idx in range(27):
                A, B, C = lines[l_idx]
                if B != 0:
                    for dx in range(-4, 5):
                        nx = orig_x + dx
                        num = -(A * nx + C)
                        if num % B == 0:
                            ny = num // B
                            if -200 <= nx <= 200 and -200 <= ny <= 200:
                                candidates.add((nx, ny))
                elif A != 0:
                    for dy in range(-4, 5):
                        ny = orig_y + dy
                        num = -(B * ny + C)
                        if num % A == 0:
                            nx = num // A
                            if -200 <= nx <= 200 and -200 <= ny <= 200:
                                candidates.add((nx, ny))
            
            for new_x, new_y in candidates:
                if (new_x, new_y) in point_set and (new_x, new_y) != (orig_x, orig_y):
                    continue
                
                new_inc = 0
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    if A * new_x + B * new_y + C == 0:
                        new_inc += 1
                        
                delta = new_inc - old_inc
                if delta > best_delta:
                    best_delta = delta
                    best_pos = (new_x, new_y)
            
            if best_pos is not None and best_delta > 0:
                point_set.discard((orig_x, orig_y))
                points[p_idx] = best_pos
                point_set.add(best_pos)
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    M[l_idx, p_idx] = (A * best_pos[0] + B * best_pos[1] + C == 0)
        
        score = int(np.sum(M))
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines.copy()
        
        # Occasionally perturb to escape local optima
        if iters % 50 == 0:
            for _ in range(3):
                p_idx = rng.integers(0, 54)
                dx = rng.integers(-5, 6)
                dy = rng.integers(-5, 6)
                nx, ny = points[p_idx][0] + dx, points[p_idx][1] + dy
                if (nx, ny) not in point_set or (nx, ny) == tuple(points[p_idx]):
                    point_set.discard(tuple(points[p_idx]))
                    points[p_idx] = [nx, ny]
                    point_set.add((nx, ny))

    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    deadline = time.time() + 620  # Safety margin before 630s limit
    
    # Generate multiple initial configurations using structural approaches
    initial_configs = []
    
    # 9x6 grid
    p1, l1 = _build_structural_config(9, 6)
    if _is_valid(p1, l1):
        initial_configs.append((p1, l1, _incidence_count(p1, l1)))
    
    # 6x9 grid
    p2, l2 = _build_structural_config(6, 9)
    if _is_valid(p2, l2):
        initial_configs.append((p2, l2, _incidence_count(p2, l2)))
    
    # 18x3 grid
    p3, l3 = _build_structural_config(18, 3)
    if _is_valid(p3, l3):
        initial_configs.append((p3, l3, _incidence_count(p3, l3)))
    
    # 3x18 grid
    p4, l4 = _build_structural_config(3, 18)
    if _is_valid(p4, l4):
        initial_configs.append((p4, l4, _incidence_count(p4, l4)))
    
    # 7x8 grid (56 points, take 54)
    p5 = _build_grid_points(7, 8)[:54]
    l5 = _greedy_line_selection(p5, 27)
    if _is_valid(p5, l5):
        initial_configs.append((p5, l5, _incidence_count(p5, l5)))
    
    # 11x5 grid (55 points, take 54)
    p6 = _build_grid_points(11, 5)[:54]
    l6 = _greedy_line_selection(p6, 27)
    if _is_valid(p6, l6):
        initial_configs.append((p6, l6, _incidence_count(p6, l6)))
    
    if not initial_configs:
        p1 = _build_grid_points(9, 6)
        l1 = _greedy_line_selection(p1, 27)
        return p1, l1
    
    # Sort by initial score
    initial_configs.sort(key=lambda x: -x[2])
    
    best_points = initial_configs[0][0].copy()
    best_lines = initial_configs[0][1].copy()
    best_score = initial_configs[0][2]
    
    # Time allocation for joint optimization
    n_configs = min(len(initial_configs), 4)  # Limit to top 4 configs
    total_time = deadline - time.time() - 15
    per_config_time = total_time / n_configs
    
    for i in range(n_configs):
        if time.time() >= deadline - 15:
            break
        
        pts, lns, score = initial_configs[i]
        config_deadline = min(deadline - 15, time.time() + per_config_time)
        
        # Run joint optimization
        pts2, lns2, score2 = _joint_optimize(pts, lns, config_deadline, seed=42 + i)
        if score2 > best_score:
            best_score = score2
            best_points = pts2
            best_lines = lns2
        
        # If time remains, run local search from the result
        if time.time() < config_deadline - 10:
            ls_deadline = min(config_deadline, time.time() + (config_deadline - time.time()) * 0.6)
            pts3, lns3, score3 = _local_search(pts2, lns2, ls_deadline, seed=100 + i)
            if score3 > best_score:
                best_score = score3
                best_points = pts3
                best_lines = lns3
        
        # If still time, run SA
        if time.time() < config_deadline - 5:
            sa_deadline = min(config_deadline, time.time() + (config_deadline - time.time()) * 0.5)
            pts4, lns4, score4 = _simulated_annealing(pts3, lns3, sa_deadline, seed=200 + i)
            if score4 > best_score:
                best_score = score4
                best_points = pts4
                best_lines = lns4
    
    # Final validation
    if not _is_valid(best_points, best_lines):
        for pts, lns, score in initial_configs:
            if _is_valid(pts, lns):
                return pts, lns
        return initial_configs[0][0], initial_configs[0][1]
        
    return best_points, best_lines
