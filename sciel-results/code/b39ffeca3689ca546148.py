# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _gcd3(a, b, c):
    a, b, c = abs(a), abs(b), abs(c)
    if a == 0 and b == 0 and c == 0:
        return 1
    g = math.gcd(a, b)
    g = math.gcd(g, c)
    return g if g != 0 else 1

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = _gcd3(A, B, C)
    A, B, C = A // g, B // g, C // g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    total = 0
    for A, B, C in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.count_nonzero(vals == 0)
    return int(total)

def _validate(points, lines):
    if points.shape != (54, 2):
        return False
    if lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if not np.all(np.isfinite(lines)):
        return False
    if not np.all(np.abs(points - np.round(points)) < 1e-6):
        return False
    if not np.all(np.abs(lines - np.round(lines)) < 1e-6):
        return False
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != 54:
        return False
    for A, B, C in lines:
        if A == 0 and B == 0:
            return False
    norm_lines = set()
    for A, B, C in lines:
        nl = _normalize_line(int(A), int(B), int(C))
        if nl in norm_lines:
            return False
        norm_lines.add(nl)
    return True

def _generate_candidate_lines_from_points(points):
    """
    Generate candidate lines from point pairs with vectorized incidence counting.
    Returns list of (normalized_line_tuple, incidence_count)
    """
    n = len(points)
    line_to_count = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            if A == 0 and B == 0:
                continue
            nl = _normalize_line(A, B, C)
            if nl not in line_to_count:
                # Vectorized count
                vals = A * points[:, 0] + B * points[:, 1] + C
                count = int(np.count_nonzero(vals == 0))
                line_to_count[nl] = count
    
    return sorted(line_to_count.items(), key=lambda x: -x[1])

def _build_incidence_matrix(points, lines_array):
    """Build (m, n) boolean incidence matrix."""
    A = lines_array[:, 0:1]
    B = lines_array[:, 1:2]
    C = lines_array[:, 2:3]
    xs = points[:, 0:1].T
    ys = points[:, 1:2].T
    vals = A * xs + B * ys + C
    return (vals == 0)

def _local_search_fast(inc_matrix, line_sums, initial_indices, num_lines=27, time_limit=15.0):
    """
    Fast local search. inc_matrix: (m,n) bool. line_sums: array of per-line counts.
    """
    n_lines = len(line_sums)
    start = time.time()
    current = set(initial_indices)
    current_inc = int(inc_matrix[list(current)].sum())
    best_inc = current_inc
    best_set = set(current)
    
    # Precompute which lines are in current for fast lookup
    in_current = np.zeros(n_lines, dtype=bool)
    for idx in current:
        in_current[idx] = True
    
    while time.time() - start < time_limit:
        improved = False
        cur_list = list(current)
        for i in cur_list:
            # Find best replacement
            best_j = -1
            best_new_inc = best_inc
            for j in range(n_lines):
                if in_current[j]:
                    continue
                # Quick upper bound
                est = current_inc - line_sums[i] + line_sums[j]
                if est <= best_inc:
                    continue
                new_set = set(current)
                new_set.discard(i)
                new_set.add(j)
                actual_inc = int(inc_matrix[list(new_set)].sum())
                if actual_inc > best_new_inc:
                    best_new_inc = actual_inc
                    best_j = j
            if best_j >= 0 and best_new_inc > best_inc:
                current.discard(i)
                current.add(best_j)
                in_current[i] = False
                in_current[best_j] = True
                current_inc = best_new_inc
                best_inc = best_new_inc
                best_set = set(current)
                improved = True
        if not improved:
            break
    
    return best_set, best_inc

def _simulated_annealing_fast(inc_matrix, line_sums, num_lines=27, time_limit=500.0):
    """Simulated annealing with fast incidence counting."""
    n_lines = len(line_sums)
    start = time.time()
    
    initial = list(range(min(num_lines, n_lines)))
    current = set(initial)
    current_inc = int(inc_matrix[list(current)].sum())
    best = set(current)
    best_inc = current_inc
    
    in_current = np.zeros(n_lines, dtype=bool)
    for idx in current:
        in_current[idx] = True
    
    T = 20.0
    T_end = 0.01
    cooling = 0.9998
    
    while T > T_end:
        if time.time() - start > time_limit - 3:
            break
        
        i = list(current)[np.random.randint(len(current))]
        j = np.random.randint(n_lines)
        if in_current[j]:
            continue
        
        delta_est = -line_sums[i] + line_sums[j]
        if delta_est >= 0 or (delta_est > -200 and np.random.random() < math.exp(delta_est / T)):
            new_set = set(current)
            new_set.discard(i)
            new_set.add(j)
            actual_inc = int(inc_matrix[list(new_set)].sum())
            current = new_set
            current_inc = actual_inc
            in_current[i] = False
            in_current[j] = True
            if current_inc > best_inc:
                best = set(current)
                best_inc = current_inc
        
        T *= cooling
    
    return best, best_inc

def _try_grid(rows, cols, max_time=580.0):
    if rows * cols != 54:
        return None
    
    xs = np.arange(1, rows + 1)
    ys = np.arange(1, cols + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    sorted_lines = _generate_candidate_lines_from_points(points)
    if len(sorted_lines) < 27:
        return None
    
    n_lines = len(sorted_lines)
    lines_array = np.array([sorted_lines[i][0] for i in range(n_lines)], dtype=np.int64)
    inc_matrix = _build_incidence_matrix(points, lines_array)
    line_sums = np.array([sorted_lines[i][1] for i in range(n_lines)], dtype=np.int64)
    
    num_lines = 27
    greedy = list(range(num_lines))
    greedy_inc = int(inc_matrix[greedy].sum())
    best_set = set(greedy)
    best_inc = greedy_inc
    
    # Local search
    t0 = time.time()
    ls_time = max(3, min(15, max_time - (time.time() - t0) - 20))
    ls_set, ls_inc = _local_search_fast(inc_matrix, line_sums, greedy, num_lines, time_limit=ls_time)
    if ls_inc > best_inc:
        best_set = ls_set
        best_inc = ls_inc
    
    # Simulated annealing
    t1 = time.time()
    remaining = max_time - (time.time() - t1)
    if remaining > 20:
        sa_set, sa_inc = _simulated_annealing_fast(inc_matrix, line_sums, num_lines, time_limit=remaining - 5)
        if sa_inc > best_inc:
            best_set = sa_set
            best_inc = sa_inc
    
    lines_arr = np.array([sorted_lines[i][0] for i in sorted(best_set)], dtype=np.int64)
    return points, lines_arr, best_inc

def szemeredi_trotter_incidences():
    start_time = time.time()
    max_time = 590.0
    
    grids_to_try = [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)]
    
    best_result = None
    best_score = -1
    
    for rows, cols in grids_to_try:
        elapsed = time.time() - start_time
        if elapsed > max_time - 60:
            break
        time_budget = max_time - 60 - elapsed
        if time_budget < 30:
            break
        result = _try_grid(rows, cols, max_time=time_budget)
        if result is not None:
            points, lines, score = result
            if score > best_score:
                best_score = score
                best_result = (points, lines)
    
    if best_result is None:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        for c in [0, 1, 2, 3, -1, 4]:
            lines.append((1, -1, c))
        for c in [7, 8, 9, 10, 6, 11]:
            lines.append((1, 1, -c))
        lines = np.array(lines, dtype=np.int64)
        best_result = (points, lines)
    
    points, lines = best_result
    
    if not _validate(points, lines):
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        for c in [0, 1, 2, 3, -1, 4]:
            lines.append((1, -1, c))
        for c in [7, 8, 9, 10, 6, 11]:
            lines.append((1, 1, -c))
        lines = np.array(lines, dtype=np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
