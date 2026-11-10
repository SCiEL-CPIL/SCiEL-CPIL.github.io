# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from itertools import combinations

_DEADLINE_MARGIN = 30.0  # seconds of safety margin before the 630 s wall-clock kill


def _incidence_count(points, lines):
    """Exact int64 count of (point, line) pairs with A*x + B*y + C == 0."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    # points: (n, 2) int64, lines: (m, 3) int64
    # Compute A*x + B*y + C for every pair using int64 arithmetic.
    # To avoid overflow for large coords, we can chunk; but our coords are small.
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    x = points[:, 0]
    y = points[:, 1]
    # Outer products: (m, n)
    val = np.outer(A, x) + np.outer(B, y) + C[:, None]
    return int(np.count_nonzero(val == 0))


def _line_key(a, b, c):
    """Canonical form of line (a,b,c): divide by gcd, make first nonzero positive."""
    if a == 0 and b == 0 and c == 0:
        return (0, 0, 0)
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        return (0, 0, 0)
    a, b, c = a // g, b // g, c // g
    # Make first nonzero entry positive
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _is_valid(points, lines):
    """Check all validity constraints."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    # Check for NaN/inf (shouldn't happen with int64 but be safe)
    if not np.all(np.isfinite(points.astype(float))):
        return False
    if not np.all(np.isfinite(lines.astype(float))):
        return False
    # Check entries are close to integers (they are int64, so always true)
    # Check for repeated points
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
    # Check lines: no A=B=0
    for i in range(27):
        if lines[i, 0] == 0 and lines[i, 1] == 0:
            return False
    # Check no duplicate lines (after canonicalization)
    keys = set()
    for i in range(27):
        k = _line_key(int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2]))
        if k in keys:
            return False
        keys.add(k)
    return True


def _candidate_lines_from_points(points, max_coeff, max_lines_per_slope):
    """
    Generate candidate lines that pass through at least 2 points, with small integer
    coefficients. Returns a list of (A, B, C) tuples.
    """
    n = len(points)
    cand = {}  # key -> (A, B, C)
    for i in range(n):
        x1, y1 = points[i, 0], points[i, 1]
        for j in range(i + 1, n):
            x2, y2 = points[j, 0], points[j, 1]
            dx = x2 - x1
            dy = y2 - y1
            # Line through (x1,y1) and (x2,y2): dy*(x - x1) - dx*(y - y1) = 0
            # => dy*x - dx*y + (dx*y1 - dy*x1) = 0
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            # Reduce by gcd
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g == 0:
                continue
            a, b, c = a // g, b // g, c // g
            # Make first nonzero positive
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in cand:
                cand[key] = (a, b, c)
    return list(cand.values())


def _select_best_lines(points, num_lines, max_coeff, time_limit):
    """
    Greedily select `num_lines` lines that maximize total incidences with `points`.
    Uses a priority-queue-like approach: at each step, pick the line with the most
    incidences among remaining points not yet "covered" by more than some threshold.
    
    Actually, a better approach: greedy with replacement.
    At each step, compute for each candidate line how many of the 54 points it covers.
    Pick the best, add it, and continue. But this is myopic.
    
    Alternative: start with all candidate lines, then iteratively replace the worst line
    with the best candidate not already selected.
    """
    start = time.time()
    n = len(points)
    
    # Generate all candidate lines through pairs of points
    candidates = _candidate_lines_from_points(points, max_coeff, None)
    
    if len(candidates) < num_lines:
        # Pad with axis-parallel lines if needed
        extra = []
        for x in range(-max_coeff, max_coeff + 1):
            extra.append((0, 1, -x))  # x = const
        for y in range(-max_coeff, max_coeff + 1):
            extra.append((1, 0, -y))  # y = const
        candidates.extend(extra)
        # Deduplicate
        seen = set()
        deduped = []
        for c in candidates:
            k = _line_key(c[0], c[1], c[2])
            if k not in seen:
                seen.add(k)
                deduped.append(c)
        candidates = deduped
    
    # Precompute incidence matrix: (num_candidates, n) boolean
    # For memory efficiency, compute in chunks
    num_cand = len(candidates)
    cand_arr = np.array(candidates, dtype=np.int64)
    
    # Compute which points each line hits
    # incidence[i, j] = 1 if point j lies on line i
    # Do it in chunks of lines to save memory
    chunk_size = 1000
    incidence = np.zeros((num_cand, n), dtype=np.int8)
    for start_i in range(0, num_cand, chunk_size):
        end_i = min(start_i + chunk_size, num_cand)
        A = cand_arr[start_i:end_i, 0]
        B = cand_arr[start_i:end_i, 1]
        C = cand_arr[start_i:end_i, 2]
        x = points[:, 0]
        y = points[:, 1]
        val = np.outer(A, x) + np.outer(B, y) + C[:, None]
        incidence[start_i:end_i] = (val == 0).astype(np.int8)
    
    # Line scores: number of points each line covers
    scores = incidence.sum(axis=1)
    
    # Greedy selection with local search
    selected = np.zeros(num_lines, dtype=np.int64)
    selected_mask = np.zeros(num_cand, dtype=bool)
    
    # Initial greedy: pick top num_lines by score
    order = np.argsort(-scores)
    for i in range(num_lines):
        selected[i] = order[i]
        selected_mask[order[i]] = True
    
    # Local search: iteratively try replacing each selected line with the best
    # non-selected candidate
    best_total = incidence[selected].sum()
    improved = True
    max_iters = 200
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        if time.time() - start > time_limit:
            break
        for i in range(num_lines):
            if time.time() - start > time_limit:
                break
            current_line = selected[i]
            current_score = scores[current_line]
            # Find best non-selected candidate
            best_candidate = -1
            best_candidate_score = -1
            # To speed up, only consider candidates with score > current_score
            # and not already selected
            for j in range(num_cand):
                if selected_mask[j]:
                    continue
                if scores[j] <= current_score:
                    continue
                # Check if this candidate adds new incidences
                # The gain is: points covered by new line that are NOT covered by any other selected line
                # But that's complex. Simple approximation: use raw score.
                if scores[j] > best_candidate_score:
                    best_candidate_score = scores[j]
                    best_candidate = j
            
            if best_candidate >= 0 and best_candidate_score > current_score:
                selected_mask[current_line] = False
                selected_mask[best_candidate] = True
                selected[i] = best_candidate
                improved = True
    
    return cand_arr[selected], incidence[selected]


def _optimize_points_given_lines(lines, points, time_limit):
    """
    Try to improve the point set given fixed lines, by moving points to positions
    that are covered by more lines.
    """
    start = time.time()
    n = len(points)
    m = len(lines)
    best_points = points.copy()
    best_score = _incidence_count(best_points, lines)
    
    # For each point, try moving it to a nearby lattice position
    # that increases the total incidence count.
    # A point at (x, y) is incident to line i if A_i*x + B_i*y + C_i == 0.
    # Moving point k to (x', y') changes the count by:
    #   (new incidences at (x',y')) - (old incidences at (x,y))
    
    # Precompute for each candidate position, how many lines pass through it.
    # But the space is infinite. Instead, for each point, consider positions
    # that lie on at least 2 lines (since those are the "interesting" positions).
    
    # Find all intersection points of pairs of lines
    intersections = set()
    for i in range(m):
        A1, B1, C1 = int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2])
        for j in range(i + 1, m):
            A2, B2, C2 = int(lines[j, 0]), int(lines[j, 1]), int(lines[j, 2])
            # Solve: A1*x + B1*y = -C1, A2*x + B2*y = -C2
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue
            # Cramer's rule
            x_num = (-C1) * B2 - (-C2) * B1
            y_num = A1 * (-C2) - A2 * (-C1)
            # x = x_num / det, y = y_num / det
            if x_num % det != 0 or y_num % det != 0:
                continue
            x = x_num // det
            y = y_num // det
            # Bound the coordinates to reasonable range
            if abs(x) > 1000 or abs(y) > 1000:
                continue
            intersections.add((x, y))
    
    # Also include current point positions
    for p in points:
        intersections.add((int(p[0]), int(p[1])))
    
    # For each intersection, compute how many lines pass through it
    pos_scores = {}
    for (x, y) in intersections:
        count = 0
        for i in range(m):
            if int(lines[i, 0]) * x + int(lines[i, 1]) * y + int(lines[i, 2]) == 0:
                count += 1
        pos_scores[(x, y)] = count
    
    # Greedy: for each point, try to move it to a position with higher score
    # that isn't already occupied by another point.
    occupied = set(map(tuple, best_points.tolist()))
    improved = True
    max_iters = 50
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        if time.time() - start > time_limit:
            break
        for k in range(n):
            if time.time() - start > time_limit:
                break
            old_x, old_y = int(best_points[k, 0]), int(best_points[k, 1])
            old_score = pos_scores.get((old_x, old_y), 0)
            # Find best alternative position
            best_pos = None
            best_pos_score = old_score
            for (x, y), sc in pos_scores.items():
                if (x, y) == (old_x, old_y):
                    continue
                if (x, y) in occupied:
                    continue
                if sc > best_pos_score:
                    best_pos_score = sc
                    best_pos = (x, y)
            if best_pos is not None:
                occupied.discard((old_x, old_y))
                occupied.add(best_pos)
                best_points[k, 0] = best_pos[0]
                best_points[k, 1] = best_pos[1]
                improved = True
    
    new_score = _incidence_count(best_points, lines)
    if new_score > best_score:
        best_score = new_score
    return best_points, best_score


def _seed_grid_config():
    """Generate the classical Szemeredi-Trotter grid configuration."""
    k = 3
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


def _seed_dense_config():
    """Generate a denser configuration: points on a grid, lines as all grid lines and diagonals."""
    # Try a 6x9 grid of points
    points = []
    for x in range(1, 7):
        for y in range(1, 10):
            points.append((x, y))
    points = np.array(points, dtype=np.int64)
    
    # Lines: horizontal, vertical, and some diagonals
    lines = []
    for y in range(1, 10):
        lines.append((0, 1, -y))  # y = const
    for x in range(1, 7):
        lines.append((1, 0, -x))  # x = const
    # Add some diagonals
    for slope in [1, -1, 2, -2]:
        for intercept in range(-5, 15):
            # slope * x - y + intercept = 0 => y = slope*x + intercept
            # Check if it passes through at least 2 grid points
            count = 0
            for (x, y) in points:
                if slope * x - y + intercept == 0:
                    count += 1
            if count >= 2:
                lines.append((slope, -1, intercept))
    
    # Deduplicate and take first 27
    seen = set()
    deduped = []
    for l in lines:
        k = _line_key(l[0], l[1], l[2])
        if k not in seen:
            seen.add(k)
            deduped.append(l)
    
    # Sort by incidence count
    def line_incidence(l):
        return sum(1 for (x, y) in points if l[0] * x + l[1] * y + l[2] == 0)
    
    deduped.sort(key=line_incidence, reverse=True)
    lines = np.array(deduped[:27], dtype=np.int64)
    return points, lines


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct 54 points and 27 lines
    to maximize point-line incidences.
    
    Strategy: Start with multiple seed configurations, then alternately optimize
    points given lines and lines given points using local search.
    """
    wall_start = time.time()
    
    def time_remaining():
        return 630.0 - (time.time() - wall_start) - _DEADLINE_MARGIN
    
    # Generate initial candidate configurations
    configs = []
    
    # Seed 1: Classical grid
    p1, l1 = _seed_grid_config()
    configs.append((p1.copy(), l1.copy()))
    
    # Seed 2: Dense grid with more line types
    p2, l2 = _seed_dense_config()
    if p2.shape == (54, 2) and l2.shape == (27, 3):
        configs.append((p2.copy(), l2.copy()))
    
    # Seed 3: Try a different grid shape - 9x6
    points3 = []
    for x in range(1, 10):
        for y in range(1, 7):
            points3.append((x, y))
    points3 = np.array(points3, dtype=np.int64)
    # Lines: all lines through pairs, pick best 27
    candidates3 = _candidate_lines_from_points(points3, 100, None)
    cand_arr3 = np.array(candidates3, dtype=np.int64)
    # Score each candidate
    x3 = points3[:, 0]
    y3 = points3[:, 1]
    scores3 = []
    for i in range(len(cand_arr3)):
        val = cand_arr3[i, 0] * x3 + cand_arr3[i, 1] * y3 + cand_arr3[i, 2]
        scores3.append(np.count_nonzero(val == 0))
    scores3 = np.array(scores3)
    order3 = np.argsort(-scores3)
    lines3 = cand_arr3[order3[:27]]
    if len(lines3) == 27:
        configs.append((points3.copy(), lines3.copy()))
    
    # Seed 4: Random-ish perturbation of grid
    points4 = []
    for x in range(1, 7):
        for y in range(1, 10):
            # Slightly perturb some points
            if (x + y) % 5 == 0:
                points4.append((x + 1, y))
            else:
                points4.append((x, y))
    points4 = np.array(points4, dtype=np.int64)
    # Ensure no duplicates
    seen4 = set()
    unique4 = []
    for p in points4:
        t = (int(p[0]), int(p[1]))
        if t not in seen4:
            seen4.add(t)
            unique4.append(p)
    points4 = np.array(unique4[:54], dtype=np.int64)
    if points4.shape == (54, 2):
        candidates4 = _candidate_lines_from_points(points4, 100, None)
        cand_arr4 = np.array(candidates4, dtype=np.int64)
        x4 = points4[:, 0]
        y4 = points4[:, 1]
        scores4 = []
        for i in range(len(cand_arr4)):
            val = cand_arr4[i, 0] * x4 + cand_arr4[i, 1] * y4 + cand_arr4[i, 2]
            scores4.append(np.count_nonzero(val == 0))
        scores4 = np.array(scores4)
        order4 = np.argsort(-scores4)
        lines4 = cand_arr4[order4[:27]]
        if len(lines4) == 27:
            configs.append((points4.copy(), lines4.copy()))
    
    # Evaluate all configs
    best_points = None
    best_lines = None
    best_score = -1
    for p, l in configs:
        if not _is_valid(p, l):
            continue
        sc = _incidence_count(p, l)
        if sc > best_score:
            best_score = sc
            best_points = p.copy()
            best_lines = l.copy()
    
    if best_points is None:
        # Fallback to seed 1
        best_points, best_lines = _seed_grid_config()
        best_score = _incidence_count(best_points, best_lines)
    
    # Alternating optimization
    remaining = time_remaining()
    if remaining < 5:
        return best_points.astype(np.int64), best_lines.astype(np.int64)
    
    num_rounds = 10
    per_round_time = remaining / num_rounds
    for rnd in range(num_rounds):
        if time_remaining() < 10:
            break
        
        # Step 1: Optimize lines given current points
        line_time = min(per_round_time / 2, 30.0)
        selected_lines, line_incidence = _select_best_lines(
            best_points, 27, 100, line_time
        )
        if len(selected_lines) == 27:
            new_lines = selected_lines
            # Validate
            if _is_valid(best_points, new_lines):
                new_score = _incidence_count(best_points, new_lines)
                if new_score > best_score:
                    best_lines = new_lines
                    best_score = new_score
        
        # Step 2: Optimize points given current lines
        point_time = min(per_round_time / 2, 30.0)
        new_points, new_score = _optimize_points_given_lines(
            best_lines, best_points, point_time
        )
        if _is_valid(new_points, best_lines):
            actual_score = _incidence_count(new_points, best_lines)
            if actual_score > best_score:
                best_points = new_points
                best_score = actual_score
        
        # Check if we can do another quick pass
        if time_remaining() < 15:
            break
    
    # Final validation
    if not _is_valid(best_points, best_lines):
        # Fallback
        best_points, best_lines = _seed_grid_config()
    
    return best_points.astype(np.int64), best_lines.astype(np.int64)


# EVOLVE-BLOCK-END
