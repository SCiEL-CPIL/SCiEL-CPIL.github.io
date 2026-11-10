# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients so first nonzero is positive and gcd divides all."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact incidences between points and lines using int64 arithmetic."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    # For each line, check which points satisfy A*x + B*y + C == 0
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        # Compute A*x + B*y + C for all points
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _generate_candidate_lines(points):
    """Generate candidate lines from pairs of points, normalized."""
    n = len(points)
    candidate_set = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2): A*x + B*y + C = 0
            # A = y2 - y1, B = x1 - x2, C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm is not None:
                candidate_set[norm] = norm
    return list(candidate_set.values())

def _compute_line_weights(points, lines):
    """Compute incidence count for each line."""
    pts = np.asarray(points, dtype=np.int64)
    weights = np.zeros(len(lines), dtype=np.int64)
    for i in range(len(lines)):
        A, B, C = lines[i]
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        weights[i] = np.sum(vals == 0)
    return weights

def _build_conflict_matrix(points, lines):
    """
    Build a conflict matrix where conflict[i,j] = True if lines i and j
    share more than one point (i.e., they are the same geometric line).
    Actually, for our purpose: two lines conflict if they share >= 2 points,
    which means they are the same line. Since we've normalized, this shouldn't happen.
    But we also want to avoid selecting redundant lines. Let's define conflict as:
    two lines are in conflict if they share 2 or more points (same line).
    """
    n = len(lines)
    # Precompute which points each line passes through
    line_points = []
    pts = np.asarray(points, dtype=np.int64)
    for i in range(n):
        A, B, C = lines[i]
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        mask = vals == 0
        line_points.append(set(np.where(mask)[0].tolist()))
    
    # Conflict: same line (share all their points, or at least 2)
    # Actually, if two different normalized lines share 2+ points, they're the same line.
    # So conflicts are really duplicates. But let's also consider: we want to maximize
    # total incidences. Two lines that share many points are "redundant" in some sense,
    # but they still each count their own incidences. The real constraint is just:
    # no two lines should be identical (same normalized form).
    # Since we already deduplicated, there are no conflicts in the traditional sense.
    # Let's just return an empty conflict structure.
    return None, line_points

def _branch_and_bound_select(points, lines, weights, k, deadline):
    """
    Select k lines from candidates to maximize total weight.
    Uses branch and bound with greedy upper bound.
    """
    n = len(lines)
    # Sort by weight descending
    order = np.argsort(-weights)
    sorted_lines = [lines[i] for i in order]
    sorted_weights = weights[order]
    
    best_total = 0
    best_selection = list(range(k))  # Initial: top k by weight
    
    # Compute initial solution
    current_total = sum(sorted_weights[:k])
    best_total = current_total
    best_selection = list(range(k))
    
    # For branch and bound, we need a good upper bound.
    # Simple bound: sum of top (k - selected) weights among remaining.
    # This is expensive to compute dynamically. Let's use a simpler approach:
    # Greedy with local search.
    
    # Actually, let's do a more careful branch and bound with memoization.
    # But given time constraints, let's use a heuristic approach:
    # 1. Start with top-k by weight
    # 2. Try swapping each selected line with each unselected line
    # 3. Accept if improvement
    
    selected = set(range(k))
    current_total = sum(sorted_weights[i] for i in selected)
    
    improved = True
    while improved:
        improved = False
        for i in list(selected):
            for j in range(n):
                if j in selected:
                    continue
                # Try swapping i with j
                new_total = current_total - sorted_weights[i] + sorted_weights[j]
                if new_total > current_total:
                    selected.remove(i)
                    selected.add(j)
                    current_total = new_total
                    best_total = current_total
                    best_selection = list(selected)
                    improved = True
                    break
            if improved:
                break
    
    return best_total, best_selection

def _local_search_lines(points, initial_lines, k, deadline):
    """
    Local search to optimize line selection.
    """
    pts = np.asarray(points, dtype=np.int64)
    n_pts = len(pts)
    
    # Generate all candidate lines
    candidates = _generate_candidate_lines(pts)
    n_cand = len(candidates)
    
    # Compute weights
    weights = _compute_line_weights(pts, candidates)
    
    # Start with initial lines
    # Map initial lines to candidate indices
    initial_norms = set()
    for line in initial_lines:
        norm = _normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if norm:
            initial_norms.add(norm)
    
    # Find which candidates match initial lines
    selected = set()
    for i, cand in enumerate(candidates):
        if cand in initial_norms:
            selected.add(i)
    
    # If we don't have k lines, add top-weight candidates
    if len(selected) < k:
        remaining = [i for i in range(n_cand) if i not in selected]
        remaining.sort(key=lambda i: -weights[i])
        for i in remaining:
            if len(selected) >= k:
                break
            selected.add(i)
    
    # Local search: try swaps
    current_total = sum(weights[i] for i in selected)
    best_total = current_total
    best_selected = set(selected)
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        for i in list(selected):
            for j in range(n_cand):
                if j in selected:
                    continue
                delta = weights[j] - weights[i]
                if delta > 0:
                    selected.remove(i)
                    selected.add(j)
                    current_total += delta
                    if current_total > best_total:
                        best_total = current_total
                        best_selected = set(selected)
                    improved = True
                    break
            if improved:
                break
    
    # Convert back to line tuples
    result_lines = [candidates[i] for i in sorted(best_selected)]
    return result_lines, best_total

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    Uses a 6x9 grid with optimized line selection via local search.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    # 1. Generate Points: 6x9 grid
    xs = np.arange(1, 7)  # 1 to 6
    ys = np.arange(1, 10) # 1 to 9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Initial line set (from original design)
    initial_lines = []
    # Horizontal lines
    for c in range(1, 10):
        initial_lines.append((0, 1, -c))
    # Vertical lines
    for c in range(1, 7):
        initial_lines.append((1, 0, -c))
    # Slope 1 lines
    for c in [0, 1, 2, 3, -1, 4]:
        initial_lines.append((-1, 1, -c))
    # Slope -1 lines
    for c in [7, 8, 9, 10, 6, 11]:
        initial_lines.append((1, 1, -c))
    
    initial_lines = np.array(initial_lines, dtype=np.int64)
    
    # 3. Local search to optimize line selection
    if time.time() < deadline - 30:
        optimized_lines, _ = _local_search_lines(points, initial_lines, 27, deadline)
        if len(optimized_lines) == 27:
            lines = np.array(optimized_lines, dtype=np.int64)
        else:
            lines = initial_lines
    else:
        lines = initial_lines
    
    # 4. Final incidence count
    final_inc = _count_incidences(points, lines)
    
    return points, lines

# EVOLVE-BLOCK-END
