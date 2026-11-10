# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _count_incidents(points, lines):
    """Count incidences: pairs (p, l) such that A*x + B*y + C == 0."""
    # points: (n, 2) int64, lines: (m, 3) int64
    # Compute A*x + B*y + C for all pairs
    # Use broadcasting: points (n,1,2) * lines (1,m,3) -> (n,m,3) sum -> (n,m)
    # To avoid huge memory, do it in chunks or just compute directly.
    # n=54, m=27, so 54*27 = 1458 pairs, very small.
    # points[:, 0:1, :] * lines[None, :, 0:2] ... no, let's do it simply.
    # A*x + B*y + C
    # lines[:, 0:1] * points[:, 0:1]  # (m, 1) * (n, 1) -> need (n, m)
    # Let's transpose lines to (3, m) and points to (2, n)
    p = points.T  # (2, n)
    l = lines.T   # (3, m)
    # A*x + B*y + C
    # l[0] is A, l[1] is B, l[2] is C
    # p[0] is x, p[1] is y
    # A*x: l[0] (1,m) * p[0] (n,1) -> (n, m)
    # B*y: l[1] (1,m) * p[1] (n,1) -> (n, m)
    # C:   l[2] (1,m) + 0 (n,1)     -> (n, m)
    val = l[0][None, :] * p[0][:, None] + l[1][None, :] * p[1][:, None] + l[2][None, :]
    return int(np.sum(val == 0))


def _line_key(A, B, C):
    """Normalize (A, B, C) to a canonical key."""
    g = math.gcd(abs(int(A)), math.gcd(abs(int(B)), abs(int(C))))
    if g == 0:
        g = 1
    A, B, C = int(A) // g, int(B) // g, int(C) // g
    # Make first nonzero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)


def _valid_lines(points, lines):
    """Check if lines are valid (no duplicates, no A=B=0)."""
    seen = set()
    for i in range(len(lines)):
        A, B, C = int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2])
        if A == 0 and B == 0:
            return False
        key = _line_key(A, B, C)
        if key in seen:
            return False
        seen.add(key)
    return True


def _valid_points(points):
    """Check if points are valid (no duplicates)."""
    seen = set()
    for i in range(len(points)):
        x, y = int(points[i, 0]), int(points[i, 1])
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True


def _try_improve(points, lines, max_steps=50000):
    """Local search: try swapping a line for a better one, or moving a point."""
    n_points = len(points)
    n_lines = len(lines)
    
    # Precompute incidences
    current_inc = _count_incidents(points, lines)
    
    # For each line, compute which points it hits
    # lines: (m, 3), points: (n, 2)
    # For speed, we can precompute for each line a set of points it hits
    # But with 27 lines and 54 points, it's small.
    
    # Local search strategy:
    # 1. For each line, try replacing it with a line through 2 or 3 points.
    # 2. For each point, try moving it to a new location.
    
    # Given time constraints, a simple heuristic:
    # Try to find lines with high incidence and ensure they are in the set.
    
    # Let's do a simple hill climbing:
    # At each step, for a random line, try replacing it with a line defined by 2 random points.
    # Or try adding a line that covers more points.
    
    # Actually, with 54 points, the number of lines determined by pairs of points is C(54,2) = 1431.
    # We can enumerate all these lines, count their incidences, and pick the best 27.
    
    # This is a global optimization over lines, given fixed points.
    # Let's do this:
    # 1. Enumerate all lines from pairs of points.
    # 2. For each line, count incidences.
    # 3. Sort by incidence (descending).
    # 4. Pick the top 27 lines (ensuring no duplicates).
    
    # But this is O(n^2 * n) = O(n^3) = 54^3 ~ 150k, which is fast.
    
    # Let's implement this.
    
    def best_lines_for_points(points):
        n = len(points)
        # Enumerate all lines from pairs
        line_dict = {}  # key -> (inc_count, (A,B,C))
        for i in range(n):
            xi, yi = int(points[i, 0]), int(points[i, 1])
            for j in range(i + 1, n):
                xj, yj = int(points[j, 0]), int(points[j, 1])
                # Line through (xi, yi) and (xj, yj)
                # A = yj - yi, B = xi - xj, C = xj*yi - xi*yj
                A = yj - yi
                B = xi - xj
                C = xj * yi - xi * yj
                if A == 0 and B == 0:
                    continue
                key = _line_key(A, B, C)
                if key in line_dict:
                    continue
                # Count incidences
                inc = 0
                for k in range(n):
                    xk, yk = int(points[k, 0]), int(points[k, 1])
                    if A * xk + B * yk + C == 0:
                        inc += 1
                line_dict[key] = (inc, (key[0], key[1], key[2]))
        
        # Also include vertical and horizontal lines if not already included
        # Actually, the pair enumeration covers all lines with at least 2 points.
        # But what about lines with only 1 point? They have inc=1, which is not useful.
        
        # Sort by inc_count descending
        items = sorted(line_dict.items(), key=lambda x: x[1][0], reverse=True)
        
        # Pick top 27 unique lines
        selected = []
        seen_keys = set()
        for key, (inc, coeffs) in items:
            if len(selected) >= 27:
                break
            if key not in seen_keys:
                selected.append(np.array(coeffs, dtype=np.int64))
                seen_keys.add(key)
        
        # If we have fewer than 27 lines, add arbitrary lines
        while len(selected) < 27:
            # Add a line through a random point with random slope
            # Or just add (1, 0, 0), (0, 1, 0), etc.
            # But we need to ensure they are valid and unique.
            # For simplicity, add lines like (1, 0, -x) for some x
            added = False
            for x in range(-100, 100):
                for y in range(-100, 100):
                    A, B, C = 1, 0, -x
                    key = _line_key(A, B, C)
                    if key not in seen_keys:
                        selected.append(np.array([A, B, C], dtype=np.int64))
                        seen_keys.add(key)
                        added = True
                        break
                if added:
                    break
            if not added:
                break
        
        return np.array(selected, dtype=np.int64)
    
    # Use the best lines for the given points
    lines = best_lines_for_points(points)
    new_inc = _count_incidents(points, lines)
    
    if new_inc > current_inc:
        return points, lines, new_inc
    
    return points, lines, current_inc


def _improve_points(points, lines, max_steps=1000):
    """Try to improve points by moving them to better positions."""
    n = len(points)
    current_inc = _count_incidents(points, lines)
    
    # For each point, try moving it to a new position
    # New position should be on a line with high incidence, or create a new line
    # This is complex, so let's do a simple random search
    
    best_points = points.copy()
    best_inc = current_inc
    
    # Try moving each point to a random nearby position
    for _ in range(max_steps):
        i = np.random.randint(n)
        old_x, old_y = int(best_points[i, 0]), int(best_points[i, 1])
        
        # Try moving to a position that lies on one of the lines
        # Pick a random line and a random point on it
        line_idx = np.random.randint(len(lines))
        A, B, C = int(lines[line_idx, 0]), int(lines[line_idx, 1]), int(lines[line_idx, 2])
        
        if A != 0:
            # x is free, y = -(A*x + C) / B
            # Need B != 0 for this to work
            if B != 0:
                # Choose a random x, compute y
                x = old_x + np.random.randint(-5, 6)
                y_num = -(A * x + C)
                if y_num % B == 0:
                    y = y_num // B
                else:
                    continue
            else:
                # A*x + C = 0 => x = -C/A
                if C % A == 0:
                    x = -C // A
                    y = old_y + np.random.randint(-5, 6)
                else:
                    continue
        else:
            # B*y + C = 0 => y = -C/B
            if C % B == 0:
                y = -C // B
                x = old_x + np.random.randint(-5, 6)
            else:
                continue
        
        # Check if (x, y) is already used
        if any(int(best_points[k, 0]) == x and int(best_points[k, 1]) == y for k in range(n) if k != i):
            continue
        
        # Try the move
        new_points = best_points.copy()
        new_points[i, 0] = x
        new_points[i, 1] = y
        
        new_inc = _count_incidents(new_points, lines)
        if new_inc > best_inc:
            best_points = new_points
            best_inc = new_inc
    
    return best_points, best_inc


def _clique_star_hybrid():
    """
    Clique-star hybrid construction:
    - 27 points on a vertical line x=0, y=1..27
    - 27 points on a 3x9 grid: x=1..3, y=1..9
    - Then optimize lines.
    """
    # Points on x=0
    p0 = np.array([(0, y) for y in range(1, 28)], dtype=np.int64)
    
    # Points on 3x9 grid
    grid_points = []
    for x in range(1, 4):
        for y in range(1, 10):
            grid_points.append((x, y))
    p1 = np.array(grid_points, dtype=np.int64)
    
    points = np.vstack([p0, p1])
    
    # Initial lines:
    # 1. x=0 (the vertical line through p0)
    # 2. x=1, x=2, x=3 (vertical lines through grid)
    # 3. y=1..9 (horizontal lines through grid)
    # 4. Diagonals and other lines
    
    lines = []
    # x=0
    lines.append([1, 0, 0])
    # x=1, x=2, x=3
    for x in [1, 2, 3]:
        lines.append([1, 0, -x])
    # y=1..9
    for y in range(1, 10):
        lines.append([0, 1, -y])
    
    # That's 1 + 3 + 9 = 13 lines. Need 14 more.
    # Add lines through pairs of points
    # Use the best_lines_for_points function
    
    # First, let's just use the best lines algorithm
    lines = _best_lines_for_points_impl(points)
    
    return points, lines


def _best_lines_for_points_impl(points):
    """Find the best 27 lines for given points."""
    n = len(points)
    line_dict = {}
    
    # Enumerate all lines from pairs of points
    for i in range(n):
        xi, yi = int(points[i, 0]), int(points[i, 1])
        for j in range(i + 1, n):
            xj, yj = int(points[j, 0]), int(points[j, 1])
            A = yj - yi
            B = xi - xj
            C = xj * yi - xi * yj
            if A == 0 and B == 0:
                continue
            key = _line_key(A, B, C)
            if key in line_dict:
                continue
            # Count incidences
            inc = 0
            for k in range(n):
                xk, yk = int(points[k, 0]), int(points[k, 1])
                if A * xk + B * yk + C == 0:
                    inc += 1
            line_dict[key] = (inc, (key[0], key[1], key[2]))
    
    # Sort by incidence descending
    items = sorted(line_dict.items(), key=lambda x: x[1][0], reverse=True)
    
    # Pick top 27 unique lines
    selected = []
    seen_keys = set()
    for key, (inc, coeffs) in items:
        if len(selected) >= 27:
            break
        if key not in seen_keys:
            selected.append(np.array(coeffs, dtype=np.int64))
            seen_keys.add(key)
    
    # If we have fewer than 27 lines, add arbitrary lines
    while len(selected) < 27:
        added = False
        for x in range(-50, 50):
            for y in range(-50, 50):
                A, B, C = 1, 1, -(x + y)
                key = _line_key(A, B, C)
                if key not in seen_keys:
                    selected.append(np.array([A, B, C], dtype=np.int64))
                    seen_keys.add(key)
                    added = True
                    break
            if added:
                break
        if not added:
            break
    
    return np.array(selected, dtype=np.int64)


def _grid_construction():
    """
    Improved grid construction:
    Use a 3x18 grid (54 points) and find the best 27 lines.
    Or use a different configuration.
    """
    # Let's try a 3x18 grid
    xs = [1, 2, 3]
    ys = list(range(1, 19))
    points = []
    for x in xs:
        for y in ys:
            points.append((x, y))
    points = np.array(points, dtype=np.int64)
    
    lines = _best_lines_for_points_impl(points)
    
    return points, lines


def _random_search(max_iters=1000):
    """Random search over point configurations."""
    best_points = None
    best_lines = None
    best_inc = -1
    
    for _ in range(max_iters):
        # Random point configuration
        # Use a grid-like structure with some randomness
        xs = list(range(1, 6))
        ys = list(range(1, 12))
        points = []
        for x in xs:
            for y in ys:
                if np.random.rand() < 0.7:  # 70% chance to include
                    points.append((x + np.random.randint(-1, 2), y + np.random.randint(-1, 2)))
        
        # Ensure exactly 54 unique points
        unique_points = list(set(points))
        if len(unique_points) < 54:
            # Add more random points
            while len(unique_points) < 54:
                x = np.random.randint(1, 20)
                y = np.random.randint(1, 20)
                if (x, y) not in unique_points:
                    unique_points.append((x, y))
            unique_points = unique_points[:54]
        
        points = np.array(unique_points, dtype=np.int64)
        
        # Find best lines
        lines = _best_lines_for_points_impl(points)
        
        inc = _count_incidents(points, lines)
        if inc > best_inc:
            best_inc = inc
            best_points = points
            best_lines = lines
    
    return best_points, best_lines, best_inc


def _deterministic_good_construction():
    """
    A carefully designed construction that should yield high incidence.
    
    Idea: Use multiple parallel lines with many points on each.
    For example, 6 vertical lines, each with 9 points.
    Then the vertical lines give 6*9 = 54 incidences.
    Plus horizontal lines: 9 horizontal lines, each with 6 points.
    That's another 54 incidences.
    Plus diagonals.
    """
    # 6x9 grid
    xs = list(range(1, 7))
    ys = list(range(1, 10))
    points = []
    for x in xs:
        for y in ys:
            points.append((x, y))
    points = np.array(points, dtype=np.int64)
    
    lines = _best_lines_for_points_impl(points)
    
    return points, lines


def _try_all_constructions():
    """Try multiple constructions and return the best one."""
    candidates = []
    
    # 1. Original Szemeredi-Trotter
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points_st = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines_st = np.array(rows, dtype=np.int64)
    candidates.append((points_st, lines_st, _count_incidents(points_st, lines_st)))
    
    # 2. Clique-star hybrid
    points_cs, lines_cs = _clique_star_hybrid()
    candidates.append((points_cs, lines_cs, _count_incidents(points_cs, lines_cs)))
    
    # 3. 3x18 grid
    points_g1, lines_g1 = _grid_construction()
    candidates.append((points_g1, lines_g1, _count_incidents(points_g1, lines_g1)))
    
    # 4. 6x9 grid
    points_g2, lines_g2 = _deterministic_good_construction()
    candidates.append((points_g2, lines_g2, _count_incidents(points_g2, lines_g2)))
    
    # 5. Random search (limited)
    if np.random.randint(0, 2):  # 50% chance to do random search
        points_rs, lines_rs, inc_rs = _random_search(max_iters=100)
        candidates.append((points_rs, lines_rs, inc_rs))
    
    # Return the best candidate
    best = max(candidates, key=lambda x: x[2])
    return best[0], best[1], best[2]


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct a set of n points and a set of
    m lines in the plane (on the integer lattice) so as to maximize the number of
    point-line incidences (pairs (p, l) with p lying on l). Here the instance is fixed
    at n = 54 points and m = 27 lines.

    Returns:
        points: np.ndarray of shape (54, 2), integer (x, y) lattice coordinates.
        lines : np.ndarray of shape (27, 3), integer (A, B, C) with A*x + B*y + C = 0.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds deadline
    
    # Try the deterministic constructions first
    points, lines, inc = _try_all_constructions()
    
    # Validate
    if not _valid_points(points) or not _valid_lines(lines):
        # Fall back to the original construction
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
    
    # Try to improve with local search if time allows
    elapsed = time.time() - start_time
    if elapsed < 500:  # Leave 100 seconds safety margin
        # Try improving points
        points, inc = _improve_points(points, lines, max_steps=1000)
        
        # Re-optimize lines
        lines = _best_lines_for_points_impl(points)
        inc = _count_incidents(points, lines)
    
    # Final validation
    if not _valid_points(points) or not _valid_lines(lines):
        # Fall back to the original construction
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
    
    # Ensure shapes are correct
    points = points.astype(np.int64)
    lines = lines.astype(np.int64)
    
    return points, lines


# EVOLVE-BLOCK-END
