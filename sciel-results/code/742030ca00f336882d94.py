"""Minimize largest 3-collinear-free subset for 20 points (no four collinear).
Uses line-based triple enumeration, greedy set-cover style local search,
and exact branch-and-bound scoring with strong pruning."""
import time
import random
from math import gcd

_DEADLINE = 140.0
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        return 0.0
    return time.time() - _start_time

def _normalize_line(x1, y1, x2, y2):
    """Return canonical (a,b,c) for line through (x1,y1),(x2,y2) with a*x+b*y+c=0."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        a, b, c = 1, 0, -x1
    elif dy == 0:
        a, b, c = 0, 1, -y1
    else:
        g = gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
            dx_n = -dx_n
            dy_n = -dy_n
        a, b = dy_n, -dx_n
        c = -(a * x1 + b * y1)
        g2 = gcd(gcd(abs(a), abs(b)), abs(c))
        if g2 > 0:
            a //= g2
            b //= g2
            c //= g2
    return (a, b, c)

def _get_lines(points):
    """
    Return dict: line_key -> sorted list of point indices on that line.
    Only lines with >= 3 points are included.
    """
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _normalize_line(points[i][0], points[i][1], points[j][0], points[j][1])
            if key not in lines:
                lines[key] = []
            # We'll rebuild below for efficiency
    # Better: build from scratch
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _normalize_line(points[i][0], points[i][1], points[j][0], points[j][1])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    # Filter to lines with >= 3 points
    result = {}
    for key, pts in lines.items():
        if len(pts) >= 3:
            result[key] = sorted(pts)
    return result

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    lines = _get_lines(points)
    for key, pts in lines.items():
        if len(pts) >= 4:
            return False
    return True

def _build_triple_conflicts(points):
    """Build list of collinear triples. Each triple is a frozenset of 3 indices."""
    lines = _get_lines(points)
    triples = []
    seen = set()
    for key, pts in lines.items():
        m = len(pts)
        for a in range(m):
            for b in range(a + 1, m):
                for c in range(b + 1, m):
                    triple = frozenset([pts[a], pts[b], pts[c]])
                    if triple not in seen:
                        seen.add(triple)
                        triples.append(triple)
    return triples

def _max_independent_set_size(triples, n):
    """
    Exact branch and bound for maximum subset with no three collinear.
    Uses strong pruning: if a triple has two selected, the third is forced out.
    """
    if not triples:
        return n

    triples_set = [set(t) for t in triples]
    index_to_triples = [[] for _ in range(n)]
    for idx, t in enumerate(triples):
        for p in t:
            index_to_triples[p].append(idx)

    degrees = [len(index_to_triples[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])

    best = [0]
    current = [False] * n
    # forced_out: points that cannot be selected because two others in a triple are selected
    forced_out = [False] * n

    def backtrack(pos, count):
        if _elapsed() > _DEADLINE - 5:
            return
        if pos == n:
            if count > best[0]:
                best[0] = count
            return

        # Upper bound with forced_out consideration
        remaining = 0
        for p in range(pos, n):
            idx = order[p]
            if not forced_out[idx]:
                remaining += 1
        if count + remaining <= best[0]:
            return

        idx = order[pos]

        # If forced out, skip
        if forced_out[idx]:
            backtrack(pos + 1, count)
            return

        # Try including idx
        can_include = True
        for t_idx in index_to_triples[idx]:
            t = triples_set[t_idx]
            others = [p for p in t if p != idx]
            if len(others) == 2:
                if current[others[0]] and current[others[1]]:
                    can_include = False
                    break

        if can_include:
            current[idx] = True
            # Check if including idx forces any other points out
            new_forced = []
            for t_idx in index_to_triples[idx]:
                t = triples_set[t_idx]
                others = [p for p in t if p != idx]
                if len(others) == 2:
                    if current[others[0]] and current[others[1]]:
                        # This shouldn't happen if can_include was true, but just in case
                        pass
                    # If one of the others is already selected, the other is forced out
                    # Actually, we need to check: if idx is selected and one other is selected,
                    # the third is forced out.
            # Let's redo: for each triple containing idx, if exactly one other is selected,
            # the remaining one is forced out.
            for t_idx in index_to_triples[idx]:
                t = triples_set[t_idx]
                others = [p for p in t if p != idx]
                if len(others) == 2:
                    o1, o2 = others
                    if current[o1] and not current[o2] and not forced_out[o2]:
                        forced_out[o2] = True
                        new_forced.append(o2)
                    elif current[o2] and not current[o1] and not forced_out[o1]:
                        forced_out[o1] = True
                        new_forced.append(o1)

            backtrack(pos + 1, count + 1)

            # Undo
            for f in new_forced:
                forced_out[f] = False
            current[idx] = False

        # Try excluding idx
        backtrack(pos + 1, count)

    backtrack(0, 0)
    return best[0]

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    triples = _build_triple_conflicts(points)
    return _max_independent_set_size(triples, len(points))

def _two_parabola_config(shift=1):
    """Generate 20 points from two shifted parabolas."""
    points = []
    x1_range = list(range(-5, 5))
    x2_range = list(range(-5, 5))

    for x in x1_range:
        points.append([x, x * x])
    for x in x2_range:
        points.append([x, x * x + shift])

    seen = set()
    unique = []
    for p in points:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    if len(unique) != 20:
        return None

    for x, y in unique:
        if abs(x) > 100 or abs(y) > 100:
            return None

    return unique

def _random_config():
    """Generate a random valid configuration."""
    points = []
    seen = set()
    attempts = 0
    while len(points) < 20 and attempts < 10000:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        key = (x, y)
        if key not in seen:
            seen.add(key)
            points.append([x, y])
        attempts += 1

    if len(points) != 20:
        return None
    return points

def _perturb(points):
    """Perturb one or two points slightly."""
    new_points = [list(p) for p in points]
    num_perturb = random.choice([1, 2])
    for _ in range(num_perturb):
        idx = random.randint(0, 19)
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        nx = new_points[idx][0] + dx
        ny = new_points[idx][1] + dy
        if -100 <= nx <= 100 and -100 <= ny <= 100:
            new_points[idx] = [nx, ny]

    seen = set()
    for p in new_points:
        key = tuple(p)
        if key in seen:
            return None
        seen.add(key)
    return new_points

def _greedy_set_cover_search(points, max_iter=5000):
    """
    Greedy set-cover style local search:
    At each step, try replacing one point with a new point that breaks the most
    existing collinear triples while maintaining the no-four-collinear constraint.
    """
    current = [list(p) for p in points]
    current_score = _score(current)
    best = [list(p) for p in current]
    best_score = current_score
    stall = 0

    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 20:
            break

        # Try replacing each point
        best_candidate = None
        best_candidate_score = current_score

        for idx in range(20):
            if _elapsed() > _DEADLINE - 20:
                break
            # Try moving this point to a nearby location
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx = current[idx][0] + dx
                    ny = current[idx][1] + dy
                    if not (-100 <= nx <= 100 and -100 <= ny <= 100):
                        continue
                    # Check uniqueness
                    if any(current[j][0] == nx and current[j][1] == ny for j in range(20) if j != idx):
                        continue

                    candidate = [list(p) for p in current]
                    candidate[idx] = [nx, ny]

                    if not _check_no_four_collinear(candidate):
                        continue

                    s = _score(candidate)
                    if s < best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate

        if best_candidate is not None:
            current = best_candidate
            current_score = best_candidate_score
            if current_score < best_score:
                best_score = current_score
                best = [list(p) for p in current]
            stall = 0
        else:
            stall += 1
            if stall > 50:
                break

    return best, best_score

def _dense_grid_init():
    """Initialize from a dense grid with perturbations to break 4-collinear."""
    points = []
    seen = set()
    for x in range(-4, 5):
        for y in range(-4, 5):
            points.append([x, y])
            seen.add((x, y))

    # We have 81 points, pick 20 with high collinearity potential
    # Actually, let's just pick 20 from a smaller range
    points = []
    seen = set()
    for x in range(-3, 4):
        for y in range(-3, 4):
            if len(points) < 20:
                points.append([x, y])
                seen.add((x, y))

    # Perturb to break 4-collinear
    for attempt in range(100):
        if _check_no_four_collinear(points):
            return points
        # Randomly perturb a point
        idx = random.randint(0, 19)
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        nx = points[idx][0] + dx
        ny = points[idx][1] + dy
        if -100 <= nx <= 100 and -100 <= ny <= 100:
            key = (nx, ny)
            if key not in seen or key == (points[idx][0], points[idx][1]):
                if key != (points[idx][0], points[idx][1]):
                    seen.discard((points[idx][0], points[idx][1]))
                    seen.add(key)
                points[idx] = [nx, ny]

    return None

def construct():
    global _start_time
    _start_time = time.time()

    best_points = None
    best_score = 21

    # Start with two-parabola configuration
    for shift in [1, 2, 3, -1, -2, 5, 7]:
        if _elapsed() > _DEADLINE - 30:
            break
        config = _two_parabola_config(shift)
        if config is None:
            continue
        if not _check_no_four_collinear(config):
            continue
        s = _score(config)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in config]

    # Try dense grid initialization
    if _elapsed() < _DEADLINE - 30:
        grid_config = _dense_grid_init()
        if grid_config is not None:
            s = _score(grid_config)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in grid_config]

    # Local search from best configuration using greedy set-cover
    if best_points is not None:
        if _elapsed() < _DEADLINE - 25:
            improved_points, improved_score = _greedy_set_cover_search(best_points, max_iter=3000)
            if improved_score < best_score:
                best_score = improved_score
                best_points = improved_points

    # Traditional local search
    if best_points is not None and _elapsed() < _DEADLINE - 15:
        current = [list(p) for p in best_points]
        current_score = best_score
        stall_count = 0

        while _elapsed() < _DEADLINE - 10:
            perturbed = _perturb(current)
            if perturbed is None:
                stall_count += 1
                if stall_count > 100:
                    break
                continue

            if not _check_no_four_collinear(perturbed):
                stall_count += 1
                if stall_count > 100:
                    break
                continue

            s = _score(perturbed)
            if s < current_score:
                current = perturbed
                current_score = s
                best_points = [list(p) for p in current]
                best_score = s
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > 200:
                    break

    # Try some random starts if time permits
    if best_points is None:
        best_points = _two_parabola_config(1)
        if best_points is None:
            best_points = [[x, x*x] for x in range(-10, 10)]

    if _elapsed() < _DEADLINE - 10:
        random.seed(42)
        for _ in range(20):
            if _elapsed() > _DEADLINE - 10:
                break
            config = _random_config()
            if config is None:
                continue
            if not _check_no_four_collinear(config):
                continue
            s = _score(config)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in config]

    return best_points
