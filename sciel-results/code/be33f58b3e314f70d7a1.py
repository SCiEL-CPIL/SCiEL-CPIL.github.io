"""Structural redesign: randomized greedy + local search for minimal maximal no-isosceles set."""
import random
import time
import itertools

N = 16
NUM_POINTS = N * N
DEADLINE = 250.0  # safety margin before 330s kill


def _build_distance_matrix():
    """Precompute squared-distance matrix D[i,j] for all grid points."""
    pts = [(x, y) for x in range(N) for y in range(N)]
    D = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        xi, yi = pts[i]
        for j in range(i + 1, NUM_POINTS):
            xj, yj = pts[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            D[i][j] = d
            D[j][i] = d
    return pts, D


PTS, D = _build_distance_matrix()


def _can_add(p, chosen):
    """Check if point p can be added to chosen without creating isosceles triangle.
    For every pair (a,b) in chosen, D[p,a], D[p,b], D[a,b] must all be distinct."""
    for i in range(len(chosen)):
        a = chosen[i]
        dpa = D[p][a]
        for j in range(i + 1, len(chosen)):
            b = chosen[j]
            d_ab = D[a][b]
            d_pb = D[p][b]
            if dpa == d_pb or dpa == d_ab or d_pb == d_ab:
                return False
    return True


def _randomized_greedy(order, chosen=None):
    """Greedy add in given order. Returns maximal set."""
    if chosen is None:
        chosen = []
    for p in order:
        if p not in chosen:
            if _can_add(p, chosen):
                chosen.append(p)
    return chosen


def _local_search(current, deadline):
    """Try dropping one point and re-greedy to get smaller maximal set.
    Also try swap moves."""
    best = current
    best_len = len(current)
    n = len(current)
    # Try each single-drop + re-greedy
    for drop_idx in range(n):
        if time.time() > deadline:
            break
        # Remove point at drop_idx
        reduced = current[:drop_idx] + current[drop_idx + 1:]
        # Re-greedy: try to add back other points in random order
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        new_set = _randomized_greedy(order, reduced)
        if len(new_set) < best_len:
            best_len = len(new_set)
            best = new_set
    return best


def construct():
    start = time.time()
    deadline = start + DEADLINE

    best = None
    best_len = NUM_POINTS + 1

    # Multiple randomized greedy attempts
    attempts = 0
    while time.time() < deadline:
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        s = _randomized_greedy(order)
        if len(s) < best_len:
            best_len = len(s)
            best = s

        # Local search on the best found so far
        if best is not None and len(best) < best_len + 1:
            improved = _local_search(best, deadline)
            if len(improved) < best_len:
                best_len = len(improved)
                best = improved

        attempts += 1
        # After some greedy attempts, focus more on local search
        if attempts % 5 == 0 and best is not None:
            improved = _local_search(best, deadline)
            if len(improved) < best_len:
                best_len = len(improved)
                best = improved

    if best is None:
        # Fallback: deterministic greedy
        best = _randomized_greedy(list(range(NUM_POINTS)))

    return best
