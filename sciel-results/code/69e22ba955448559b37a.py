"""Exact maximum-edge subgraph extraction for the 32-point triangular lattice problem,
with local-search polish as a safety net."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def _count_edges(points):
    """Count the number of unit-distance pairs among given points on triangular lattice."""
    point_set = set(points)
    edge_count = 0
    for a, b in points:
        for da, db in _NEIGHBORS:
            if (a + da, b + db) in point_set:
                edge_count += 1
    return edge_count // 2

def _harborth_bound(n):
    """Return the Harborth bound floor(3n - sqrt(12n - 3))."""
    return int(math.floor(3 * n - math.sqrt(12 * n - 3)))

def _build_lattice_graph(box=3):
    """Build the unit-distance graph over a bounded hexagonal triangular lattice region.

    Uses axial coordinates (a, b) for point a + b*omega.
    The region is { (a,b) : max(|a|,|b|,|a+b|) <= box }, a hexagon.
    """
    points = []
    for a in range(-box, box + 1):
        for b in range(-box, box + 1):
            if max(abs(a), abs(b), abs(a + b)) <= box:
                points.append((a, b))

    point_to_idx = {p: i for i, p in enumerate(points)}
    npts = len(points)

    forward_dirs = [(1, 0), (0, 1), (1, -1)]
    adj = [set() for _ in range(npts)]

    for i, (a, b) in enumerate(points):
        for da, db in forward_dirs:
            j_point = (a + da, b + db)
            if j_point in point_to_idx:
                j = point_to_idx[j_point]
                adj[i].add(j)
                adj[j].add(i)

    return points, point_to_idx, adj, npts

def _solve_exact_remove(adj, npts, k, target, time_limit, deadline):
    """Find a k-vertex induced subgraph with >= target edges by enumerating removals.

    We remove (npts - k) vertices to minimize edges lost.
    Returns set of selected vertex indices, or None.
    """
    n_remove = npts - k
    if n_remove <= 0:
        return set(range(npts))
    if n_remove > 10:
        return None

    degrees = [len(adj[i]) for i in range(npts)]
    total_edges = sum(degrees) // 2

    best_removed = None
    min_edges_lost = float('inf')

    for combo in combinations(range(npts), n_remove):
        if time.time() > deadline:
            break
        edges_lost = 0
        for v in combo:
            for u in adj[v]:
                if u not in combo or u > v:
                    edges_lost += 1
        if edges_lost < min_edges_lost:
            min_edges_lost = edges_lost
            best_removed = combo
            if total_edges - edges_lost >= target:
                break  # Harborth bound met; cannot do better

    if best_removed is None:
        return None
    return set(range(npts)) - set(best_removed)

def _construct_hexagonal_ring(n=32):
    """Construct points in hexagonal ring pattern; ring 3 selected greedily for max edges."""
    points = [(0, 0)]
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    points.extend(ring1)
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a + b)) == 2:
                points.append((a, b))

    ring3_full = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a + b)) == 3:
                ring3_full.append((a, b))

    existing = set(points)

    def connections(p):
        a, b = p
        return sum(1 for da, db in _NEIGHBORS if (a + da, b + db) in existing)

    need = n - len(points)  # 13
    ring3_sorted = sorted(ring3_full, key=lambda p: (-connections(p), p[0], p[1]))
    for p in ring3_sorted[:need]:
        points.append(p)
        existing.add(p)
    return points

def _local_improve(points, deadline):
    """Local search: move one point at a time to maximize edges."""
    current = points[:]
    point_set = set(current)
    current_edges = _count_edges(current)
    best = current[:]
    best_edges = current_edges
    n = len(current)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            if time.time() >= deadline:
                break
            old = current[idx]
            for da in range(-3, 4):
                for db in range(-3, 4):
                    if da == 0 and db == 0:
                        continue
                    new = (old[0] + da, old[1] + db)
                    if abs(new[0]) > 64 or abs(new[1]) > 64:
                        continue
                    if new in point_set and new != old:
                        continue
                    point_set.discard(old)
                    gained = sum(1 for na, nb in _NEIGHBORS if (new[0] + na, new[1] + nb) in point_set)
                    lost = sum(1 for na, nb in _NEIGHBORS if (old[0] + na, old[1] + nb) in point_set)
                    delta = gained - lost
                    point_set.add(old)
                    if delta > 0:
                        point_set.discard(old)
                        point_set.add(new)
                        current[idx] = new
                        current_edges += delta
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best = current[:]
                        improved = True
                        break
                if current[idx] != old:
                    break
    return best, best_edges

def construct():
    """Construct 32 distinct lattice points maximizing unit-distance pairs."""
    deadline = time.time() + 115
    n = 32
    target = _harborth_bound(n)  # 76

    # Primary: exact enumeration over the 37-point hexagon (remove 5).
    points_list, point_to_idx, adj, npts = _build_lattice_graph(box=3)
    if npts >= n:
        selected = _solve_exact_remove(adj, npts, k=n, target=target,
                                       time_limit=100.0, deadline=deadline)
        if selected is not None and len(selected) == n:
            result_points = [points_list[i] for i in selected]
            edges = _count_edges(result_points)
            if edges >= target:
                return [[a, b] for a, b in result_points]

    # Fallback: hexagonal ring + local polish.
    if time.time() < deadline - 5:
        points = _construct_hexagonal_ring(n)
        edges = _count_edges(points)
        if edges < target:
            points, edges = _local_improve(points, deadline - 2)
        if edges >= target and len(set(points)) == n:
            return [[a, b] for a, b in set(points)]

    # Guaranteed fallback: 16x2 strip.
    return [[a, b] for a in range(16) for b in range(2)]
# EVOLVE-BLOCK-END
