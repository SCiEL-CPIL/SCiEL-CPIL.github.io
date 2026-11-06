"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _edge_count(points):
    """Count unit-distance edges among the given set of lattice points.
    Distance 1 on the triangular lattice (axial coords a,b) corresponds to
    the six neighbours: (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1).
    """
    s = set(tuple(p) for p in points)
    count = 0
    for (a, b) in s:
        for da, db in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)):
            if (a + da, b + db) in s:
                count += 1
    return count // 2

def construct():
    deadline = time.time() + 110.0  # safety margin under 120s limit
    N = 32
    OPT = 76  # Harborth's bound for n=32

    # Known optimal: 16x2 strip
    base = [[a, b] for a in range(16) for b in range(2)]
    best_pts = base
    best_edges = _edge_count(best_pts)

    if best_edges >= OPT:
        return best_pts

    # Spiral construction (inspiration 1): generate points by increasing
    # graph distance from origin, then by angle, and take the first N.
    import math
    R = 12
    candidates = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            d = (abs(a) + abs(b) + abs(a + b)) // 2
            x = a + 0.5 * b
            y = (math.sqrt(3) / 2) * b
            angle = math.atan2(y, x)
            candidates.append((d, angle, a, b))
    candidates.sort(key=lambda x: (x[0], x[1]))
    spiral_pts = [[a, b] for d, angle, a, b in candidates[:N]]
    spiral_edges = _edge_count(spiral_pts)
    if spiral_edges > best_edges:
        best_pts = spiral_pts
        best_edges = spiral_edges
    if best_edges >= OPT:
        return best_pts

    # Local search: move one point to an adjacent empty cell if it improves
    # the edge count (or ties and reduces max distance from centroid).
    def _centroid(pts):
        sa = sum(p[0] for p in pts) / len(pts)
        sb = sum(p[1] for p in pts) / len(pts)
        return sa, sb

    def _spread(pts):
        sa, sb = _centroid(pts)
        return sum((p[0] - sa)**2 + (p[1] - sb)**2 for p in pts)

    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    random.seed(42)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        pts = [list(p) for p in best_pts]
        pt_set = set(tuple(p) for p in pts)
        for i in range(N):
            if time.time() > deadline:
                break
            a, b = pts[i]
            for da, db in directions:
                na, nb = a + da, b + db
                if (na, nb) in pt_set:
                    continue
                if abs(na) > 64 or abs(nb) > 64:
                    continue
                # Tentative move
                pts[i] = [na, nb]
                pt_set.discard((a, b))
                pt_set.add((na, nb))
                new_edges = _edge_count(pts)
                if new_edges > best_edges:
                    best_pts = [list(p) for p in pts]
                    best_edges = new_edges
                    improved = True
                    if best_edges >= OPT:
                        return best_pts
                else:
                    # Revert
                    pts[i] = [a, b]
                    pt_set.discard((na, nb))
                    pt_set.add((a, b))

    return best_pts
# EVOLVE-BLOCK-END
