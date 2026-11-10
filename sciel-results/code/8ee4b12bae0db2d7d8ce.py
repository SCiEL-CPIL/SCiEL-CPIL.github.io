"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150 s)


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def _covering_radius(pts):
    """Covering radius (radians) = largest circumradius of convex-hull facets,
    only counting facets whose circumcenter lies on the unit sphere (empty cap)."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # invalid -> huge
    best = 0.0
    for f in hull.simplices:
        tri = pts[f]
        # circumcenter of the triangle in its plane
        a, b, c = tri[0], tri[1], tri[2]
        d = 2.0 * (a[0] * (b[1] * c[2] - b[2] * c[1])
                   + a[1] * (b[2] * c[0] - b[0] * c[2])
                   + a[2] * (b[0] * c[1] - b[1] * c[0]))
        if abs(d) < 1e-12:
            continue
        ux = ((np.dot(a, a)) * (b[1] * c[2] - b[2] * c[1])
              - (np.dot(b, b)) * (a[1] * c[2] - a[2] * c[1])
              + (np.dot(c, c)) * (a[1] * b[2] - a[2] * b[1])) / d
        uy = ((np.dot(a, a)) * (b[2] * c[0] - b[0] * c[2])
              - (np.dot(b, b)) * (a[2] * c[0] - a[0] * c[2])
              + (np.dot(c, c)) * (a[2] * b[0] - a[0] * b[2])) / d
        uz = ((np.dot(a, a)) * (b[0] * c[1] - b[1] * c[0])
              - (np.dot(b, b)) * (a[0] * c[1] - a[1] * c[0])
              + (np.dot(c, c)) * (a[0] * b[1] - a[1] * b[0])) / d
        cc = np.array([ux, uy, uz])
        # circumcenter must lie on the unit sphere (empty cap)
        if abs(np.dot(cc, cc) - 1.0) > 1e-4:
            continue
        # circumradius = angle from circumcenter to a vertex
        cosv = np.clip(np.dot(cc, a), -1.0, 1.0)
        r = np.arccos(cosv)
        if r > best:
            best = r
    if best == 0.0:
        # fallback: nearest-neighbor angular distance
        G = np.clip(pts @ pts.T, -1.0, 1.0)
        np.fill_diagonal(G, -1.0)
        best = np.arccos(G.min())
    return best


def _build_truncated_icosahedron():
    """Build a truncated icosahedron (football) with 12 deg-5 and 38 deg-6 vertices.
    Returns 50 unit vectors."""
    t = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    # Base icosahedron vertices (12)
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico = _normalize(ico)

    # Truncated icosahedron: 12 pentagonal faces (from ico vertices) + 20 hexagonal faces
    # Standard construction: vertices are midpoints of icosahedron edges, scaled.
    # A truncated icosahedron has 60 edges -> 60 vertices? No: 60 vertices total.
    # We need only 50, so instead use the known Hardin-Sloane-Smith 50-point layout:
    # It is a truncated icosahedron with 10 vertices removed (or a related polytope).
    #
    # Simpler robust approach: use the 60-vertex truncated icosahedron, then
    # select 50 by removing 10 well-spread vertices, then re-optimize.
    #
    # Truncated icosahedron vertices: take icosahedron, truncate each vertex.
    # New vertices sit at 1/3 along each original edge from each endpoint.
    # 60 edges -> 60 vertices.
    #
    # Find all edges of the icosahedron (pairs at min distance).
    D = 1.0 - ico @ ico.T
    min_d = D.min()
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_d) < 1e-8:
                edges.append((i, j))
    verts = []
    for (i, j) in edges:
        # two new vertices per edge (one near i, one near j)
        verts.append(ico[i] * (2.0 / 3.0) + ico[j] * (1.0 / 3.0))
        verts.append(ico[i] * (1.0 / 3.0) + ico[j] * (2.0 / 3.0))
    verts = np.array(verts)
    verts = _normalize(verts)
    # 60 vertices. Remove 10 to get 50: remove one from each pentagonal face ring.
    # The 12 icosahedron vertices correspond to 12 pentagonal faces; each pentagon
    # has 5 vertices. Removing a full pentagon (5) from two faces = 10 vertices.
    # Identify vertices belonging to the pentagon around ico vertex k: those edges
    # incident to k.
    # Build vertex -> set of incident ico-vertex indices.
    from collections import defaultdict
    inc = defaultdict(list)
    for (i, j) in edges:
        for (a, b) in [(i, j), (j, i)]:
            pass
    # Map each truncated vertex to its two endpoint ico indices.
    vmap = []
    for (i, j) in edges:
        vmap.append((i, j))
        vmap.append((i, j))
    # For each ico vertex k, the pentagon = truncated vertices on edges incident to k.
    inc_ico = defaultdict(list)
    for idx, (i, j) in enumerate(vmap):
        inc_ico[i].append(idx)
        inc_ico[j].append(idx)
    # Choose two ico vertices to remove their pentagons: pick two that are far apart
    # (antipodal-ish) to keep the rest symmetric.
    # Find antipodal pair in icosahedron.
    G = ico @ ico.T
    np.fill_diagonal(G, -2.0)
    pair_i, pair_j = np.unravel_index(np.argmin(G), G.shape)
    # Remove pentagons around pair_i and pair_j
    remove = set(inc_ico[pair_i]) | set(inc_ico[pair_j])
    keep = [k for k in range(len(verts)) if k not in remove]
    out = verts[keep]
    # Should be 50.
    if len(out) != 50:
        # fallback: keep first 50
        out = out[:50]
    return _normalize(out)


def _local_optimize(pts, time_budget):
    """Greedy coordinate-wise descent on the covering radius (radians)."""
    pts = _normalize(pts).copy()
    best = _covering_radius(pts)
    start = time.time()
    it = 0
    while time.time() - start < time_budget:
        it += 1
        improved = False
        for i in range(N):
            for axis in range(3):
                for delta in (1e-3, 3e-3, 1e-4, 1e-5):
                    for sgn in (1.0, -1.0):
                        cand = pts.copy()
                        cand[i, axis] += sgn * delta
                        cand[i] = cand[i] / (np.linalg.norm(cand[i]) + 1e-15)
                        r = _covering_radius(cand)
                        if r < best - 1e-10:
                            pts = cand
                            best = r
                            improved = True
        if not improved:
            # shrink step
            break
    return pts, best


def construct():
    t0 = time.time()
    # Start from the truncated-icosahedron-derived 50-point layout.
    try:
        pts = _build_truncated_icosahedron()
    except Exception:
        pts = None
    if pts is None or len(pts) != 50 or not np.all(np.isfinite(pts)):
        # Fallback: Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_pts = _normalize(pts)
    best_r = _covering_radius(best_pts)

    # Local optimization with a time budget, leaving margin before deadline.
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5.0:
        try:
            opt_pts, opt_r = _local_optimize(best_pts, max(0.0, remaining - 5.0))
            if opt_r < best_r:
                best_pts, best_r = opt_pts, opt_r
        except Exception:
            pass

    # Final safety: ensure unit norm and finite.
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return best_pts
# EVOLVE-BLOCK-END
