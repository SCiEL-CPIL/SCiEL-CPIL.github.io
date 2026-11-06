"""Rewritten program: simulated-annealing covering-radius minimizer for N=50 points on S^2.

Approach family: stochastic local search (Metropolis annealing) directly minimizing the
covering radius, estimated on a dense spherical probe grid.  This is a different algorithm
family from the seed's deterministic Fibonacci-spiral construction.
"""
import numpy as np

N = 50


# ---------------------------------------------------------------------------
# Probe grid: a dense set of points on the sphere used to *estimate* the
# covering radius.  We build it from an icosphere subdivision, which gives a
# nearly uniform distribution.  The true covering radius is the max circumradius
# of the convex-hull facets; a fine probe grid is a tight, cheap surrogate.
# ---------------------------------------------------------------------------
def _icosphere(subdiv: int = 4) -> np.ndarray:
    """Return points on the unit sphere from an icosahedron subdivided `subdiv` times."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t, 0], [ 1,  t, 0], [-1, -t, 0], [ 1, -t, 0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t, 0, -1], [ t, 0,  1], [-t, 0, -1], [-t, 0,  1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    # cache midpoints
    mid = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            v = (verts[a] + verts[b]) / 2.0
            v /= np.linalg.norm(v)
            mid[key] = len(verts)
            verts = np.vstack([verts, v])
        return mid[key]

    for _ in range(subdiv):
        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return verts


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _normalize(pts: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return pts / n


def _covering_radius(pts: np.ndarray, probes: np.ndarray) -> float:
    """Max over probes of the angular distance (radians) to the nearest point."""
    # pts: (N,3), probes: (M,3).  Compute nearest distance via dot products.
    dots = probes @ pts.T            # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    nearest_cos = dots.max(axis=1)   # (M,)
    nearest_cos = np.clip(nearest_cos, -1.0, 1.0)
    ang = np.arccos(nearest_cos)
    return float(ang.max())


def _covering_radius_grad_argmax(pts: np.ndarray, probes: np.ndarray):
    """Return (radius, probe_index) of the worst probe."""
    dots = probes @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    nearest_cos = dots.max(axis=1)
    nearest_cos = np.clip(nearest_cos, -1.0, 1.0)
    ang = np.arccos(nearest_cos)
    idx = int(ang.argmax())
    return float(ang[idx]), idx


# ---------------------------------------------------------------------------
# Deterministic polish: push the point that owns the worst probe toward the
# circumcenter of the facet that contains that probe.  Repeat a few times.
# ---------------------------------------------------------------------------
def _polish(pts: np.ndarray, probes: np.ndarray, iters: int = 200) -> np.ndarray:
    pts = _normalize(pts.copy())
    for _ in range(iters):
        _, pidx = _covering_radius_grad_argmax(pts, probes)
        probe = probes[pidx]
        # The three nearest points define the facet containing `probe`.
        dots = probe @ pts.T
        order = np.argsort(-dots)
        tri = order[:3]
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        # Circumcenter of the spherical triangle (a,b,c) = normalized
        # a+b+c works well for small triangles (the pole of the great-circle
        # plane through them).  Use the plane normal as the facet circumcenter.
        nrm = np.cross(b - a, c - a)
        nrm_n = np.linalg.norm(nrm)
        if nrm_n < 1e-12:
            continue
        # The facet's circumcenter on the sphere is the normalized normal
        # pointing toward the probe (same hemisphere as probe).
        cc = nrm / nrm_n
        if np.dot(cc, probe) < 0:
            cc = -cc
        # Move each of the three points a small step toward cc.
        step = 0.15
        for k in tri:
            pts[k] = pts[k] + step * (cc - pts[k])
            pts[k] /= np.linalg.norm(pts[k])
    return pts


# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------
def _anneal(pts0: np.ndarray, probes: np.ndarray, rng: np.random.Generator,
            t0: float = 0.15, t1: float = 1e-4, n_iter: int = 60000) -> np.ndarray:
    pts = _normalize(pts0.copy())
    cur = _covering_radius(pts, probes)
    best = cur
    best_pts = pts.copy()

    n = pts.shape[0]
    # geometric cooling
    ratio = (t1 / t0) ** (1.0 / max(1, n_iter))
    T = t0
    for it in range(n_iter):
        T = max(t1, T * ratio)
        # pick a random point to perturb
        k = rng.integers(n)
        # random direction on the sphere
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        # step size scales with temperature
        step = 0.05 * (T / t0 + 0.05)
        new_pts = pts.copy()
        new_pts[k] = pts[k] + step * d
        new_pts[k] /= np.linalg.norm(new_pts[k])

        new_cost = _covering_radius(new_pts, probes)
        delta = new_cost - cur
        if delta < 0 or rng.random() < np.exp(-delta / T):
            pts = new_pts
            cur = new_cost
            if cur < best:
                best = cur
                best_pts = pts.copy()
    return best_pts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def construct():
    rng = np.random.default_rng(12345)

    # Dense probe grid for the surrogate covering radius.
    probes = _icosphere(subdiv=4)   # ~2562 points, good resolution
    probes = _normalize(probes)

    # --- Initial configuration: several diverse starts, keep the best after a
    # short anneal, then do a longer anneal from the best.
    starts = []

    # Start 1: Fibonacci spiral (well spread).
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    starts.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

    # Start 2: random points.
    starts.append(_normalize(rng.normal(size=(N, 3))))

    # Start 3: antipodal pairs + jitter (guarantees good spread).
    base = _normalize(rng.normal(size=(25, 3)))
    ant = np.vstack([base, -base])
    ant = _normalize(ant + 0.02 * rng.normal(size=(N, 3)))
    starts.append(ant)

    # Quick anneal each start, keep the best.
    best_pts = None
    best_val = np.inf
    for s in starts:
        cand = _anneal(s, probes, rng, t0=0.12, t1=1e-3, n_iter=25000)
        v = _covering_radius(cand, probes)
        if v < best_val:
            best_val = v
            best_pts = cand

    # Longer anneal from the best.
    best_pts = _anneal(best_pts, probes, rng, t0=0.10, t1=1e-4, n_iter=80000)

    # Deterministic polish to tighten the worst facet.
    best_pts = _polish(best_pts, probes, iters=300)

    # Final safety: ensure unit length and non-zero.
    best_pts = _normalize(best_pts)
    return best_pts
