"""Fresh design: Delaunay face-normal slerp equalization + point refinement for sphere covering."""
import time
import numpy as np

N = 50

# ---------- helpers ----------

def _normalize(pts):
    pts = np.asarray(pts, dtype=float)
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def _spherical_delaunay(pts):
    """Return list of triangular facets (indices) on the sphere via 3D Delaunay.
    Uses scipy.spatial.Delaunay on the 3D points; facets whose outward normal
    points away from the centroid (i.e. toward the sphere) are the spherical ones."""
    from scipy.spatial import Delaunay
    tri = Delaunay(pts)
    faces = []
    for simplex in tri.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        # centroid of triangle
        cen = (a + b + c) / 3.0
        # outward if n dot cen > 0
        if np.dot(n, cen) > 0:
            faces.append(simplex)
    return faces


def _facet_circumradius(pts, face):
    """Angular circumradius of a spherical triangle = max angular distance from
    the (spherical) circumcenter to the 3 vertices. We approximate the circumcenter
    as the normalized centroid of the 3 vertices (a good proxy for small facets)
    and take the max angular distance to the vertices."""
    a, b, c = pts[face[0]], pts[face[1]], pts[face[2]]
    cen = a + b + c
    nc = np.linalg.norm(cen)
    if nc < 1e-12:
        return np.pi
    cen = cen / nc
    d = []
    for v in (a, b, c):
        cosang = np.clip(np.dot(cen, v), -1.0, 1.0)
        d.append(np.arccos(cosang))
    return float(np.max(d))


def _covering_radius_degrees(pts):
    """Covering radius in degrees: max over facets of its circumradius (empty-cap check).
    We evaluate by sampling many points and also by facet circumradii; take the max."""
    pts = _normalize(pts)
    faces = _spherical_delaunay(pts)
    if not faces:
        # fallback: brute sample
        best = 0.0
        for _ in range(2000):
            y = np.random.randn(3)
            y /= np.linalg.norm(y)
            d = np.arccos(np.clip(pts @ y, -1, 1))
            best = max(best, np.min(d))
        return np.degrees(best)
    cr = 0.0
    for f in faces:
        cr = max(cr, _facet_circumradius(pts, f))
    return np.degrees(cr)


def _slerp(a, b, t):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    omega = np.arccos(np.clip(np.dot(a, b), -1, 1))
    if omega < 1e-12:
        return a
    return (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)


def _equalize_facet_normals(pts, iters=200, lr=0.05):
    """Dual-polytope view: each vertex of the point set is the pole (normal) of a
    Voronoi facet of the Delaunay triangulation. We move each pole via slerp toward
    the average of the circumcenters of its incident facets, weighted to reduce the
    largest circumradii (push big facets' poles outward to shrink them)."""
    pts = _normalize(pts).copy()
    for it in range(iters):
        faces = _spherical_delaunay(pts)
        if not faces:
            break
        # circumradius per facet
        crs = [_facet_circumradius(pts, f) for f in faces]
        mean_cr = np.mean(crs)
        # for each vertex, accumulate a target direction
        target = np.zeros_like(pts)
        weights = np.zeros(N)
        for idx, f in enumerate(faces):
            a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
            cen = a + b + c
            nc = np.linalg.norm(cen)
            if nc < 1e-12:
                continue
            cen = cen / nc
            # weight: bigger facet -> stronger pull toward its circumcenter
            w = 1.0 + 3.0 * max(0.0, crs[idx] - mean_cr)
            for v in f:
                target[v] += w * cen
                weights[v] += w
        moved = False
        for v in range(N):
            if weights[v] < 1e-12:
                continue
            tgt = target[v] / weights[v]
            nt = np.linalg.norm(tgt)
            if nt < 1e-12:
                continue
            tgt = tgt / nt
            newp = _slerp(pts[v], tgt, lr)
            if np.linalg.norm(newp - pts[v]) > 1e-10:
                pts[v] = newp
                moved = True
        if not moved:
            break
    return _normalize(pts)


def _refine_points(pts, iters=300, lr=0.02):
    """Point-movement refinement: move each point away from the center of its
    largest incident facet (reduce local covering radius) via gradient ascent on
    the minimum angular distance to the point, with a repulsion term."""
    pts = _normalize(pts).copy()
    for it in range(iters):
        faces = _spherical_delaunay(pts)
        if not faces:
            break
        grad = np.zeros_like(pts)
        for f in faces:
            a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
            cen = a + b + c
            nc = np.linalg.norm(cen)
            if nc < 1e-12:
                continue
            cen = cen / nc
            cr = _facet_circumradius(pts, f)
            # push each vertex away from the facet center, scaled by circumradius
            push = cr * cr
            for v in f:
                # direction from facet center to vertex (on tangent plane)
                d = pts[v] - np.dot(pts[v], cen) * cen
                dn = np.linalg.norm(d)
                if dn > 1e-12:
                    grad[v] += push * (d / dn)
        # normalize gradient per point and apply
        for v in range(N):
            gn = np.linalg.norm(grad[v])
            if gn < 1e-12:
                continue
            newp = _slerp(pts[v], pts[v] + (lr / gn) * grad[v], 1.0)
            pts[v] = newp
        pts = _normalize(pts)
    return _normalize(pts)


def _fibonacci_seed(jitter=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        pts = pts + jitter * np.random.randn(N, 3)
    return _normalize(pts)


def _voronoi_seed(jitter=0.0):
    # start from fibonacci then run a few equalization passes to get a Voronoi-like
    # uniform distribution
    pts = _fibonacci_seed(jitter)
    pts = _equalize_facet_normals(pts, iters=50, lr=0.03)
    return pts


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin under 150s
    rng = np.random.default_rng(12345)

    # candidate pool of seeds
    seeds = []
    seeds.append(_fibonacci_seed(0.0))
    seeds.append(_voronoi_seed(0.0))
    # jittered fibonacci restarts
    for j in (0.02, 0.05, 0.10, 0.20):
        seeds.append(_fibonacci_seed(j))
    # a couple of random-ish starts projected to sphere
    for _ in range(2):
        seeds.append(_normalize(rng.standard_normal((N, 3))))

    best = seeds[0]
    best_score = _covering_radius_degrees(best)

    for seed in seeds:
        if time.time() > DEADLINE:
            break
        cand = seed.copy()
        # Stage 1: dual equalization (face-normal slerp)
        cand = _equalize_facet_normals(cand, iters=120, lr=0.05)
        s1 = _covering_radius_degrees(cand)
        # Stage 2: point refinement
        cand2 = _refine_points(cand, iters=200, lr=0.02)
        s2 = _covering_radius_degrees(cand2)
        # Stage 3: one more equalization pass on the refined config
        cand3 = _equalize_facet_normals(cand2, iters=80, lr=0.03)
        s3 = _covering_radius_degrees(cand3)

        for c, s in ((cand, s1), (cand2, s2), (cand3, s3)):
            if s < best_score:
                best = c
                best_score = s

    return _normalize(best)
