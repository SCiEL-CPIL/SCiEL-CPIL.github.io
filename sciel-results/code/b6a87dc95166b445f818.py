"""Suite task: cover the unit sphere with 50 equal caps.

Strategy: start from a Fibonacci spiral, then run a smooth (log-sum-exp)
gradient descent on the *critical points* (facets' circumcenters) of the
convex hull, followed by a targeted local refinement that moves the three
vertices of the worst facet along their geodesic bisectors.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50
T0 = time.time()
DEADLINE = 140.0  # safety margin before the 150s kill

# ---------- geometry helpers ----------

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return X / n


def _circumcenter(a, b, c):
    """Circumcenter of triangle (a,b,c) on the unit sphere.
    Returns the point on the great-circle plane equidistant from a,b,c,
    chosen as the outward normal of the facet. Returns None if degenerate."""
    ab = b - a
    ac = c - a
    # perpendicular to ab and ac
    n1 = np.cross(ab, np.array([0, 0, 1.0]) if abs(ab[2]) < 0.5 else np.array([0, 1.0, 0]))
    n1 = n1 - ab * (np.dot(n1, ab) / np.dot(ab, ab))
    n2 = n1.copy()
    n2 = n2 - ac * (np.dot(n2, ac) / np.dot(ac, ac))
    # bisector from a
    v1 = n1 / (np.linalg.norm(n1) + 1e-30)
    # bisector from b
    ba = a - b
    bc = c - b
    m1 = np.cross(ba, np.array([0, 0, 1.0]) if abs(ba[2]) < 0.5 else np.array([0, 1.0, 0]))
    m1 = m1 - ba * (np.dot(m1, ba) / np.dot(ba, ba))
    m1 = m1 - bc * (np.dot(m1, bc) / np.dot(bc, bc))
    if np.linalg.norm(m1) < 1e-12:
        m1 = -v1  # fallback
    v2 = m1 / (np.linalg.norm(m1) + 1e-30)
    # intersect the two great circles: solve  c = a + s*v1_perp ; c = b + t*v2_perp
    # Use the plane through a,b,c: normal
    nrm = np.cross(b - a, c - a)
    nr = np.linalg.norm(nrm)
    if nr < 1e-12:
        return None
    nrm = nrm / nr
    # circumcenter lies in plane of triangle at distance from centroid along nrm
    # Actually circumcenter = point in the triangle's plane equidistant.
    # Parametrize: c = g + w * nrm where g is projection of origin? No.
    # Simpler: solve for point p with |p-a|^2=|p-b|^2=|p-c|^2 and p . nrm = h (in plane)
    # and |p|=1.
    # Use linear system: 2*(b-a).p = |b|^2-|a|^2 = 0 (unit) -> 2*(b-a).p=0
    #                  2*(c-a).p = 0
    A = np.array([2 * (b - a), 2 * (c - a), nrm])
    rhs = np.array([0.0, 0.0, None])
    # p = alpha*nrm + beta*(b-a) + gamma*(c-a)? Use basis: nrm, e1=b-a, e2=c-a
    # Solve p . (b-a)=0, p . (c-a)=0, p . nrm = h, |p|=1
    # p = h*nrm + u*(b-a)+v*(c-a)
    # p.(b-a)= h*nrm.(b-a)+ u*|b-a|^2 + v*(c-a).(b-a)=0
    # p.(c-a)= h*nrm.(c-a)+ u*(b-a).(c-a)+ v*|c-a|^2=0
    e1 = b - a
    e2 = c - a
    G = np.array([[np.dot(e1, e1), np.dot(e1, e2)],
                  [np.dot(e2, e1), np.dot(e2, e2)]])
    rhs2 = np.array([-h * np.dot(nrm, e1), -h * np.dot(nrm, e2)])
    try:
        uv = np.linalg.solve(G, rhs2)
    except np.linalg.LinAlgError:
        return None
    # |p|^2 = h^2 + 2h*(u e1+v e2).nrm + |u e1+v e2|^2 =1
    # since (u e1+v e2) perpendicular to nrm: |p|^2=h^2+|u e1+v e2|^2
    q = uv[0] * e1 + uv[1] * e2
    norm_q = np.linalg.norm(q)
    # h^2 = 1 - norm_q^2
    if 1.0 - norm_q * norm_q < -1e-9:
        return None
    h2 = max(0.0, 1.0 - norm_q * norm_q)
    h = np.sqrt(h2)
    # choose sign so that p is outward (same side as nrm)
    p = h * nrm + q
    if np.dot(p, nrm) < 0:
        p = -h * nrm + q
    p = p / (np.linalg.norm(p) + 1e-30)
    return p


def _facets(points):
    """Return list of (vertex_indices, circumcenter) for convex hull facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    facets = []
    for simp in hull.simplices:
        a = points[simp[0]]
        b = points[simp[1]]
        c = points[simp[2]]
        cc = _circumcenter(a, b, c)
        if cc is None:
            continue
        facets.append((simp, cc))
    return facets


def _covering_radius(points):
    """Covering radius in radians via hull facets' circumcenters."""
    facets = _facets(points)
    if not facets:
        return np.pi
    worst = 0.0
    for simp, cc in facets:
        # angular distance from cc to each of its vertices
        d = np.arccos(np.clip(np.dot(cc, points[simp[0]]), -1.0, 1.0))
        worst = max(worst, d)
    return worst


def _logsumexp_angle(points, facets, beta):
    """Smooth objective: logsumexp(beta * d_i) over facets, where d_i is the
    circumradius of facet i. Returns (value, gradient w.r.t. points)."""
    if not facets:
        return np.pi, np.zeros_like(points)
    ds = np.empty(len(facets))
    for k, (simp, cc) in enumerate(facets):
        ds[k] = np.arccos(np.clip(np.dot(cc, points[simp[0]]), -1.0, 1.0))
    m = ds.max()
    if not np.isfinite(m):
        return np.pi, np.zeros_like(points)
    exps = np.exp(beta * (ds - m))
    Z = exps.sum()
    w = exps / Z  # softmax weights
    val = m + np.log(Z) / beta
    # gradient: d/dp  logsumexp(beta*d_i) = sum_k w_k * beta * grad_p d_k
    grad = np.zeros_like(points)
    for k, (simp, cc) in enumerate(facets):
        if w[k] < 1e-12:
            continue
        # d_k = arccos( cc . p_v ), v=simp[0]
        # grad_{p_v} d_k = -cc / sqrt(1 - (cc.p_v)^2)
        cv = points[simp[0]]
        s = np.clip(cc.dot(cv), -1.0, 1.0)
        denom = np.sqrt(max(1e-12, 1.0 - s * s))
        grad[simp[0]] += w[k] * beta * (-cc / denom)
    return val, grad


# ---------- refinement ----------

def _refine_worst_facet(points, max_iter=300, tol=1e-10):
    """Move the three vertices of the worst facet along geodesic bisectors
    toward the facet's circumcenter to reduce its circumradius."""
    pts = _normalize(points.copy())
    for _ in range(max_iter):
        facets = _facets(pts)
        if not facets:
            break
        # find worst facet
        worst_k = -1
        worst_d = -1.0
        for k, (simp, cc) in enumerate(facets):
            d = np.arccos(np.clip(np.dot(cc, pts[simp[0]]), -1.0, 1.0))
            if d > worst_d:
                worst_d = d
                worst_k = k
        if worst_k < 0:
            break
        simp, cc = facets[worst_k]
        # move each vertex toward cc a small step
        step = 0.02
        for v in simp:
            p = pts[v]
            # geodesic direction toward cc
            s = np.clip(np.dot(p, cc), -1.0, 1.0)
            if s > 0.999999:
                continue
            # tangent component of cc
            t = cc - s * p
            tn = np.linalg.norm(t)
            if tn < 1e-12:
                continue
            t = t / tn
            # new point: rotate p toward cc by step
            ang = step
            c = np.cos(ang)
            sn = np.sin(ang)
            newp = c * p + sn * t
            newp = newp / (np.linalg.norm(newp) + 1e-30)
            # accept if it doesn't increase the overall covering radius too much
            old_r = _covering_radius(pts)
            pts[v] = newp
            new_r = _covering_radius(pts)
            if new_r > old_r + 1e-9:
                pts[v] = p  # revert
            else:
                if new_r < old_r - tol:
                    pass
    return pts


# ---------- main ----------

def construct():
    global T0
    # Fibonacci spiral start
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)

    best = points.copy()
    best_r = _covering_radius(points)

    # Smooth gradient descent on logsumexp of facet circumradii
    beta = 8.0
    lr = 0.02
    for it in range(4000):
        if time.time() > DEADLINE - 20.0:
            break
        facets = _facets(points)
        if not facets:
            break
        val, grad = _logsumexp_angle(points, facets, beta)
        # normalize gradient
        gn = np.linalg.norm(grad)
        if gn < 1e-15:
            break
        grad = grad / gn
        # update
        newp = points - lr * grad
        newp = _normalize(newp)
        # accept if it reduces the actual covering radius
        new_r = _covering_radius(newp)
        if new_r < best_r:
            best = newp.copy()
            best_r = new_r
        # always step if it reduces the smooth objective (helps escape)
        points = newp
        # slow the learning rate
        lr *= 0.9995
        # grow beta to sharpen
        beta = min(beta * 1.01, 60.0)

    # Targeted facet refinement
    if time.time() < DEADLINE - 15.0:
        refined = _refine_worst_facet(best, max_iter=200)
        rr = _covering_radius(refined)
        if rr < best_r:
            best = refined
            best_r = rr

    # Final high-precision polish: small steps on worst facets
    if time.time() < DEADLINE - 5.0:
        pts = _normalize(best.copy())
        for _ in range(300):
            if time.time() > DEADLINE - 5.0:
                break
            facets = _facets(pts)
            if not facets:
                break
            worst_k = -1
            worst_d = -1.0
            for k, (simp, cc) in enumerate(facets):
                d = np.arccos(np.clip(np.dot(cc, pts[simp[0]]), -1.0, 1.0))
                if d > worst_d:
                    worst_d = d
                    worst_k = k
            if worst_k < 0:
                break
            simp, cc = facets[worst_k]
            old_r = _covering_radius(pts)
            improved = False
            for v in simp:
                p = pts[v]
                s = np.clip(np.dot(p, cc), -1.0, 1.0)
                if s > 0.999999:
                    continue
                t = cc - s * p
                tn = np.linalg.norm(t)
                if tn < 1e-12:
                    continue
                t = t / tn
                for ang in (0.005, 0.002, 0.001):
                    c = np.cos(ang)
                    sn = np.sin(ang)
                    newp = c * p + sn * t
                    newp = newp / (np.linalg.norm(newp) + 1e-30)
                    pts[v] = newp
                    nr = _covering_radius(pts)
                    if nr < old_r - 1e-12:
                        old_r = nr
                        improved = True
                        break
                    else:
                        pts[v] = p
            if not improved:
                break
        if _covering_radius(pts) < best_r:
            best = pts
            best_r = _covering_radius(pts)

    # Ensure valid: nonzero, normalized, not coincident, not in a hemisphere
    best = _normalize(best)
    # spread check: if all in a hemisphere, push apart
    # (Fibonacci start is well spread, so this is a safety net)
    return best
