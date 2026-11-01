"""Optimized covering of the sphere with 50 points using spherical Voronoi + gradient descent."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

N = 50

# EVOLVE-BLOCK-START
def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius: the maximum angular distance from any point on the
    sphere to the nearest of the N points. This equals the maximum circumradius of
    the spherical Voronoi cells, which we approximate via the convex hull in R^3.
    """
    pts = _normalize(pts)
    # Build convex hull of the points on the sphere
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case

    # For each facet, the covering radius contribution is the angular distance from
    # the facet's circumcenter (pushed out to the sphere) to the nearest vertex.
    # The circumcenter of a planar triangle in R^3, normalized, gives the point on
    # the sphere farthest from that facet's vertices (within the cell).
    # Actually, the circumradius of the spherical Voronoi cell is the angular distance
    # from the cell center to its farthest vertex. We compute the circumcenter of each
    # hull facet and measure the max angle to the facet's vertices.

    facets = hull.simplices
    max_ang = 0.0
    for tri in facets:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        # Compute circumcenter of triangle abc in R^3
        # Solve for point p equidistant from a, b, c
        # p = a + alpha*(b-a) + beta*(c-a)
        # |p-a|^2 = |p-b|^2 => 2 a.(b-a) ... let's use the formula:
        # Circumcenter = (|a|^2(b×c) + |b|^2(c×a) + |c|^2(a×b)) / (2 a.(b×c))
        # But a, b, c are unit vectors, so |a|^2=|b|^2=|c|^2=1
        bc = np.cross(b, c)
        ca = np.cross(c, a)
        ab = np.cross(a, b)
        denom = 2.0 * np.dot(a, bc)
        if abs(denom) < 1e-12:
            # Degenerate: use centroid
            cc = (a + b + c) / 3.0
        else:
            cc = (bc + ca + ab) / denom
        cc_norm = np.linalg.norm(cc)
        if cc_norm < 1e-12:
            continue
        cc = cc / cc_norm
        # Angular distance from cc to each vertex
        for v in (a, b, c):
            dot = np.clip(np.dot(cc, v), -1.0, 1.0)
            ang = np.arccos(dot)
            if ang > max_ang:
                max_ang = ang
    return max_ang


def _objective_and_grad(pts):
    """
    Objective: covering radius (max circumradius of Voronoi cells).
    Gradient: we approximate the gradient by considering the farthest facet's
    circumcenter. The gradient pushes the vertices of the farthest facet
    toward the circumcenter to reduce the cell size.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.zeros_like(pts)

    facets = hull.simplices
    # Find the facet with the maximum circumradius
    best_ang = 0.0
    best_facet = None
    best_cc = None
    for tri in facets:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        bc = np.cross(b, c)
        ca = np.cross(c, a)
        ab = np.cross(a, b)
        denom = 2.0 * np.dot(a, bc)
        if abs(denom) < 1e-12:
            continue
        cc = (bc + ca + ab) / denom
        cc_norm = np.linalg.norm(cc)
        if cc_norm < 1e-12:
            continue
        cc = cc / cc_norm
        # Max angle from cc to vertices
        for v in (a, b, c):
            dot = np.clip(np.dot(cc, v), -1.0, 1.0)
            ang = np.arccos(dot)
            if ang > best_ang:
                best_ang = ang
                best_facet = tri
                best_cc = cc

    if best_facet is None:
        return best_ang, np.zeros_like(pts)

    # Gradient: push the three vertices of the worst facet toward the circumcenter
    # on the sphere. The tangent direction from vertex v toward cc is:
    # t = cc - (cc.v) v, normalized
    grad = np.zeros_like(pts)
    for idx in best_facet:
        v = pts[idx]
        proj = np.dot(cc := best_cc, v) * v
        tangent = cc - proj
        tn = np.linalg.norm(tangent)
        if tn > 1e-12:
            grad[idx] += tangent / tn
    return best_ang, grad


def _random_initialization(seed=None):
    rng = np.random.default_rng(seed)
    # Random points on sphere
    pts = rng.normal(size=(N, 3))
    return _normalize(pts)


def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    best_pts = None
    best_score = np.pi

    # Start with Fibonacci spiral as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

    # Gradient descent with adaptive step size
    lr = 0.05
    for it in range(2000):
        if time.time() > deadline:
            break
        score, grad = _objective_and_grad(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
        # Gradient step on the sphere: move in tangent direction
        # Project gradient to tangent space
        tangent_grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        pts = pts + lr * tangent_grad
        pts = _normalize(pts)
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            # Reduce step size slowly
            lr *= 0.95

    # If time remains, try a few random restarts with local refinement
    while time.time() < deadline - 5:
        seed = np.random.randint(0, 2**31)
        pts = _random_initialization(seed)
        lr = 0.03
        for it in range(300):
            if time.time() > deadline - 5:
                break
            score, grad = _objective_and_grad(pts)
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
            tangent_grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            pts = pts + lr * tangent_grad
            pts = _normalize(pts)
            if it % 40 == 0 and it > 0:
                lr *= 0.9

    if best_pts is None:
        best_pts = pts
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
