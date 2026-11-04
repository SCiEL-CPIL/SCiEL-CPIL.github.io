import numpy as np
from scipy.optimize import minimize

N = 12
SQ3 = np.sqrt(3.0)


def _hex_vertices(cx, cy, ang):
    """6 vertices of a unit hexagon centered at (cx,cy) rotated by ang (radians)."""
    base = np.array([[1.0, 0.0], [0.5, SQ3 / 2.0], [-0.5, SQ3 / 2.0],
                     [-1.0, 0.0], [-0.5, -SQ3 / 2.0], [0.5, -SQ3 / 2.0]])
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    v = (base @ R.T) + np.array([cx, cy])
    return v


def _outer_halfwidths(side, orient):
    """Return 6 outward half-widths (normals . vertex) of outer hexagon (centered at origin)."""
    a = orient + np.arange(6) * (np.pi / 3.0)
    return side * np.cos(np.pi / 6.0) * np.ones(6), a


def _max_projection(pos, angles):
    """Given inner positions/angles, return (max_over_vertices of max_over_normals n.v, best_orient).
    Actually we need side = max_v max_n n.v for a given orient. We search orient."""
    # precompute all vertices of all hexagons
    verts = np.vstack([_hex_vertices(pos[i][0], pos[i][1], angles[i]) for i in range(N)])
    return verts


def _required_side(verts, orient):
    a = orient + np.arange(6) * (np.pi / 3.0)
    cos30 = np.cos(np.pi / 6.0)
    # normals: directions of outer vertices
    normals = np.stack([np.cos(a), np.sin(a)], axis=1)
    # For each vertex v and each normal n, constraint v.n >= -cos30
    # side must satisfy side*cos30 >= v.n for all. So side = max(v.n)/cos30
    dots = verts @ normals.T  # (V, 6)
    m = dots.max(axis=0)      # (6,)
    return (m.max() / cos30)


def _objective(theta):
    pos = theta[:2 * N].reshape(N, 2)
    angles = theta[2 * N:2 * N + N]
    orient = theta[2 * N + N]
    verts = _max_projection(pos, angles)
    side = _required_side(verts, orient)
    # penalty for overlaps (approx via distance between centers < 2*sqrt3 means touching at closest)
    # minimal center distance for non-overlap of two unit hexes (any rotation) is 2*sqrt3/2? No.
    # For two unit hexes with optimal relative rotation, min center distance = sqrt(3) (flat-to-flat touching? no).
    # Actually two unit hexes can be as close as sqrt(3) apart if flat edges touch (distance between parallel flats = sqrt(3)).
    # So min center distance = sqrt(3) when an edge touches another edge flat-to-flat.
    # To be safe, penalize d < sqrt(3).
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    iu = np.triu_indices(N, k=1)
    dd = d[iu]
    pen = np.maximum(0.0, SQ3 - dd)
    pen2 = np.sum(pen ** 2)
    return side + 10.0 * pen2


def _lattice_init():
    """Return (pos, angles) for a compact 12-hex cluster in hexagonal lattice."""
    # Hex lattice coordinates: use axial. Positions:
    # ring 0: (0,0)
    # ring 1: 6 neighbors at distance sqrt(3)
    # ring 2: we take 5 more from ring 2 to total 12
    pts = [(0.0, 0.0)]
    for k in range(6):
        ang = k * (np.pi / 3.0)
        pts.append((SQ3 * np.cos(ang), SQ3 * np.sin(ang)))
    # ring 2 positions: distance 2*sqrt(3) at angles k*60, plus the "corner" directions at 30+60k with distance?
    # Actually ring 2 has 12 positions. Take 5 evenly spaced ones.
    for k in range(5):
        ang = k * (2 * np.pi / 5.0)
        # ring 2 distance in this lattice is 2*sqrt(3) along axes, but off-axis it's sqrt(3)*sqrt(3)=3? 
        # Use the standard: ring 2 points are at positions like (2*sqrt(3),0), (sqrt(3), sqrt(3)*sqrt(3)/1)... 
        # Simpler: use the 6 axis points of ring 2 plus pick.
        pass
    # Just place 5 more around at radius 2*sqrt(3) at 5 angles
    for k in range(5):
        ang = k * (2 * np.pi / 5.0) + np.pi / 5.0
        pts.append((2 * SQ3 * np.cos(ang), 2 * SQ3 * np.sin(ang)))
    pos = np.array(pts[:N])
    angles = np.zeros(N)
    return pos, angles


def _spectral_init():
    """Build adjacency on 12 nodes in a ring + center + extras; get positions from Laplacian eigenvectors."""
    n = N
    W = np.zeros((n, n))
    # ring of 12
    for i in range(n):
        W[i, (i + 1) % n] = 1.0
        W[(i + 1) % n, i] = 1.0
    # also connect every other to make it more 2D
    for i in range(n):
        W[i, (i + 2) % n] = 0.5
        W[(i + 2) % n, i] = 0.5
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    # eigendecomposition
    w, V = np.linalg.eigh(L)
    # take the two smallest nonzero eigenvectors
    # sort
    idx = np.argsort(w)
    e1 = V[:, idx[1]]
    e2 = V[:, idx[2]]
    pos = np.stack([e1, e2], axis=1)
    # scale to reasonable size
    pos = pos / (np.linalg.norm(pos) + 1e-9) * (2 * SQ3)
    angles = np.zeros(N)
    return pos, angles


def construct():
    # Initial guess from spectral embedding
    pos, angles = _spectral_init()
    # Blend with lattice for a better start
    lp, la = _lattice_init()
    # scale lattice to similar magnitude
    lp = lp / (np.linalg.norm(lp) + 1e-9) * (np.linalg.norm(pos) + 1e-9)
    pos0 = 0.5 * pos + 0.5 * lp
    angles0 = np.zeros(N)
    orient0 = 0.0
    theta0 = np.concatenate([pos0.ravel(), angles0, [orient0]])

    best_theta = theta0.copy()
    best_val = _objective(theta0)

    # Nelder-Mead refinement
    try:
        res = minimize(_objective, theta0, method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_val:
            best_theta = res.x
            best_val = res.fun
    except Exception:
        pass

    pos = best_theta[:2 * N].reshape(N, 2)
    angles = best_theta[2 * N:2 * N + N]
    orient = best_theta[2 * N + N]

    # Compute the true required side (no penalty) for the side we report
    verts = _max_projection(pos, angles)
    side = _required_side(verts, orient)

    # Fallback safety: if side looks huge, use lattice directly with a generous container
    if side > 20.0:
        pos, angles = _lattice_init()
        orient = 0.0
        verts = _max_projection(pos, angles)
        side = _required_side(verts, orient) + 0.5

    inner = [[float(pos[i][0]), float(pos[i][1]), float(np.degrees(angles[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(side), float(np.degrees(orient))
