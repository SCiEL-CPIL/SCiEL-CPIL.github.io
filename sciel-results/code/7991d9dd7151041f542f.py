"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 12
DEADLINE = 130.0  # safety margin before the 150s kill

# EVOLVE-BLOCK-START

def _hex_vertices(theta):
    # flat-top regular hexagon, side 1, circumradius 1, rotated by theta
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _max_support(pts, theta):
    # max over 6 edge-normal directions of the support function
    norms = np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                       math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    return float(np.max(pts @ norms.T))

def _bounding_side(pts, theta):
    # side length of the smallest regular hexagon (orientation theta) containing pts
    norms = np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                       math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    P = pts @ norms.T
    return 2.0 * float(np.max(P - P.min(axis=0))) / math.sqrt(3.0)

def _hexagon_overlap(A, B):
    # SAT test between two convex polygons (arrays of (x,y)); True if overlapping
    for poly in (A, B):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            projA = A @ normal
            projB = B @ normal
            if projA.max() < projB.min() or projB.max() < projA.min():
                return False
    return True

def _valid(inner, outer, side, outer_theta):
    # check no overlaps and all inner hexagons inside the outer hexagon
    centers = np.array([[r[0], r[1]] for r in inner])
    verts = []
    for r in inner:
        verts.append(_hex_vertices(math.radians(r[2])))
    for i in range(N):
        for j in range(i + 1, N):
            if _hexagon_overlap(verts[i], verts[j]):
                return False
    outer_pts = outer * side
    for i in range(N):
        pts = verts[i] + centers[i]
        if _bounding_side(pts - outer, outer_theta) > side + 1e-6:
            return False
    return True

def _score(inner, outer, side, outer_theta):
    centers = np.array([[r[0], r[1]] for r in inner])
    verts = [_hex_vertices(math.radians(r[2])) for r in inner]
    # penalty for overlaps
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if _hexagon_overlap(verts[i], verts[j]):
                pen += 1.0
    # penalty for containment
    outer_pts = outer * side
    for i in range(N):
        pts = verts[i] + centers[i]
        b = _bounding_side(pts - outer, outer_theta)
        if b > side + 1e-6:
            pen += 1.0
    return pen

def _seed():
    # compact triangular-lattice 4x3 block of unit hexagons
    d = 2.0 * math.sqrt(3.0) + 1e-6
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    pts = []
    for i in range(-1, 3):
        for j in range(-1, 2):
            pts.append(i * a1 + j * a2)
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return [[float(pts[k][0]), float(pts[k][1]), 0.0] for k in range(N)]

def _simulate(inner, outer, side, outer_theta, dt=0.02, steps=4000):
    centers = np.array([[r[0], r[1]] for r in inner], dtype=float)
    angles = np.array([math.radians(r[2]) for r in inner])
    vel = np.zeros((N, 2))
    omega = np.zeros(N)
    outer_pts = np.array(outer, dtype=float) * side
    verts = [_hex_vertices(angles[k]) for k in range(N)]

    for s in range(steps):
        if time.time() > DEADLINE:
            break
        # ---- forces ----
        for i in range(N):
            # attraction toward outer incenter
            vel[i] += -0.02 * (centers[i] - outer_pts)
            omega[i] += -0.005 * angles[i]  # gentle alignment toward 0
        # pairwise repulsion via SAT minimum translation vector
        for i in range(N):
            for j in range(i + 1, N):
                A = verts[i] + centers[i]
                B = verts[j] + centers[j]
                if _hexagon_overlap(A, B):
                    # find the separating axis that currently gives least separation
                    best_pen = -1.0
                    best_axis = None
                    for poly in (A, B):
                        for k in range(6):
                            p1 = poly[k]
                            p2 = poly[(k + 1) % 6]
                            e = p2 - p1
                            n = np.array([-e[1], e[0]])
                            nl = math.hypot(n[0], n[1])
                            if nl < 1e-9:
                                continue
                            n = n / nl
                            pa = A @ n
                            pb = B @ n
                            # separation (positive if disjoint)
                            sep = min(pa.max() - pb.min(), pb.max() - pa.min())
                            if sep < 0 and sep > best_pen:
                                best_pen = sep
                                best_axis = n
                    if best_axis is not None:
                        f = 0.5 * (-best_pen)  # push apart
                        vel[i] += f * best_axis
                        vel[j] -= f * best_axis
        # ---- containment: push inward if a hexagon pokes out ----
        for i in range(N):
            pts = verts[i] + centers[i]
            P = pts @ np.array([[math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                                 math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)]).T
            for k in range(6):
                n = np.array([math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                              math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)])
                exceed = P.max(axis=0)[k] - P.min(axis=0)[k] - side * math.sqrt(3.0)
                if exceed > 0:
                    # which vertex is farthest in +n
                    idx = np.argmax(pts @ n)
                    vel[i] -= 0.5 * exceed * n
        # ---- integrate (symplectic Euler) ----
        vel *= 0.85
        omega *= 0.85
        centers += vel * dt
        angles += omega * dt
        for k in range(N):
            verts[k] = _hex_vertices(angles[k])

    return [[float(centers[k][0]), float(centers[k][1]), float(math.degrees(angles[k]))] for k in range(N)]

def _refine_outer(inner, outer, side, outer_theta):
    centers = np.array([[r[0], r[1]] for r in inner])
    verts = [_hex_vertices(math.radians(r[2])) for r in inner]
    allpts = []
    for k in range(N):
        allpts.append(verts[k] + centers[k])
    allpts = np.vstack(allpts)
    best = (side, outer, outer_theta)
    for ang in np.linspace(0.0, math.pi / 3.0, 121):
        b = _bounding_side(allpts, ang)
        # re-center outer so bounding hexagon is centered
        norms = np.array([[math.cos(ang + math.pi / 6.0 + k * math.pi / 3.0),
                           math.sin(ang + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)]).T
        P = allpts @ norms
        mid = 0.5 * (P.max(axis=0) + P.min(axis=0))
        newc = (allpts.mean(axis=0) + (mid - P.mean(axis=0)) @ np.linalg.pinv(norms))
        if b < best[0] - 1e-9:
            best = (b, tuple(float(x) for x in newc), float(ang))
    return best

def construct():
    start = time.time()
    inner = _seed()
    # initial outer estimate
    centers = np.array([[r[0], r[1]] for r in inner])
    side0, outer0, theta0 = _refine_outer(inner, (0.0, 0.0), 10.0, 0.0)

    best_side = side0
    best_inner = [list(r) for r in inner]
    best_outer = outer0
    best_theta = theta0

    # a few short physics relaxation rounds from the compact seed
    for rnd in range(3):
        if time.time() > DEADLINE:
            break
        relaxed = _simulate(inner, outer0, best_side, best_theta)
        # re-refine outer for the relaxed config
        s, c, t = _refine_outer(relaxed, outer0, best_side, best_theta)
        if s < best_side - 1e-9:
            best_side = s
            best_inner = relaxed
            best_outer = c
            best_theta = t
            inner = relaxed
            outer0 = c
            theta0 = t

    # final fine outer-angle polish on the best config
    s, c, t = _refine_outer(best_inner, best_outer, best_side, best_theta)
    if s < best_side:
        best_side = s
        best_outer = c
        best_theta = t

    return best_inner, best_outer, best_side, math.degrees(best_theta)
# EVOLVE-BLOCK-END
