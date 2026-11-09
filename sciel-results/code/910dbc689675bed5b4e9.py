# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def hexagon_packing_11():
    """
    Pack 11 disjoint unit regular hexagons inside the smallest enclosing regular hexagon.
    Full 33-DOF (x,y,angle per inner hex) optimized with L-BFGS-B on a penalty objective.
    """
    t0 = time.time()
    DEADLINE = 600.0
    MARGIN = 30.0

    def remaining():
        return DEADLINE - MARGIN - (time.time() - t0)

    # ---------- geometry helpers ----------
    def hex_vertices(cx, cy, s, a_deg):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60 * np.arange(6))
        return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

    def hex_edge_normals(verts):
        # outward unit normals of the 6 edges (convex, CCW)
        e = np.roll(verts, -1, axis=0) - verts
        e_len = np.linalg.norm(e, axis=1)
        e = e / e_len[:, None]
        n = np.column_stack((-e[:, 1], e[:, 0]))
        # ensure outward: normal dot (edge_mid - center) > 0
        mids = (verts + np.roll(verts, -1, axis=0)) / 2.0
        c = verts.mean(axis=0)
        d = mids - c
        flip = np.sum(n * d, axis=1) < 0
        n[flip] *= -1.0
        return n

    def enclosing_side(inner_verts_list):
        # min s such that all inner vertices lie inside outer hex (center 0, angle 0, side s)
        # outer half-width (apothem) = s * sqrt(3)/2 ; outer vertices at radius s.
        # For a regular hexagon centered at origin with flat-top orientation (angle 0 -> vertex at +x),
        # the 6 outward support constraints are: |dot(p, n_k)| <= s * sqrt(3)/2 for 6 normals at 0,60,...,300 deg.
        # Actually for vertex-at-0 orientation, outward normals are at 30,90,...,330 deg.
        # Let's just compute via: for each of 6 edge normals (at 30+60k), require p.n <= s*sqrt(3)/2.
        normals = np.column_stack((np.cos(np.deg2rad(30 + 60 * np.arange(6))),
                                   np.sin(np.deg2rad(30 + 60 * np.arange(6)))))
        max_proj = 0.0
        for v in inner_verts_list:
            proj = np.max(np.abs(v @ normals.T), axis=1)  # (6,)
            max_proj = max(max_proj, float(np.max(proj)))
        return 2.0 * max_proj / np.sqrt(3.0)

    def separation_violation(v1, n1, v2, n2):
        # For each of the 12 normals (6 of each hex), check if projected intervals are apart.
        # If for ALL 12 normals the intervals overlap (or touch within tol), it's a violation.
        # We want a penalty that is 0 when separated, positive when overlapping.
        tol = 1e-6
        for n in list(n1) + list(n2):
            p1 = v1 @ n
            p2 = v2 @ n
            lo1, hi1 = p1.min(), p1.max()
            lo2, hi2 = p2.min(), p2.max()
            gap = max(lo1, lo2) - min(hi1, hi2)
            if gap > tol:
                return 0.0  # separated on this normal
        # overlapping on all 12 normals -> compute penetration depth as penalty
        # use max over normals of (min(hi1,hi2) - max(lo1,lo2))
        pen = 0.0
        for n in list(n1) + list(n2):
            p1 = v1 @ n
            p2 = v2 @ n
            lo1, hi1 = p1.min(), p1.max()
            lo2, hi2 = p2.min(), p2.max()
            overlap = min(hi1, hi2) - max(lo1, lo2)
            if overlap > pen:
                pen = overlap
        return pen

    def build_and_evaluate(params, penalty_w=1000.0, center_penalty_w=100.0):
        # params: (33,) -> 11 * (x, y, angle_deg)
        n = 11
        centers = params[:2 * n].reshape(n, 2)
        angles = params[2 * n:]
        verts_list = []
        normals_list = []
        for i in range(n):
            v = hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
            verts_list.append(v)
            normals_list.append(hex_edge_normals(v))
        # re-center inner set around origin (outer center is origin)
        all_c = centers.mean(axis=0)
        shifted_verts = [v - all_c for v in verts_list]
        # side length
        side = enclosing_side(shifted_verts)
        # separation penalty
        sep_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                sep_pen += separation_violation(shifted_verts[i], normals_list[i],
                                                shifted_verts[j], normals_list[j])
        # keep centers near origin to avoid drift (helps L-BFGS)
        center_pen = np.sum(centers ** 2)
        obj = side + penalty_w * sep_pen + center_penalty_w * center_pen
        return obj, side, sep_pen, centers, angles

    # ---------- initial guess: dense hex lattice ----------
    # hex lattice with spacing 2 (touching) for 11 hexes
    # positions in axial coords (q,r) -> x = 2*(q + r/2), y = 2*sqrt(3)/2*r = sqrt(3)*r
    axial = [(0, 0),
             (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
             (2, 0), (-2, 0), (0, 2), (0, -2)]
    init_centers = np.zeros((11, 2))
    for i, (q, r) in enumerate(axial):
        init_centers[i, 0] = 2.0 * (q + r / 2.0)
        init_centers[i, 1] = np.sqrt(3.0) * r
    init_angles = np.zeros(11)
    init_params = np.concatenate([init_centers.ravel(), init_angles])

    best_params = init_params.copy()
    best_side, best_sep = np.inf, np.inf
    _, s0, sp0, _, _ = build_and_evaluate(best_params)
    best_side, best_sep = s0, sp0

    # ---------- L-BFGS-B optimization ----------
    def objective(p):
        obj, side, sep, _, _ = build_and_evaluate(p)
        return obj

    # gradient via finite differences would be slow; use L-BFGS-B with numerical grad
    # To speed up, we can provide an analytic-ish gradient via finite diff with small step.
    # L-BFGS-B does its own finite diff; just call it.

    # We'll run a sequence of L-BFGS-B calls with increasing penalty weight.
    for pw in [100.0, 1000.0, 10000.0]:
        if remaining() < 15.0:
            break

        def obj_pw(p):
            obj, side, sep, _, _ = build_and_evaluate(p, penalty_w=pw)
            return obj

        res = minimize(obj_pw, best_params, method='L-BFGS-B',
                       options={'maxiter': 200, 'maxls': 50,
                                'ftol': 1e-10, 'gtol': 1e-8,
                                'maxfun': 5000})
        if remaining() < 15.0:
            break
        if res.x is not None and np.all(np.isfinite(res.x)):
            _, side, sep, _, _ = build_and_evaluate(res.x)
            if side < best_side and sep < 1e-4:
                best_params = res.x
                best_side, best_sep = side, sep
            elif side < best_side * 1.05:
                # accept even if slightly worse side but better separation
                _, s2, sp2, _, _ = build_and_evaluate(best_params)
                if sep < sp2:
                    best_params = res.x
                    best_side, best_sep = side, sep

    # ---------- final polish: local Nelder-Mead on side only ----------
    if remaining() > 30.0:
        def side_only(p):
            obj, side, sep, _, _ = build_and_evaluate(p, penalty_w=1e6)
            return side + 1e6 * sep

        res2 = minimize(side_only, best_params, method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if res2.x is not None and np.all(np.isfinite(res2.x)):
            _, side, sep, _, _ = build_and_evaluate(res2.x)
            if side < best_side and sep < 1e-6:
                best_params = res2.x
                best_side, best_sep = side, sep

    # ---------- construct final result ----------
    n = 11
    centers = best_params[:2 * n].reshape(n, 2)
    angles = best_params[2 * n:]
    # re-center around origin
    all_c = centers.mean(axis=0)
    centers = centers - all_c

    inner = np.column_stack([centers[:, 0], centers[:, 1], angles])
    outer = np.array([0.0, 0.0, 0.0])
    side = best_side

    # sanity: ensure side is at least sqrt(11)
    if side < np.sqrt(11.0):
        side = np.sqrt(11.0)

    return inner, outer, float(side)


# EVOLVE-BLOCK-END
