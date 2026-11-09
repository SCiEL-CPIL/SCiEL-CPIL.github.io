"""Packing 12 unit hexagons inside a regular hexagon (outer side minimized).

Strategy
--------
1. Candidate inner centers = the 19 centers of a side-3 (radius-2) flat-top
   hexagonal lattice.  Adjacent centers are distance sqrt(3) apart, so unit
   hexagons placed on them are exactly touching (a valid packing).
2. The OUTER hexagon orientation is a free degree of freedom (the task lets it
   rotate).  For a given outer orientation phi, every inner hexagon's optimal
   rotation depends only on its position, and the required outer side is the
   max over the chosen centers.
3. Disjointness is verified rigorously with the Minkowski-sum / separating-axis
   (SAT) metric: two unit hexagons are disjoint iff
       max_{n in N(t1) u N(t2)}  n.(p1-p2) - (sup(t1,n)+sup(t2,-n))  >=  0.
4. We scan the outer orientation phi in [0,60) (6-fold symmetry), pick the
   12-center subset that minimizes the required outer side, then refine phi with
   Nelder-Mead (re-selecting the subset at every evaluation) and polish each
   inner rotation.
"""
import time
import math
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = t0 + 135.0          # safety margin before the 150 s kill

    # ------------------------------------------------------------------
    # Geometry helpers (flat-top unit hexagon: vertex at angle 0 -> pointy axis
    # along +x, flat top/bottom; circumradius = 1, inradius = sqrt(3)/2).
    # ------------------------------------------------------------------
    def hex_vertices(cx, cy, theta):
        a = theta + np.arange(6) * (math.pi / 3.0)
        return np.column_stack((cx + np.cos(a), cy + np.sin(a)))

    def hex_normals(theta):
        a = theta + (math.pi / 6.0) + np.arange(6) * (math.pi / 3.0)
        return np.column_stack((np.cos(a), np.sin(a)))

    def support(theta, n):
        v = np.column_stack((
            np.cos(theta + np.arange(6) * (math.pi / 3.0)),
            np.sin(theta + np.arange(6) * (math.pi / 3.0)),
        ))
        return float(np.max(v @ n))

    # ------------------------------------------------------------------
    # Candidate centers: 19 centers of a radius-2 flat-top hexagonal lattice.
    # ------------------------------------------------------------------
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                centers.append((1.5 * q, SQRT3 * (r + q / 2.0)))
    centers = np.array(centers)          # (19, 2)

    def outer_normals(phi):
        a = phi + (math.pi / 6.0) + np.arange(6) * (math.pi / 3.0)
        return np.column_stack((np.cos(a), np.sin(a)))

    def evaluate(phi, rot_res=0.05):
        """Per-center optimal rotation + required outer side, best 12-subset."""
        ON = outer_normals(phi)
        theta_best = np.zeros(19)
        L_best = np.full(19, np.inf)
        for i in range(19):
            p = centers[i]
            rad = math.atan2(p[1], p[0])
            best_L = np.inf
            best_t = rad
            for t in np.arange(rad - math.pi / 6.0, rad + math.pi / 6.0 + rot_res, rot_res):
                proj = hex_vertices(p[0], p[1], t) @ ON.T
                L = float(np.max(proj, axis=0) / (SQRT3 / 2.0))
                if L < best_L:
                    best_L, best_t = L, t
            theta_best[i] = best_t
            L_best[i] = best_L
        idx = min(combinations(range(19), 12), key=lambda s: max(L_best[j] for j in s))
        return float(max(L_best[j] for j in idx)), idx, theta_best

    def sat_disjoint(pos, theta, tol=1e-9):
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                d = pos[a] - pos[b]
                n = np.vstack((hex_normals(theta[a]), hex_normals(theta[b])))
                supa = np.array([support(theta[a], n[k]) for k in range(len(n))])
                supb = np.array([support(theta[b], -n[k]) for k in range(len(n))])
                if float(np.max(n @ d - (supa + supb))) < tol:
                    return False
        return True

    # ------------------------------------------------------------------
    # 1) Coarse scan of the outer orientation.
    # ------------------------------------------------------------------
    best_L = np.inf
    best_phi = 0.0
    best_idx = tuple(range(12))
    best_theta = None
    for deg in range(0, 60, 3):
        if time.time() > DEADLINE:
            break
        phi = math.radians(deg)
        L, idx, th = evaluate(phi, rot_res=0.05)
        if L < best_L:
            best_L, best_phi, best_idx, best_theta = L, phi, idx, th

    # ------------------------------------------------------------------
    # 2) Nelder-Mead refinement of the outer orientation.
    # ------------------------------------------------------------------
    def objective(phi_arr):
        if time.time() > DEADLINE:
            return best_L
        L, idx, th = evaluate(phi_arr[0], rot_res=0.05)
        if L < best_L:
            best_L, best_phi, best_idx, best_theta = L, float(phi_arr[0]), idx, th
        return L

    if time.time() < DEADLINE - 20:
        x0 = np.array([best_phi])
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 1e-5, "fatol": 1e-7,
                                "adaptive": True})
        if res.x is not None and time.time() < DEADLINE - 20:
            L, idx, th = evaluate(float(res.x[0]), rot_res=0.05)
            if L < best_L:
                best_L, best_phi, best_idx, best_theta = L, float(res.x[0]), idx, th

    # ------------------------------------------------------------------
    # 3) Polish each inner rotation with a fine local search.
    # ------------------------------------------------------------------
    if time.time() < DEADLINE - 10:
        ON = outer_normals(best_phi)
        pos = centers[best_idx]
        theta = [float(best_theta[i]) for i in best_idx]
        for _ in range(3):
            if time.time() > DEADLINE - 10:
                break
            changed = False
            for k in range(len(pos)):
                p = pos[k]
                base = theta[k]
                cur_L = float(np.max(hex_vertices(p[0], p[1], base) @ ON.T, axis=0) / (SQRT3 / 2.0))
                for t in np.arange(base - 0.05, base + 0.05 + 0.002, 0.002):
                    L = float(np.max(hex_vertices(p[0], p[1], t) @ ON.T, axis=0) / (SQRT3 / 2.0))
                    if L < cur_L - 1e-10:
                        cur_L, theta[k], changed = L, float(t), True
            if not changed:
                break
        best_theta = np.array(best_theta)
        for k, i in enumerate(best_idx):
            best_theta[i] = theta[k]

    # ------------------------------------------------------------------
    # 4) Build placement; tiny scale about origin to make touching hexes strict.
    # ------------------------------------------------------------------
    pos = centers[best_idx]
    theta = np.array(best_theta[best_idx])

    scale = 1.0
    for s in (1.0, 0.9999, 0.9998, 0.9997, 0.9995, 0.999):
        ps = pos * s
        if sat_disjoint(ps, theta, tol=1e-9):
            scale = s
            break

    inner = []
    for k in range(len(pos)):
        x, y = pos[k] * scale
        inner.append([float(x), float(y), float(math.degrees(theta[k]))])

    ON = outer_normals(best_phi)
    final_L = 0.0
    for k in range(len(pos)):
        x, y = pos[k] * scale
        L = float(np.max(hex_vertices(x, y, theta[k]) @ ON.T, axis=0) / (SQRT3 / 2.0))
        final_L = max(final_L, L)

    return inner, (0.0, 0.0), float(final_L), float(math.degrees(best_phi))
# EVOLVE-BLOCK-END
